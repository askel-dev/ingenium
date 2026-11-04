// Per-frame GPU culling + LOD selection. One thread per tree: frustum test, pick a
// LOD from distance, then append the tree index into that (species, LOD) bucket and
// bump the bucket's indirect instance count. The CPU never looks at individual trees.
//
// The camera culls in two passes. The first appends only the near LODs, which are drawn
// first as occluders; the second, after a depth pyramid has been built from them and the
// terrain, appends the distant LODs that aren't hidden behind them.

const MODE_ALL = 0u;        // every LOD, no occlusion test (the shadow cascades)
const MODE_OCCLUDERS = 1u;  // only the LODs below counts.z
const MODE_OCCLUDED = 2u;   // only the LODs from counts.z up, if not hidden in the depth pyramid

struct Cull {
    planes: array<vec4f, 8>,   // unused ones are (0, 0, 0, 1)
    cam: vec4f,                 // xyz LOD reference position, w = max draw distance
    lod: vec4f,                 // x,y,z LOD switch distances, w = LOD bias
    counts: vec4u,              // x = tree count, y = mode, z = LODs drawn as occluders
    regions: array<vec4u, 12>,  // x = first slot in visible[], y = capacity
    bounds: array<vec4f, 4>,    // per species: x = sphere centre height, y = radius
    view: mat4x4f,              // the camera, for the occlusion test
    proj: vec4f,                // x, y = the projection's x and y scale, z = near plane
    hiz: vec4f,                 // depth pyramid: screen width and height in pixels, levels
}

@group(1) @binding(0) var<uniform> C: Cull;
@group(1) @binding(1) var<storage, read> trees: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> visible: array<u32>;
@group(1) @binding(4) var hiz: texture_2d<f32>;

fn tree_scale(packed: u32) -> f32 {
    return 0.75 + f32((packed >> 10u) & 255u) / 255.0 * 0.55;
}

// Whether a sphere is hidden behind everything the depth pyramid holds: its nearest point is
// farther away than the farthest depth over the box it covers on screen (reversed Z: 1 near).
fn occluded(center: vec3f, radius: f32) -> bool {
    let v = (C.view * vec4f(center, 1.0)).xyz;
    let c = vec3f(v.xy, -v.z); // z = distance in front of the camera
    let near = C.proj.z;
    if (c.z <= radius + near) {
        return false;
    }
    // the exact screen box of the projected sphere (Mara and McGuire 2013, "2D Polyhedral
    // Bounds of a Clipped, Perspective-Projected 3D Sphere")
    let cr = c * radius;
    let czr2 = c.z * c.z - radius * radius;
    let vx = sqrt(c.x * c.x + czr2);
    let vy = sqrt(c.y * c.y + czr2);
    let ndc_lo = vec2f((vx * c.x - cr.z) / (vx * c.z + cr.x), (vy * c.y - cr.z) / (vy * c.z + cr.y)) * C.proj.xy;
    let ndc_hi = vec2f((vx * c.x + cr.z) / (vx * c.z - cr.x), (vy * c.y + cr.z) / (vy * c.z - cr.y)) * C.proj.xy;
    // in pixels (NDC y points up, pixel rows down)
    let lo = clamp(vec2f(ndc_lo.x, -ndc_hi.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * C.hiz.xy;
    let hi = clamp(vec2f(ndc_hi.x, -ndc_lo.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * C.hiz.xy;
    // The level where the box spans at most four texels each way (a level L texel is
    // 2^(L+1) pixels wide). A coarser one would be fewer reads, but a texel is as far as the
    // farthest sample under it, and through foliage a gap is never far away: the finer the
    // texels, the fewer gaps a tree is tested against that it doesn't actually cover.
    let levels = u32(C.hiz.z);
    let span = (hi - lo) * 0.5;
    var level = u32(clamp(ceil(log2(max(max(span.x, span.y), 1.0))) - 2.0, 0.0, f32(levels - 1u)));
    var a: vec2u;
    var b: vec2u;
    loop {
        let last = (vec2u(C.hiz.xy) - 1u) >> vec2u(level + 1u); // the level's last texel
        a = min(vec2u(lo) >> vec2u(level + 1u), last);
        b = min(vec2u(hi) >> vec2u(level + 1u), last);
        if (all(b - a <= vec2u(4u)) || level + 1u >= levels) {
            break;
        }
        level += 1u;
    }
    let depth = near / (c.z - radius);
    for (var y = a.y; y <= b.y; y++) {
        for (var x = a.x; x <= b.x; x++) {
            if (textureLoad(hiz, vec2u(x, y), level).r <= depth) {
                return false;
            }
        }
    }
    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= C.counts.x) {
        return;
    }
    let t = trees[i];
    let pos = bitcast<vec3f>(t.xyz);
    let packed = t.w;
    let species = packed & 3u;
    let scale = tree_scale(packed);
    let b = C.bounds[species];
    let center = pos + vec3f(0.0, b.x * scale, 0.0);
    let radius = b.y * scale;
    for (var k = 0; k < 8; k++) {
        let pl = C.planes[k];
        if (dot(pl.xyz, center) + pl.w < -radius) {
            return;
        }
    }
    let dist = distance(C.cam.xyz, center);
    // thin trees out over the last 20% of the range instead of a hard cut-off line
    let cutoff = C.cam.w * (0.8 + 0.2 * u01(pcg(packed ^ 0x5bd1e995u)));
    if (dist - radius > cutoff) {
        return;
    }
    let jitter = 1.0 + (u01(pcg(packed)) - 0.5) * 0.25;
    let d = dist / scale * jitter;
    var lod = 3u;
    if (d < C.lod.x) {
        lod = 0u;
    } else if (d < C.lod.y) {
        lod = 1u;
    } else if (d < C.lod.z) {
        lod = 2u;
    }
    lod = min(lod + u32(C.lod.w), 3u);
    let mode = C.counts.y;
    if (mode == MODE_OCCLUDERS && lod >= C.counts.z) {
        return;
    }
    if (mode == MODE_OCCLUDED && (lod < C.counts.z || occluded(center, radius))) {
        return;
    }
    let draw = species * 4u + lod;
    let reg = C.regions[draw];
    let slot = atomicAdd(&args[draw * 5u + 1u], 1u);
    if (slot >= reg.y) {
        atomicSub(&args[draw * 5u + 1u], 1u);
        return;
    }
    visible[reg.x + slot] = i;
}
