// Per-frame GPU culling + LOD selection. One thread per tree: frustum test, pick a
// LOD from distance, then append the tree index into that (species, LOD) bucket and
// bump the bucket's indirect instance count. The CPU never looks at individual trees.

struct Cull {
    planes: array<vec4f, 6>,
    cam: vec4f,                 // xyz LOD reference position, w = max draw distance
    lod: vec4f,                 // x,y,z LOD switch distances, w = LOD bias
    counts: vec4u,              // x = tree count
    regions: array<vec4u, 12>,  // x = first slot in visible[], y = capacity
    bounds: array<vec4f, 4>,    // per species: x = sphere centre height, y = radius
}

@group(1) @binding(0) var<uniform> C: Cull;
@group(1) @binding(1) var<storage, read> trees: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> visible: array<u32>;

fn tree_scale(packed: u32) -> f32 {
    return 0.75 + f32((packed >> 10u) & 255u) / 255.0 * 0.55;
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
    for (var k = 0; k < 6; k++) {
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
    let draw = species * 4u + lod;
    let reg = C.regions[draw];
    let slot = atomicAdd(&args[draw * 5u + 1u], 1u);
    if (slot >= reg.y) {
        atomicSub(&args[draw * 5u + 1u], 1u);
        return;
    }
    visible[reg.x + slot] = i;
}
