// Per-frame placement of the forest-floor plants around the camera (undergrowth.rs). One
// cell of a world-anchored grid holds at most one plant; which kind grows there follows
// patches of noise (ferns in drifts, ground cover in mats, the odd shrub), more of all of
// it where the canopy opens up. Records have the tree layout (scatter.wgsl), so the tree
// shader draws them: bits 0-1 the leaf texture layer, 2-9 rotation, 10-17 scale, 18-31 seed.
// (Their meshes carry no tree shape, so only the record's layout matters.)

struct Params {
    planes: array<vec4f, 6>,
    origin: vec4f,             // x, z of the grid's first cell, cell size, radius
    dims: vec4u,               // cells per side, capacity per kind
    kinds: array<vec4f, 3>,    // x = texture layer, y = bounding radius
}

@group(1) @binding(0) var<uniform> U: Params;
@group(1) @binding(1) var<storage, read_write> plants: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>>;

const FERN = 0u;
const SHRUB = 1u;
const COVER = 2u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dim = U.dims.x;
    if (id.x >= dim || id.y >= dim) {
        return;
    }
    let cs = U.origin.z;
    let radius = U.origin.w;
    let cell = vec2i(floor(U.origin.xy / cs + 0.5)) + vec2i(id.xy);
    let h0 = hash_i2(cell + vec2i(7919, -3571));
    let h1 = pcg(h0);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let h4 = pcg(h3);
    let h5 = pcg(h4);
    let p = (vec2f(cell) + 0.1 + 0.8 * vec2f(u01(h0), u01(h1))) * cs;
    let d = distance(p, F.cam_pos.xz);
    if (d > radius) {
        return;
    }
    let dat = data_at(p);
    let forest = smoothstep(0.3, 0.65, dat.x) * (1.0 - smoothstep(0.3, 0.6, dat.y));
    if (forest <= 0.0 || normal_at(p).y < 0.82) {
        return;
    }
    // drifts of fern, mats of ground cover, shrubs scattered and in thickets
    let fern = smoothstep(-0.15, 0.35, gnoise(p * 0.045 + vec2f(11.0, -4.0))) * 0.55;
    let cover = smoothstep(-0.05, 0.4, gnoise(p * 0.08 + vec2f(-6.0, 2.0))) * 0.45;
    let shrub = 0.03 + 0.12 * smoothstep(0.2, 0.6, gnoise(p * 0.03 + vec2f(3.0, 8.0)));
    let r = u01(h2) / forest;
    var kind = FERN;
    if (r < fern) {
        kind = FERN;
    } else if (r < fern + shrub) {
        kind = SHRUB;
    } else if (r < fern + shrub + cover) {
        kind = COVER;
    } else {
        return;
    }
    // thin out towards the edge of the grid, so plants fade in instead of appearing on a line
    if (u01(h3) > 1.0 - smoothstep(radius * 0.65, radius, d)) {
        return;
    }
    // keep the camera's head out of the shrubs (the player walks through them)
    if (kind == SHRUB && d < 2.5) {
        return;
    }
    let info = U.kinds[kind];
    let y = ground_h(p) - 0.04;
    let center = vec3f(p.x, y + info.y * 0.5, p.y);
    for (var i = 0; i < 6; i++) {
        let pl = U.planes[i];
        if (dot(pl.xyz, center) + pl.w < -info.y) {
            return;
        }
    }
    let rot = h4 & 255u;
    let scale_q = tree_scale_bits(mix(0.75, 1.3, u01(pcg(h4))));
    let seed = (h5 >> 8u) & 0x3FFFu;
    let packed = u32(info.x) | (rot << 2u) | (scale_q << 10u) | (seed << 18u);
    let slot = atomicAdd(&args[kind * 5u + 1u], 1u);
    if (slot >= U.dims.y) {
        atomicSub(&args[kind * 5u + 1u], 1u);
        return;
    }
    plants[kind * U.dims.y + slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(y), bitcast<u32>(p.y), packed);
}
