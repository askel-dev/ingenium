// Per-frame placement of the forest-floor plants around the camera (undergrowth.rs). One
// cell of a world-anchored grid holds at most one plant; which kind grows there follows
// patches of noise (ferns in drifts, ground cover in mats, the odd shrub), more of all of
// it where the canopy opens up. Records have the tree layout (scatter.wgsl), so the tree
// shader draws them: bits 0-1 the species (bark and tint; the cards carry their own foliage
// atlas layer), 2-9 rotation, 10-17 scale, 18-19 variant (0), 20-31 seed.

struct Params {
    planes: array<vec4f, 6>,
    origin: vec4f,             // x, z of the grid's first cell, cell size, radius
    dims: vec4u,               // cells per side, capacity per kind
    kinds: array<vec4f, 8>,    // x = species bits, y = bounding radius
}

@group(1) @binding(0) var<uniform> U: Params;
@group(1) @binding(1) var<storage, read_write> plants: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>>;

const FERN = 0u;
const SHRUB = 1u;
const COVER = 2u;
const LOG = 3u;
const BRANCH = 4u;
const STUMP = 5u;
const ROCK = 6u;
const SLAB = 7u;

// How close (m) the segment a +- d comes to the nearest spot the forest scatter could have put
// a tree (scatter.wgsl: one candidate per cell, whether or not it grew), so that nothing lies
// through a trunk.
fn trunk_clearance(a: vec2f, d: vec2f) -> f32 {
    let cell = 5.5;
    let c0 = vec2i(floor(a / cell));
    var best = 1e9;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = c0 + vec2i(x, y);
            let h0 = hash_i2(c);
            let h1 = pcg(h0);
            let t = (vec2f(c) + 0.05 + vec2f(u01(h0), u01(h1)) * 0.9) * cell;
            let s = clamp(dot(t - a, d) / max(dot(d, d), 1e-6), -1.0, 1.0);
            best = min(best, distance(t, a + d * s));
        }
    }
    return best;
}

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
    let rot = h4 & 255u;
    // Clutter on the forest floor: fallen logs and branches, the odd stump, rocks in patches
    // (more where the ground is rocky). Rarer than the plants, so it rolls its own dice first.
    let ang = f32(rot) / 255.0 * TAU;
    let axis = vec2f(cos(ang), sin(ang));
    let rocky = smoothstep(0.1, 0.5, gnoise(p * 0.02 + vec2f(-13.0, 7.0))) + dat.y * 2.0;
    let c = u01(pcg(h5 ^ 0x2C1Bu)) / forest;
    let p_log = 0.003;
    let p_branch = 0.035;
    let p_stump = 0.0015;
    let p_rock = 0.002 + 0.012 * rocky;
    var kind = 99u;
    if (c < p_log) {
        kind = LOG;
    } else if (c < p_log + p_branch) {
        kind = BRANCH;
    } else if (c < p_log + p_branch + p_stump) {
        kind = STUMP;
    } else if (c < p_log + p_branch + p_stump + p_rock) {
        kind = select(ROCK, SLAB, (h5 & 1u) == 1u);
    }
    if (kind != 99u) {
        // the mesh lies along x before it is turned (tree_common.wgsl, `inst_vec`: x -> (cos, sin))
        // (half its length at the largest scale and stretch)
        let half = array<f32, 5>(3.0, 1.9, 0.0, 0.0, 0.0)[kind - LOG];
        let clear = array<f32, 5>(0.9, 0.45, 1.2, 1.6, 1.6)[kind - LOG];
        if (trunk_clearance(p, axis * half) < clear || normal_at(p).y < 0.9) {
            return;
        }
    } else {
    // drifts of fern, mats of ground cover, shrubs scattered and in thickets
    let fern = smoothstep(-0.15, 0.35, gnoise(p * 0.045 + vec2f(11.0, -4.0))) * 0.55;
    let cover = smoothstep(-0.05, 0.4, gnoise(p * 0.08 + vec2f(-6.0, 2.0))) * 0.45;
    let shrub = 0.03 + 0.12 * smoothstep(0.2, 0.6, gnoise(p * 0.03 + vec2f(3.0, 8.0)));
    let r = u01(h2) / forest;
    kind = FERN;
    if (r < fern) {
        kind = FERN;
    } else if (r < fern + shrub) {
        kind = SHRUB;
    } else if (r < fern + shrub + cover) {
        kind = COVER;
    } else {
        return;
    }
    // keep the camera's head out of the shrubs (the player walks through them)
    if (kind == SHRUB && d < 2.5) {
        return;
    }
    }
    // thin out towards the edge of the grid, so plants fade in instead of appearing on a line
    if (u01(h3) > 1.0 - smoothstep(radius * 0.65, radius, d)) {
        return;
    }
    var info = U.kinds[kind];
    if (kind >= LOG && kind <= STUMP) {
        // wood of what grows around: spruce high up, now and then a birch
        let alt = height_at(p) + fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3) * 140.0;
        info.x = select(select(0.0, 1.0, (h3 & 7u) == 0u), 2.0, alt > 315.0);
    }
    let y = ground_h(p) - 0.04;
    let center = vec3f(p.x, y + info.y * 0.5, p.y);
    for (var i = 0; i < 6; i++) {
        let pl = U.planes[i];
        if (dot(pl.xyz, center) + pl.w < -info.y) {
            return;
        }
    }
    let scale_q = (h4 >> 8u) & 255u;
    let seed = (h5 >> 8u) & 0xFFFu;
    let packed = u32(info.x) | (rot << 2u) | (scale_q << 10u) | (seed << 20u);
    let slot = atomicAdd(&args[kind * 5u + 1u], 1u);
    if (slot >= U.dims.y) {
        atomicSub(&args[kind * 5u + 1u], 1u);
        return;
    }
    plants[kind * U.dims.y + slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(y), bitcast<u32>(p.y), packed);
}
