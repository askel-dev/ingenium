// Per-frame meadow generation around the camera, in two world-anchored grids so nothing swims
// as the camera moves:
//   - tufts of grass blades (geometry, textured with scanned blades), with the odd flower stalk
//     or seed stalk among them, out to ~26 m;
//   - textured cards of whole scanned grass tufts, out to ~60 m, filling in the sward between
//     and beyond the blades until the ground's own colour takes over.
// Flowers and seed heads are cards on top of their stalks (grass.wgsl).

struct GrassParams {
    planes: array<vec4f, 6>,
    origin: vec4f,    // blade grid: x, z of the first cell, cell size, radius
    origin2: vec4f,   // tuft card grid: the same
    dims: vec4u,      // blade grid cells per side, blade capacity, card grid cells per side, card capacity
}

@group(1) @binding(0) var<uniform> G: GrassParams;
@group(1) @binding(1) var<storage, read_write> blades: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> cards: array<vec4f>;
// two draw_indirect argument blocks: blades (7 vertices each), cards (4 each)
@group(1) @binding(3) var<storage, read_write> args: array<atomic<u32>>;

// The blade strips are scanned blades scaled to this average (foliage.rs, MEAN_BLADE); a
// blade's tint is the colour it should have divided by it.
const MEAN_BLADE: vec3f = vec3f(0.06, 0.1, 0.025);

fn meadow_amount(dat: vec4f) -> f32 {
    return (1.0 - smoothstep(0.35, 0.75, dat.x)) * (1.0 - smoothstep(0.3, 0.6, dat.y));
}

fn in_view(c: vec3f, r: f32) -> bool {
    for (var i = 0; i < 6; i++) {
        let pl = G.planes[i];
        if (dot(pl.xyz, c) + pl.w < -r) {
            return false;
        }
    }
    return true;
}

// The grass colour at p, as the ground further out has it (meadow_tint), varied per blade:
// some blades brighter, the odd one dry.
// (A dry blade's strip is a scan of a dry blade: it brings its own colour.)
fn blade_tint(base: vec3f, rnd: f32, dry: f32) -> vec3f {
    let c = base * (1.3 + 0.5 * rnd);
    return mix(c / MEAN_BLADE, vec3f(1.0), dry);
}

fn push_card(c0: vec4f, c1: vec4f) {
    let slot = atomicAdd(&args[5], 1u);
    if (slot >= G.dims.w) {
        atomicSub(&args[5], 1u);
        return;
    }
    cards[slot * 2u] = c0;
    cards[slot * 2u + 1u] = c1;
}

// Returns the blade's index, or -1 when the buffer is full.
fn push_blade(b0: vec4f, b1: vec4f, b2: vec4f) -> i32 {
    let slot = atomicAdd(&args[1], 1u);
    if (slot >= G.dims.y) {
        atomicSub(&args[1], 1u);
        return -1;
    }
    blades[slot * 3u] = b0;
    blades[slot * 3u + 1u] = b1;
    blades[slot * 3u + 2u] = b2;
    return i32(slot);
}

// Flower kind at p, as flower_color() mixes them: 0 buttercup, 1 daisy, 2 clover.
fn flower_kind(p: vec2f) -> u32 {
    let k = gnoise(p * 0.03 + vec2f(-5.0, 9.0));
    if (k > 0.425) {
        return 2u;
    }
    if (k > 0.05) {
        return 1u;
    }
    return 0u;
}

@compute @workgroup_size(8, 8)
fn blades_main(@builtin(global_invocation_id) id: vec3u) {
    let dim = G.dims.x;
    if (id.x >= dim || id.y >= dim) {
        return;
    }
    let cs = G.origin.z;
    let radius = G.origin.w;
    let cell = vec2i(floor(G.origin.xy / cs + 0.5)) + vec2i(id.xy);
    let h0 = hash_i2(cell);
    let h1 = pcg(h0);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let p = (vec2f(cell) + vec2f(u01(h0), u01(h1))) * cs;
    let d = distance(p, F.cam_pos.xz);
    if (d > radius) {
        return;
    }
    // thin out towards the edge of the grid, where the tuft cards carry on
    if (u01(h2) > 1.0 - smoothstep(GRASS_NEAR, GRASS_FAR, d)) {
        return;
    }
    let dat = data_at(p);
    if (u01(h3) > meadow_amount(dat)) {
        return;
    }
    let y = ground_h(p);
    let wp = vec3f(p.x, y, p.y);
    if (!in_view(wp + vec3f(0.0, 0.4, 0.0), 1.0)) {
        return;
    }
    let h4 = pcg(h3);
    let h5 = pcg(h4);
    let h6 = pcg(h5);
    let h7 = pcg(h6);
    let nrm = normal_at(p);
    let base_tint = meadow_tint(p, dat, nrm, sky_vis(p));
    // taller grass in patches; the seed heads stand in the tall patches
    let tall = smoothstep(-0.25, 0.5, fbm(p * 0.06, 3));
    let d3 = distance(wp, F.cam_pos.xyz);
    // far blades get wider so the sward keeps its cover as they thin out
    let widen = 1.0 + smoothstep(4.0, radius, d3) * 2.2;
    let count = 4u + (h4 & 3u);
    let fan = u01(h5) * TAU;
    for (var k = 0u; k < count; k++) {
        let hk = pcg(h7 + k * 0x9E3779B9u);
        let hk2 = pcg(hk);
        let hk3 = pcg(hk2);
        let r = u01(hk);
        let off = vec2f(cos(f32(k) * 2.4 + fan), sin(f32(k) * 2.4 + fan)) * (0.015 + 0.05 * u01(hk2));
        let bp = vec3f(p.x + off.x, y - 0.01, p.y + off.y);
        var height = mix(0.16, 0.5, r * r) * mix(0.55, 1.6, tall);
        if (r > 0.94) {
            height *= 1.5;
        }
        let width = mix(0.011, 0.019, u01(hk3)) * widen;
        // blades lean out from the middle of their tuft
        let angle = f32(k) * 2.4 + fan + (u01(hk3) - 0.5) * 0.8;
        let dry = select(0.0, 1.0, u01(pcg(hk3)) < 0.03 + 0.05 * tall);
        var strip = f32(hk2 % 6u) + select(0.0, 8.0, (hk3 & 1u) == 1u);
        if (dry > 0.5) {
            strip = 6.0 + f32(hk2 & 1u) + select(0.0, 8.0, (hk3 & 2u) == 2u);
        }
        push_blade(vec4f(bp, angle), vec4f(height, width, 0.0, r), vec4f(blade_tint(base_tint, r, dry), strip));
    }
    // a flower among the grass, in the drifts the ground paints too
    let hf = pcg(h6);
    if (u01(hf) < dat.z * flower_drift(p) * 0.6) {
        let kind = flower_kind(p);
        let r = u01(pcg(hf));
        let height = select(mix(0.22, 0.42, r), mix(0.12, 0.26, r), kind == 2u) * mix(0.8, 1.2, tall);
        let bp = vec3f(p.x, y - 0.01, p.y);
        let i = push_blade(vec4f(bp, fan + 1.0), vec4f(height, 0.004, 1.0, r), vec4f(blade_tint(base_tint, 0.3, 0.0) * 0.8, f32(pcg(hf) % 6u)));
        if (i >= 0) {
            var cell_i = (pcg(hf + 7u) % 3u);
            if (kind == 2u) {
                cell_i = 6u + (pcg(hf + 7u) & 1u);
            } else if (kind == 1u) {
                cell_i = 3u + cell_i;
            }
            let size = mix(0.045, 0.065, u01(pcg(hf + 9u))) * (1.0 + smoothstep(6.0, radius, d3) * 1.5);
            push_card(vec4f(f32(i), 0.0, 0.0, size), vec4f(vec3f(mix(0.85, 1.1, r)), f32(cell_i + 16u)));
        }
    }
    // a grass going to seed, in the tall patches
    let hs = pcg(h7 ^ 0x5bd1e995u);
    if (u01(hs) < 0.04 + 0.3 * tall * tall) {
        let r = u01(pcg(hs));
        let height = mix(0.55, 0.95, r) * mix(0.8, 1.15, tall);
        let bp = vec3f(p.x - 0.02, y - 0.01, p.y + 0.02);
        let i = push_blade(vec4f(bp, fan + 2.0), vec4f(height, 0.005, 2.0, r), vec4f(blade_tint(base_tint, 0.5, 0.3), 6.0 + f32(hs & 1u)));
        if (i >= 0) {
            let size = mix(0.18, 0.3, u01(pcg(hs + 3u))) * (1.0 + smoothstep(8.0, radius, d3));
            push_card(vec4f(f32(i), 0.0, 0.0, size), vec4f(vec3f(mix(0.8, 1.1, r)), f32((pcg(hs + 5u) & 7u) + 32u)));
        }
    }
}

@compute @workgroup_size(8, 8)
fn cards_main(@builtin(global_invocation_id) id: vec3u) {
    let dim = G.dims.z;
    if (id.x >= dim || id.y >= dim) {
        return;
    }
    let cs = G.origin2.z;
    let radius = G.origin2.w;
    let cell = vec2i(floor(G.origin2.xy / cs + 0.5)) + vec2i(id.xy);
    let h0 = hash_i2(cell + vec2i(-7717, 3203));
    let h1 = pcg(h0);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let p = (vec2f(cell) + vec2f(u01(h0), u01(h1))) * cs;
    let d = distance(p, F.cam_pos.xz);
    if (d > radius || d < 1.2) {
        return;
    }
    if (u01(h2) > 1.0 - smoothstep(radius * 0.55, radius, d)) {
        return;
    }
    let dat = data_at(p);
    if (u01(h3) > meadow_amount(dat)) {
        return;
    }
    let y = ground_h(p);
    let h4 = pcg(h3);
    let h5 = pcg(h4);
    let tall = smoothstep(-0.25, 0.5, fbm(p * 0.06, 3));
    let d3 = distance(vec3f(p.x, y, p.y), F.cam_pos.xyz);
    // bigger further out, where there are fewer of them
    let size = mix(0.4, 0.65, u01(h4)) * mix(0.7, 1.3, tall) * (1.0 + smoothstep(10.0, radius, d3) * 1.2);
    // the tufts' roots (dark, dead leaves) go into the sward
    let wp = vec3f(p.x, y - size * 0.12, p.y);
    if (!in_view(wp + vec3f(0.0, size * 0.4, 0.0), size)) {
        return;
    }
    let r = u01(h5);
    let tint = blade_tint(meadow_tint(p, dat, normal_at(p), sky_vis(p)), r, 0.0);
    // (the tufts' scans are darker at the base than single blades, with dead leaves in them)
    push_card(vec4f(wp, size), vec4f(tint * 1.5, f32(h5 & 3u)));
}
