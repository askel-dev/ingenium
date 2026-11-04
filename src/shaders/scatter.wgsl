// One-time forest placement: at most one tree or shrub per grid cell. Writes a compact list.
//
// Tree record (vec4u): xyz = bitcast position, w = packed
//   bits 0-1 species, 2-9 rotation, 10-17 scale (tree_scale), 18-31 seed. The seed's low 3
//   bits pick the shape (treegen.rs): 0-3 forest-grown, 4-5 open-grown, 6-7 an edge shrub.
//
// What grows where:
//   - inside the forest, forest-grown trees, in stands of different ages: young stands of
//     small trees, old ones with emergents above the canopy, and trees held back below it;
//   - on its edges (open ground within ~17 m), open-grown trees whose crowns come down low,
//     with shrubs between them and in a band out into the meadow, so the forest steps down to
//     the grass instead of ending in a fence of bare trunks;
//   - out in the open, groups of open-grown trees ringed by shrubs, and the odd lone tree.

struct Scatter {
    cell: f32,
    dim: u32,
    cap: u32,
    pad: u32,
}

@group(1) @binding(0) var<uniform> S: Scatter;
@group(1) @binding(1) var<storage, read_write> trees: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;

const FOREST = 0u;
const OPEN = 4u;
const SHRUB = 6u;
const NONE = 99u;

// Ground that can carry trees at all: not steep, not rock.
fn ground_ok(p: vec2f) -> f32 {
    let n = normal_at(p);
    return (1.0 - smoothstep(0.22, 0.4, 1.0 - n.y)) * (1.0 - data_at(p).y * 0.85);
}

// Closed forest at p: where the world data says forest (from where the ground turns to
// forest floor), before the gaps where an old tree fell.
fn forest_at(p: vec2f) -> f32 {
    return smoothstep(0.3, 0.55, data_at(p).x) * ground_ok(p);
}

// Groups of trees out in the open: at most one per 140 m cell, in about a quarter of them,
// 7-24 m across with a ragged outline. x = 1 at the group's centre, 0 on its rim, below 0
// outside; y = the group's hash.
const GROUP_CELL: f32 = 140.0;

fn group_at(p: vec2f) -> vec2f {
    let gc = vec2i(floor(p / GROUP_CELL));
    var best = vec2f(-10.0, 0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = gc + vec2i(x, y);
            let h = hash_i2(c + vec2i(-7717, 3301));
            if (u01(h) > 0.27) {
                continue;
            }
            let h1 = pcg(h);
            let h2 = pcg(h1);
            let h3 = pcg(h2);
            let centre = (vec2f(c) + 0.2 + 0.6 * vec2f(u01(h1), u01(h2))) * GROUP_CELL;
            let radius = mix(7.0, 24.0, u01(h3) * u01(h3));
            let reach = 1.0 - distance(p, centre) / radius + gnoise(p * 0.11 + vec2f(f32(h3 & 255u))) * 0.3;
            if (reach > best.x) {
                best = vec2f(reach, f32(h3 >> 8u));
            }
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= S.dim || id.y >= S.dim) {
        return;
    }
    let cell = vec2i(id.xy);
    let h0 = hash_i2(cell);
    let h1 = pcg(h0);
    let h2 = pcg(h1);
    let h3 = pcg(h2);
    let h4 = pcg(h3);
    let h5 = pcg(h4);
    let h6 = pcg(h5);
    let h7 = pcg(h6);
    let h8 = pcg(h7);
    let h9 = pcg(h8);
    let p = (vec2f(cell) + 0.05 + vec2f(u01(h0), u01(h1)) * 0.9) * S.cell;
    if (p.x < 8.0 || p.y < 8.0 || p.x > F.world.x - 8.0 || p.y > F.world.x - 8.0) {
        return;
    }
    let h = height_at(p);
    let ground = ground_ok(p);
    let forest = forest_at(p);

    // The forest's edge: how close open ground is, sampled on rings 8 and 17 m out.
    var near_min = 1.0;
    var far_min = 1.0;
    var near_max = 0.0;
    let a0 = u01(h6) * 0.8;
    for (var k = 0; k < 8; k++) {
        let a = a0 + f32(k) * (TAU / 8.0);
        let dir = vec2f(cos(a), sin(a));
        let n = forest_at(p + dir * 8.0);
        let f = forest_at(p + dir * 17.0);
        near_min = min(near_min, n);
        far_min = min(far_min, f);
        near_max = max(near_max, max(n, f * 0.6));
    }
    let edge = max(1.0 - smoothstep(0.1, 0.45, near_min), (1.0 - smoothstep(0.1, 0.45, far_min)) * 0.55);

    // A closed forest still has gaps: where an old tree fell, thinner stands. They let
    // sunflecks through to the floor. (Not on its edges, which the sun keeps dense.)
    let gaps = mix(0.45, 1.0, smoothstep(-0.35, 0.2, gnoise(p * 0.03 + vec2f(5.1, -2.3))));
    // Stands of different ages, a few hundred metres across.
    let age = smoothstep(-0.3, 0.35, fbm(p * 0.004 + vec2f(9.1, 3.3), 3));

    let r = u01(h2);
    let r2 = u01(h7);
    var form = NONE;
    var scale = 1.0;
    var group = vec2f(-10.0, 0.0);
    if (forest > 0.02) {
        if (r < forest * 0.85 * mix(gaps, 1.0, edge)) {
            // On the edge a tree grows in the light, with its crown down low.
            form = select(FOREST, OPEN, r2 < smoothstep(0.15, 0.7, edge) * 0.85);
            // Within a stand trees vary; some are held back under the canopy, a few emerge above it.
            scale = mix(0.72, 1.22, age) * exp2((u01(h8) + u01(h9) - 1.0) * 0.4);
            let k = u01(pcg(h9));
            if (form == FOREST && k < 0.13) {
                scale *= 0.55;
            } else if (k > 0.965) {
                scale *= 1.3;
            }
            // and the trees step down towards the open
            scale *= mix(1.0, 0.8, edge);
        } else if (r2 < edge * 0.55) {
            form = SHRUB;
        }
    } else if (near_max > 0.15) {
        // the mantle: a band of shrubs and young trees in front of the forest
        let band = smoothstep(0.15, 0.6, near_max) * ground;
        if (r < band * 0.5) {
            form = SHRUB;
        } else if (r < band * 0.6) {
            form = OPEN;
            scale = mix(0.5, 0.85, u01(h8));
        }
    } else {
        group = group_at(p);
        if (group.x > 0.0) {
            if (r < 0.7 * sqrt(group.x) * ground) {
                form = OPEN;
                scale = mix(0.75, 1.3, group.x) * mix(0.8, 1.15, u01(h8));
            }
        } else if (group.x > -0.4) {
            if (r < 0.32 * (1.0 + group.x / 0.4) * ground) {
                form = SHRUB;
            }
        } else if (r < 0.0005 * ground) {
            // a lone tree, grown broad in the open
            form = OPEN;
            scale = mix(1.0, 1.5, u01(h8));
        }
    }
    if (form == NONE) {
        return;
    }

    // species: spruce up high, birch in patches and along forest edges, broadleaf elsewhere;
    // a group out in the open is mostly of one kind
    let sp_noise = fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3);
    let alt = h + sp_noise * 140.0;
    let spruce_p = smoothstep(250.0, 380.0, alt);
    let birch_p = clamp(smoothstep(0.12, 0.32, sp_noise) * 0.8 + edge * 0.35, 0.0, 1.0) * (1.0 - spruce_p);
    let in_group = group.x > -0.4;
    var rs = u01(h3);
    if (in_group && u01(h5) < 0.85) {
        rs = u01(pcg(u32(group.y)));
    }
    var species = 0u;
    if (rs < spruce_p) {
        species = 2u;
    } else if (rs < spruce_p + birch_p * select(1.0, 1.4, in_group)) {
        species = 1u;
    }
    if (form == SHRUB) {
        species = 0u;
        scale = mix(0.7, 1.3, u01(h8));
    }

    let rot = h5 & 255u;
    let variant = form + select(h4 & 3u, h4 & 1u, form != FOREST);
    let seed = (((h5 >> 8u) & 0x7FFu) << 3u) | variant;
    let packed = species | (rot << 2u) | (tree_scale_bits(scale) << 10u) | (seed << 18u);

    let slot = atomicAdd(&counter[0], 1u);
    if (slot >= S.cap) {
        return;
    }
    let y = ground_h(p) - 0.2;
    trees[slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(y), bitcast<u32>(p.y), packed);
}
