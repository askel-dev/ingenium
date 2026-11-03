// One-time forest placement: one candidate per grid cell, kept with probability
// equal to the forest density from the world data map. Writes a compact tree list.
//
// Tree record (vec4u): xyz = bitcast position, w = packed
//   bits 0-1 species, 2-9 rotation, 10-17 scale, 18-31 seed

struct Scatter {
    cell: f32,
    dim: u32,
    cap: u32,
    pad: u32,
}

@group(1) @binding(0) var<uniform> S: Scatter;
@group(1) @binding(1) var<storage, read_write> trees: array<vec4u>;
@group(1) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;

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
    let p = (vec2f(cell) + 0.05 + vec2f(u01(h0), u01(h1)) * 0.9) * S.cell;
    if (p.x < 8.0 || p.y < 8.0 || p.x > F.world.x - 8.0 || p.y > F.world.x - 8.0) {
        return;
    }
    let dat = data_at(p);
    let h = height_at(p);
    let n = normal_at(p);
    var density = dat.x * 0.95 + 0.003;
    density *= 1.0 - smoothstep(0.22, 0.4, 1.0 - n.y);
    density *= 1.0 - dat.y * 0.85;
    if (u01(h2) >= density) {
        return;
    }

    // species: spruce up high, birch in patches and along forest edges, broadleaf elsewhere
    let sp_noise = fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3);
    let alt = h + sp_noise * 140.0;
    let spruce_p = smoothstep(250.0, 380.0, alt);
    let edge = 1.0 - smoothstep(0.25, 0.65, dat.x);
    let birch_p = clamp(smoothstep(0.12, 0.32, sp_noise) * 0.8 + edge * 0.35, 0.0, 1.0) * (1.0 - spruce_p);
    let r = u01(h3);
    var species = 0u;
    if (r < spruce_p) {
        species = 2u;
    } else if (r < spruce_p + birch_p) {
        species = 1u;
    }

    let rot = h5 & 255u;
    let scale_q = (h4 >> 8u) & 255u;
    let seed = (h5 >> 8u) & 0x3FFFu;
    let packed = species | (rot << 2u) | (scale_q << 10u) | (seed << 18u);

    let slot = atomicAdd(&counter[0], 1u);
    if (slot >= S.cap) {
        return;
    }
    let y = ground_h(p) - 0.2;
    trees[slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(y), bitcast<u32>(p.y), packed);
}
