// One-time bake after the heightmap: the terrain normal (xyz) and how much of the sky each
// point sees (w), for the ambient light. For each of 16 directions it marches outwards over
// ~2 km of heightmap, finds the highest horizon above the local tangent plane, and counts
// the cosine-weighted share of the sky below it as blocked. Valleys, gullies and the foot
// of a slope come out darker; ridges and open plains see the whole sky.
//
// The forest counts too: around the point, the ground is raised by a canopy height wherever
// the data map has forest, so a clearing or the meadow along a forest edge loses part of its
// sky, and the forest floor most of it. Leaves let some sky through, so the canopy only does
// part of the blocking.

struct GenParams {
    world: f32,
    res: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> G: GenParams;
@group(0) @binding(1) var t_h: texture_2d<f32>;
@group(0) @binding(2) var o_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var t_data: texture_2d<f32>;

const DIRS: u32 = 16u;
const STEPS: u32 = 22u;

const CANOPY_H: f32 = 15.0;
// how much of the sky a canopy overhead blocks
const CANOPY_BLOCK: f32 = 0.6;

fn h_texel(i: vec2i) -> f32 {
    let r = i32(G.res) - 1;
    return textureLoad(t_h, clamp(i, vec2i(0), vec2i(r)), 0).r;
}

// ground (x) and canopy top (y) at a texel
fn hc_texel(i: vec2i) -> vec2f {
    let r = i32(G.res) - 1;
    let c = clamp(i, vec2i(0), vec2i(r));
    let h = textureLoad(t_h, c, 0).r;
    let forest = textureLoad(t_data, c, 0).r;
    return vec2f(h, h + CANOPY_H * smoothstep(0.2, 0.7, forest));
}

// bilinear, in texel units
fn hc_at(x: vec2f) -> vec2f {
    let fl = floor(x - 0.5);
    let f = x - 0.5 - fl;
    let i = vec2i(fl);
    let a = hc_texel(i);
    let b = hc_texel(i + vec2i(1, 0));
    let c = hc_texel(i + vec2i(0, 1));
    let d = hc_texel(i + vec2i(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn blocked(tan_h: f32, tan_t: f32) -> f32 {
    let s = sin(max(atan(tan_h) - atan(tan_t), 0.0));
    return s * s;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let res = u32(G.res);
    if (id.x >= res || id.y >= res) {
        return;
    }
    let texel = G.world / G.res;
    let ip = vec2i(id.xy);
    let h0 = h_texel(ip);
    // same central difference as the terrain generator used to compute analytically
    let n = normalize(vec3f(
        h_texel(ip - vec2i(1, 0)) - h_texel(ip + vec2i(1, 0)),
        2.0 * texel,
        h_texel(ip - vec2i(0, 1)) - h_texel(ip + vec2i(0, 1))));

    let x0 = vec2f(ip) + 0.5;
    let jitter = u01(hash_i2(ip));
    var occ_ground = 0.0;
    var occ_canopy = 0.0;
    for (var d = 0u; d < DIRS; d++) {
        let a = (f32(d) + jitter) / f32(DIRS) * TAU;
        let dir = vec2f(cos(a), sin(a));
        // rise per metre of the tangent plane along dir
        let tan_t = -(n.x * dir.x + n.z * dir.y) / max(n.y, 0.05);
        var max_g = tan_t;
        var max_c = tan_t;
        var s = 1.5;
        for (var k = 0u; k < STEPS; k++) {
            let hs = hc_at(x0 + dir * s);
            let dist = s * texel;
            max_g = max(max_g, (hs.x - h0) / dist);
            max_c = max(max_c, (hs.y - h0) / dist);
            s *= 1.33;
        }
        occ_ground += blocked(max_g, tan_t);
        occ_canopy += blocked(max_c, tan_t);
    }
    occ_ground /= f32(DIRS);
    occ_canopy /= f32(DIRS);
    let vis = 1.0 - mix(occ_ground, occ_canopy, CANOPY_BLOCK);
    textureStore(o_normal, ip, vec4f(n, vis));
}
