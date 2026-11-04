// One-time terrain generation: height and a "world data" map (forest density, rock,
// flowers, colour variation). The normal map and sky visibility follow in sky_vis.wgsl.

struct GenParams {
    world: f32,
    res: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> G: GenParams;
@group(0) @binding(1) var o_height: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var o_data: texture_storage_2d<rgba8unorm, write>;

fn bump(p: vec2f, c: vec2f, r: f32) -> f32 {
    let d = (p - c) / r;
    return exp(-dot(d, d));
}

fn terrain_h(p: vec2f) -> f32 {
    let c = p / G.world - 0.5;
    let d = length(c) * 2.0;
    let wq = p * 0.00032;
    let warp = vec2f(fbm(wq, 3), fbm(wq + vec2f(5.2, 1.3), 3)) * 700.0;
    let q = p + warp;
    let r = ridged(q * 0.00042, 6);
    let hills = fbm(q * 0.00085, 6);
    let mountains = smoothstep(0.5, 1.0, d);
    var h = 70.0 + hills * 110.0;
    h += (r * r * 620.0 + 140.0) * mountains;
    h += 320.0 * bump(p, vec2f(5300.0, 3500.0), 850.0) * (0.55 + 0.7 * r);
    h += 230.0 * bump(p, vec2f(2800.0, 5200.0), 700.0) * (0.55 + 0.7 * r);
    h += 150.0 * bump(p, vec2f(3300.0, 2600.0), 520.0) * (0.6 + 0.6 * r);
    // a broad, gentle valley floor in the middle of the map
    let valley = 1.0 - smoothstep(0.0, 0.45, d);
    h = mix(h, 55.0 + hills * 45.0, valley * 0.55);
    return h;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let res = u32(G.res);
    if (id.x >= res || id.y >= res) {
        return;
    }
    let texel = G.world / G.res;
    let p = (vec2f(id.xy) + 0.5) * texel;
    let h = terrain_h(p);
    let e = texel;
    let hx0 = terrain_h(p - vec2f(e, 0.0));
    let hx1 = terrain_h(p + vec2f(e, 0.0));
    let hz0 = terrain_h(p - vec2f(0.0, e));
    let hz1 = terrain_h(p + vec2f(0.0, e));
    let n = normalize(vec3f(hx0 - hx1, 2.0 * e, hz0 - hz1));
    let slope = 1.0 - n.y;

    let fnoise = fbm(p * 0.0010 + vec2f(31.7, -12.3), 5);
    var forest = smoothstep(-0.05, 0.12, fnoise + (h - 120.0) * 0.0012);
    forest *= 1.0 - smoothstep(0.18, 0.34, slope);
    forest *= 1.0 - smoothstep(520.0, 640.0, h + fbm(p * 0.004, 3) * 40.0);
    let rock = clamp(
        smoothstep(0.2, 0.36, slope + fbm(p * 0.01, 3) * 0.1) + smoothstep(620.0, 720.0, h),
        0.0, 1.0);
    let flowers = smoothstep(0.1, 0.45, fbm(p * 0.008 + vec2f(3.1, 8.2), 4)) * (1.0 - forest);
    let variation = clamp(fbm(p * 0.0035 + vec2f(-7.0, 2.0), 4) * 0.7 + 0.5, 0.0, 1.0);

    let ip = vec2i(id.xy);
    textureStore(o_height, ip, vec4f(h, 0.0, 0.0, 0.0));
    textureStore(o_data, ip, vec4f(forest, rock, flowers, variation));
}
