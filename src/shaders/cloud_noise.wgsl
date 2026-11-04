// One-time generation of the tiling cloud noise: a 3D "shape" volume (Perlin-Worley plus three
// Worley octaves), a 3D "detail" volume (three Worley octaves), their mip chains, and a 2D
// weather map (coverage, cloud type). Every noise function here tiles with the given period.

@group(0) @binding(0) var o_vol: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var o_map: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var src_vol: texture_3d<f32>;

fn hash3(c: vec3i, seed: u32) -> vec3f {
    var v = bitcast<vec3u>(c) * 1664525u + vec3u(1013904223u + seed * 2654435761u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3f(v >> vec3u(8u)) * (1.0 / 16777216.0);
}

fn wrap3(c: vec3i, period: i32) -> vec3i {
    let n = vec3i(period);
    return ((c % n) + n) % n;
}

// Inverted cellular noise: 1 on a feature point, falling to 0 a cell away.
fn worley(p: vec3f, period: i32, seed: u32) -> f32 {
    let cell = vec3i(floor(p));
    let f = p - floor(p);
    var d2 = 10.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3i(x, y, z);
                let v = vec3f(o) + hash3(wrap3(cell + o, period), seed) - f;
                d2 = min(d2, dot(v, v));
            }
        }
    }
    return 1.0 - min(sqrt(d2), 1.0);
}

fn worley_fbm(uvw: vec3f, freq: i32, seed: u32) -> f32 {
    return worley(uvw * f32(freq), freq, seed) * 0.625
        + worley(uvw * f32(freq * 2), freq * 2, seed + 1u) * 0.25
        + worley(uvw * f32(freq * 4), freq * 4, seed + 2u) * 0.125;
}

fn grad3(c: vec3i, period: i32, seed: u32) -> vec3f {
    let h = hash3(wrap3(c, period), seed) * 2.0 - 1.0;
    return h / max(length(h), 1e-3);
}

fn perlin(p: vec3f, period: i32, seed: u32) -> f32 {
    let i = vec3i(floor(p));
    let f = p - floor(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let n000 = dot(grad3(i, period, seed), f);
    let n100 = dot(grad3(i + vec3i(1, 0, 0), period, seed), f - vec3f(1.0, 0.0, 0.0));
    let n010 = dot(grad3(i + vec3i(0, 1, 0), period, seed), f - vec3f(0.0, 1.0, 0.0));
    let n110 = dot(grad3(i + vec3i(1, 1, 0), period, seed), f - vec3f(1.0, 1.0, 0.0));
    let n001 = dot(grad3(i + vec3i(0, 0, 1), period, seed), f - vec3f(0.0, 0.0, 1.0));
    let n101 = dot(grad3(i + vec3i(1, 0, 1), period, seed), f - vec3f(1.0, 0.0, 1.0));
    let n011 = dot(grad3(i + vec3i(0, 1, 1), period, seed), f - vec3f(0.0, 1.0, 1.0));
    let n111 = dot(grad3(i + vec3i(1, 1, 1), period, seed), f - vec3f(1.0, 1.0, 1.0));
    let y0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    let y1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    return mix(y0, y1, u.z);
}

// Perlin fBm remapped to roughly [0, 1].
fn perlin01(uvw: vec3f, freq: i32, octaves: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var fq = freq;
    for (var o = 0; o < octaves; o++) {
        sum += perlin(uvw * f32(fq), fq, seed + u32(o)) * amp;
        fq *= 2;
        amp *= 0.5;
    }
    return saturate(0.5 + sum * 2.0);
}

@compute @workgroup_size(4, 4, 4)
fn gen_shape(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(o_vol);
    if (any(id >= n)) {
        return;
    }
    let uvw = (vec3f(id) + 0.5) / vec3f(n);
    let w4 = worley_fbm(uvw, 4, 10u);
    // Perlin "dilated" by Worley: billowy blobs where the cells are, Perlin in between
    let pw = w4 + perlin01(uvw, 4, 6, 1u) * (1.0 - w4);
    textureStore(o_vol, id, vec4f(pw, w4, worley_fbm(uvw, 8, 20u), worley_fbm(uvw, 16, 30u)));
}

@compute @workgroup_size(4, 4, 4)
fn gen_detail(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(o_vol);
    if (any(id >= n)) {
        return;
    }
    let uvw = (vec3f(id) + 0.5) / vec3f(n);
    textureStore(o_vol, id, vec4f(worley_fbm(uvw, 2, 40u), worley_fbm(uvw, 4, 50u), worley_fbm(uvw, 8, 60u), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn gen_weather(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(o_map);
    if (any(id.xy >= n)) {
        return;
    }
    let uv = vec3f((vec2f(id.xy) + 0.5) / vec2f(n), 0.37);
    // coverage: clusters of cells, swelling and thinning over larger regions
    let cells = worley(uv * 10.0, 10, 70u) * 0.75 + worley(uv * 20.0, 20, 71u) * 0.25;
    let regions = perlin01(uv, 4, 4, 80u);
    let cover = saturate(cells * 0.8 + regions * 0.6 - 0.25);
    // cloud type: 0 flat stratus .. 1 towering cumulus, varying slowly
    let ctype = perlin01(uv + vec3f(0.31, 0.57, 0.0), 3, 3, 90u);
    textureStore(o_map, id.xy, vec4f(cover, ctype, 0.0, 1.0));
}

// One mip level of a volume: 2x2x2 box filter of the level above.
@compute @workgroup_size(4, 4, 4)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(o_vol);
    if (any(id >= n)) {
        return;
    }
    var s = vec4f(0.0);
    for (var i = 0u; i < 8u; i++) {
        let o = vec3u(i & 1u, (i >> 1u) & 1u, i >> 2u);
        s += textureLoad(src_vol, vec3i(id * 2u + o), 0);
    }
    textureStore(o_vol, id, s * 0.125);
}
