// HDR bloom (the Call of Duty: Advanced Warfare / Jimenez 2014 scheme): the HDR image is
// filtered down a mip chain with a 13-tap filter, then added back up it with a 3x3 tent, so
// each level adds a wider and softer halo. No threshold: everything blooms a little and only
// very bright things (the sun, silver linings) bloom visibly. The first step averages with
// Karis weights so single hot pixels don't flicker.

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_lin: sampler;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    var o: VOut;
    o.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(uv.x, 1.0 - uv.y);
    return o;
}

fn tap(uv: vec2f, o: vec2f) -> vec3f {
    return textureSampleLevel(t_src, s_lin, uv + o, 0.0).rgb;
}

fn karis(c: vec3f) -> f32 {
    return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722)));
}

fn down(uv: vec2f, first: bool) -> vec3f {
    let t = 1.0 / vec2f(textureDimensions(t_src));
    let a = tap(uv, t * vec2f(-2.0, -2.0));
    let b = tap(uv, t * vec2f(0.0, -2.0));
    let c = tap(uv, t * vec2f(2.0, -2.0));
    let d = tap(uv, t * vec2f(-2.0, 0.0));
    let e = tap(uv, vec2f(0.0));
    let f = tap(uv, t * vec2f(2.0, 0.0));
    let g = tap(uv, t * vec2f(-2.0, 2.0));
    let h = tap(uv, t * vec2f(0.0, 2.0));
    let i = tap(uv, t * vec2f(2.0, 2.0));
    let j = tap(uv, t * vec2f(-1.0, -1.0));
    let k = tap(uv, t * vec2f(1.0, -1.0));
    let l = tap(uv, t * vec2f(-1.0, 1.0));
    let m = tap(uv, t * vec2f(1.0, 1.0));
    // five overlapping 2x2 boxes: the centre one weighted 0.5, the corners 0.125 each
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25;
    let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25;
    let g4 = (e + f + h + i) * 0.25;
    if (first) {
        let w = vec4f(karis(g1), karis(g2), karis(g3), karis(g4)) * 0.125;
        let w0 = karis(g0) * 0.5;
        return (g0 * w0 + g1 * w.x + g2 * w.y + g3 * w.z + g4 * w.w) / (w0 + w.x + w.y + w.z + w.w);
    }
    return g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125;
}

@fragment
fn fs_down_first(in: VOut) -> @location(0) vec4f {
    return vec4f(down(in.uv, true), 1.0);
}

@fragment
fn fs_down(in: VOut) -> @location(0) vec4f {
    return vec4f(down(in.uv, false), 1.0);
}

// Upsample the smaller level with a 3x3 tent; blended additively onto the larger one.
@fragment
fn fs_up(in: VOut) -> @location(0) vec4f {
    let t = 1.0 / vec2f(textureDimensions(t_src));
    var s = tap(in.uv, vec2f(0.0)) * 4.0;
    s += (tap(in.uv, t * vec2f(-1.0, 0.0)) + tap(in.uv, t * vec2f(1.0, 0.0)) + tap(in.uv, t * vec2f(0.0, -1.0)) + tap(in.uv, t * vec2f(0.0, 1.0))) * 2.0;
    s += tap(in.uv, t * vec2f(-1.0, -1.0)) + tap(in.uv, t * vec2f(1.0, -1.0)) + tap(in.uv, t * vec2f(-1.0, 1.0)) + tap(in.uv, t * vec2f(1.0, 1.0));
    return vec4f(s / 16.0, 1.0);
}
