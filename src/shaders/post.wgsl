// Bloom, exposure, tonemapping and final grade from the HDR buffer to the swapchain.

struct Post {
    exposure: f32,
    bloom: f32,      // how much of the bloom is mixed in
    bloom_norm: f32, // 1 / number of bloom levels summed
    key: f32,        // light on flat ground at which `exposure` applies unchanged
    upscale: f32,    // 1 when the scene was rendered below the output size
}

struct SkyLight {
    sun: vec4f,
    amb_up: vec4f,
    amb_ground: vec4f,
    horizon: vec4f, // w = illuminance on flat ground
}

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> PP: Post;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var s_lin: sampler;
@group(0) @binding(4) var<uniform> SKY: SkyLight;
@group(0) @binding(5) var t_ao: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> EXPO: array<f32, 4>; // x = metered log luminance (exposure.wgsl)

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

// Catmull-Rom upscale of the scene in five bilinear taps (the corner taps of the 4x4 kernel
// carry little weight and are left out). Sharper than one bilinear tap; it can overshoot
// next to very bright texels, so the result is clamped at zero.
fn upsample(uv: vec2f) -> vec3f {
    let size = vec2f(textureDimensions(t_hdr));
    let pos = uv * size;
    let c = floor(pos - 0.5) + 0.5;
    let f = pos - c;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c - 1.0) / size;
    let t3 = (c + 2.0) / size;
    let t12 = (c + w2 / w12) / size;
    var sum = textureSampleLevel(t_hdr, s_lin, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
    sum += textureSampleLevel(t_hdr, s_lin, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
    sum += textureSampleLevel(t_hdr, s_lin, t12, 0.0).rgb * (w12.x * w12.y);
    sum += textureSampleLevel(t_hdr, s_lin, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
    sum += textureSampleLevel(t_hdr, s_lin, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(sum / wsum, vec3f(0.0));
}

// The average reflectance of an open view (meadow, sky and hills) for the metering: a view
// that comes out darker than ground of this albedo in the same light is opened up.
const METER_ALBEDO: f32 = 0.14;

fn aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy);
    var c: vec3f;
    if PP.upscale > 0.5 {
        c = upsample(in.uv);
    } else {
        c = textureLoad(t_hdr, px, 0).rgb;
    }
    // ambient occlusion (ssao.wgsl), already limited to the ambient share of the light
    c *= textureSampleLevel(t_ao, s_lin, in.uv, 0.0).r;
    let bloom = textureSampleLevel(t_bloom, s_lin, in.uv, 0.0).rgb * PP.bloom_norm;
    c = mix(c, bloom, PP.bloom);
    // partial eye adaptation: open up as the light falls towards sunset, but only by about
    // the square root, so dusk still looks darker than noon
    let adapt = clamp(pow(PP.key / max(SKY.horizon.w, 1e-4), 0.5), 0.5, 4.0);
    // and like a camera, open up in a view darker than open ground (under the trees): partway,
    // and by at most about 1.6 stops, so a forest still reads darker than a meadow
    let open_ground = SKY.horizon.w * METER_ALBEDO / PI;
    let metered = exp(EXPO[0]);
    let open_up = clamp(pow(open_ground / max(metered, 1e-6), 0.6), 1.0, 3.0);
    c *= PP.exposure * adapt * open_up;
    c = aces(c);
    // gentle saturation lift and vignette
    let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    c = mix(vec3f(l), c, 1.0);
    let v = in.uv - 0.5;
    c *= 1.0 - dot(v, v) * 0.35;
    // dither to hide banding in the sky
    let h = u01(pcg(u32(px.x) * 1973u + u32(px.y) * 9277u)) - 0.5;
    c += h / 255.0;
    return vec4f(max(c, vec3f(0.0)), 1.0);
}
