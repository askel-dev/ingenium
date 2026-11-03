// Per-frame globals shared by every render and compute pipeline (bind group 0),
// plus terrain sampling, lighting, shadows and fog helpers.

struct Frame {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    shadow_vp: array<mat4x4f, 2>,
    cam_pos: vec4f,      // xyz camera, w = time (s)
    sun_dir: vec4f,      // xyz towards the sun
    sun_color: vec4f,    // rgb sun radiance, w = ambient strength
    sky_zenith: vec4f,
    sky_horizon: vec4f,
    fog: vec4f,          // x density, y height falloff, z base height, w cloud cover
    wind: vec4f,         // xy direction, z strength
    screen: vec4f,       // width, height, 1/width, 1/height
    world: vec4f,        // world size, heightmap res, texel size, shadow map res
    cascade: vec4f,      // split0, split1, texel0 (m), texel1 (m)
    lod: vec4f,          // tree LOD distances
    debug: vec4f,        // x = LOD tint
    morph: array<vec4f, 12>,
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var t_height: texture_2d<f32>;
@group(0) @binding(2) var t_data: texture_2d<f32>;
@group(0) @binding(3) var t_normal: texture_2d<f32>;
@group(0) @binding(4) var t_shadow: texture_depth_2d_array;
@group(0) @binding(5) var s_shadow: sampler_comparison;
@group(0) @binding(6) var s_repeat: sampler;
@group(0) @binding(7) var t_leaf: texture_2d_array<f32>;
@group(0) @binding(8) var s_clamp: sampler;

struct Pass {
    view_proj: mat4x4f,
    ref_cam: vec4f,   // main camera position (LOD / morph reference)
    params: vec4f,    // x = 1 for shadow passes
}

fn height_at(p: vec2f) -> f32 {
    let res = i32(F.world.y);
    let uv = p / F.world.z - 0.5;
    let fl = floor(uv);
    let i = vec2i(fl);
    let f = uv - fl;
    let lo = vec2i(0);
    let hi = vec2i(res - 1);
    let a = textureLoad(t_height, clamp(i, lo, hi), 0).r;
    let b = textureLoad(t_height, clamp(i + vec2i(1, 0), lo, hi), 0).r;
    let c = textureLoad(t_height, clamp(i + vec2i(0, 1), lo, hi), 0).r;
    let d = textureLoad(t_height, clamp(i + vec2i(1, 1), lo, hi), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn detail_h(p: vec2f) -> f32 {
    return gnoise(p * 0.21) * 0.35 + gnoise(p * 0.73) * 0.12;
}

fn ground_h(p: vec2f) -> f32 {
    return height_at(p) + detail_h(p);
}

fn data_at(p: vec2f) -> vec4f {
    return textureSampleLevel(t_data, s_clamp, p / F.world.x, 0.0);
}

fn normal_at(p: vec2f) -> vec3f {
    return normalize(textureSampleLevel(t_normal, s_clamp, p / F.world.x, 0.0).xyz);
}

fn sky_radiance(dir: vec3f) -> vec3f {
    let sun = F.sun_dir.xyz;
    let y = max(dir.y, 0.0);
    var col = mix(F.sky_horizon.rgb, F.sky_zenith.rgb, pow(y, 0.5));
    let mu = max(dot(dir, sun), 0.0);
    col += F.sun_color.rgb * (pow(mu, 6.0) * 0.08 + pow(mu, 48.0) * 0.25);
    if (dir.y < 0.0) {
        col = mix(F.sky_horizon.rgb, F.sky_horizon.rgb * 0.55, min(-dir.y * 3.0, 1.0));
    }
    return col;
}

fn ambient(n: vec3f) -> vec3f {
    let up = n.y * 0.5 + 0.5;
    let sky = mix(F.sky_horizon.rgb, F.sky_zenith.rgb, 0.6);
    let ground = vec3f(0.09, 0.10, 0.05) * (F.sun_color.rgb * 0.15 + 0.4);
    return mix(ground, sky, up) * F.sun_color.w;
}

fn apply_fog(col: vec3f, wp: vec3f) -> vec3f {
    let cam = F.cam_pos.xyz;
    let v = wp - cam;
    let dist = length(v);
    let dir = v / max(dist, 1e-4);
    let a = F.fog.x;
    let b = F.fog.y;
    let base = a * exp(-b * (cam.y - F.fog.z));
    let k = b * v.y;
    var amount = base * dist;
    if (abs(k) > 1e-4) {
        amount = base * dist * (1.0 - exp(-k)) / k;
    }
    let ext = clamp(1.0 - exp(-amount), 0.0, 1.0);
    let mu = max(dot(dir, F.sun_dir.xyz), 0.0);
    let fog_col = F.sky_horizon.rgb * 0.92 + F.sun_color.rgb * pow(mu, 6.0) * 0.25;
    return mix(col, fog_col, ext);
}

fn shadow_sample(ci: i32, wp: vec3f) -> f32 {
    let lp = F.shadow_vp[ci] * vec4f(wp, 1.0);
    let uv = lp.xy * vec2f(0.5, -0.5) + 0.5;
    if (uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0 || lp.z >= 1.0 || lp.z <= 0.0) {
        return 1.0;
    }
    let texel = 1.0 / F.world.w;
    var s = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            s += textureSampleCompareLevel(t_shadow, s_shadow, uv + vec2f(f32(x), f32(y)) * texel, ci, lp.z);
        }
    }
    return s / 9.0;
}

fn shadow_at(wp: vec3f, n: vec3f) -> f32 {
    let d = distance(wp, F.cam_pos.xyz);
    if (d < F.cascade.x) {
        let s0 = shadow_sample(0, wp + n * F.cascade.z * 1.5);
        let s1 = shadow_sample(1, wp + n * F.cascade.w * 1.5);
        return mix(s0, s1, smoothstep(F.cascade.x * 0.8, F.cascade.x, d));
    }
    if (d < F.cascade.y) {
        let s = shadow_sample(1, wp + n * F.cascade.w * 1.5);
        return mix(s, 1.0, smoothstep(F.cascade.y * 0.75, F.cascade.y, d));
    }
    return 1.0;
}

fn lod_tint(lod: u32) -> vec3f {
    switch lod {
        case 0u: { return vec3f(1.0, 0.25, 0.2); }
        case 1u: { return vec3f(1.0, 0.85, 0.2); }
        case 2u: { return vec3f(0.25, 1.0, 0.3); }
        default: { return vec3f(0.25, 0.5, 1.0); }
    }
}
