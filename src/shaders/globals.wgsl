// Per-frame globals shared by every render and compute pipeline (bind group 0),
// plus terrain sampling, lighting, shadows and fog helpers.

struct Frame {
    view_proj: mat4x4f,
    inv_view_proj_rot: mat4x4f, // inverse of proj * view rotation (camera at origin): ndc -> view ray
    shadow_vp: array<mat4x4f, 2>,
    cam_pos: vec4f,      // xyz camera, w = time (s)
    sun_dir: vec4f,      // xyz towards the sun
    sun_color: vec4f,    // rgb sun illuminance above the atmosphere, w = ambient strength
    atmos: vec4f,        // x haze (aerosol density scale), y overcast 0..1
    fog: vec4f,          // x density, y height falloff, z base height
    wind: vec4f,         // xy direction, z strength
    screen: vec4f,       // width, height, 1/width, 1/height
    world: vec4f,        // world size, heightmap res, texel size, shadow map res
    cascade: vec4f,      // split0, split1, texel0 (m), texel1 (m)
    lod: vec4f,          // tree LOD distances
    debug: vec4f,        // x = LOD tint
    clouds: vec4f,       // x base altitude, y top altitude, z coverage, w density
    cloud_type: vec4f,   // x, y range of cloud type (0 stratus .. 1 cumulus), z drift speed (m/s)
    cloud_shadow: vec4f, // cloud shadow map: xy centre (m), z 1 / extent, w value beyond the map
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
@group(0) @binding(9) var<uniform> SKY: SkyLight;
@group(0) @binding(10) var t_trans: texture_2d<f32>;
@group(0) @binding(11) var t_skyview: texture_2d<f32>;
@group(0) @binding(12) var t_cloud_shadow: texture_2d<f32>;
@group(0) @binding(13) var t_ap_scat: texture_3d<f32>;
@group(0) @binding(14) var t_ap_trans: texture_3d<f32>;

// Aerial perspective volume (aerial.wgsl): froxels over the screen, slice i at distance
// AP_DIST * ((i + 0.5) / slices)^2, so the near slices are finer.
const AP_DIST: f32 = 16000.0;
// Below this sun height the cloud shadow map stops stretching with the sun (it would get
// infinitely long); the sun is nearly gone by then anyway.
const CLOUD_SHADOW_MIN_MU: f32 = 0.1;

// Filled each frame on the GPU from the atmosphere tables (atmosphere.wgsl).
struct SkyLight {
    sun: vec4f,        // rgb sun illuminance reaching the camera
    amb_up: vec4f,     // sky irradiance / pi on an upward surface
    amb_ground: vec4f, // light bounced up from the ground
    horizon: vec4f,    // rgb average horizon radiance, w = illuminance on flat ground
}

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

// macro terrain normal (w of the map holds the sky visibility)
fn normal_at(p: vec2f) -> vec3f {
    return normalize(textureSampleLevel(t_normal, s_clamp, p / F.world.x, 0.0).xyz);
}

fn cam_radius() -> f32 {
    return PLANET_R + max(F.cam_pos.y, 1.0) * 0.001;
}

// Sky radiance seen from the camera along dir, from the sky-view table (no sun disk).
fn sky_radiance(dir: vec3f) -> vec3f {
    let sun = F.sun_dir.xyz;
    let elev = asin(clamp(dir.y, -1.0, 1.0));
    let h = dir.xz;
    let hs = sun.xz;
    var c = 1.0;
    if (dot(h, h) > 1e-8 && dot(hs, hs) > 1e-8) {
        c = dot(normalize(h), normalize(hs));
    }
    let azim = acos(clamp(c, -1.0, 1.0));
    return textureSampleLevel(t_skyview, s_clamp, sky_uv(elev, azim, cam_radius()), 0.0).rgb;
}

// Sun light reaching altitude alt (m), from the transmittance table; for the cloud layer.
fn sun_light_at(alt: f32) -> vec3f {
    let r = PLANET_R + max(alt, 0.0) * 0.001;
    let mu = F.sun_dir.y;
    if (ray_hits_ground(r, mu)) {
        return vec3f(0.0);
    }
    return F.sun_color.rgb * textureSampleLevel(t_trans, s_clamp, trans_uv(r, mu), 0.0).rgb;
}

// Sunlight that gets through the clouds to wp: x as the direct beam, y scattered by the cloud
// into diffuse light from above (as a fraction of the sun on a surface facing it). The map is
// looked up where wp's sun ray crosses the cloud base; it wraps around (see clouds.wgsl), so
// the bilinear filter is done by hand.
fn cloud_shadow_from(t: texture_2d<f32>, wp: vec3f) -> vec2f {
    let L = F.sun_dir.xyz;
    let q = wp.xz + L.xz * (F.clouds.x - wp.y) / max(L.y, CLOUD_SHADOW_MIN_MU);
    let n = vec2i(textureDimensions(t));
    let x = q * F.cloud_shadow.z * vec2f(n) - 0.5;
    let fl = floor(x);
    let f = x - fl;
    let i0 = (vec2i(fl) % n + n) % n;
    let i1 = (i0 + 1) % n;
    let a = textureLoad(t, i0, 0).rg;
    let b = textureLoad(t, vec2i(i1.x, i0.y), 0).rg;
    let c = textureLoad(t, vec2i(i0.x, i1.y), 0).rg;
    let d = textureLoad(t, i1, 0).rg;
    let v = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    // fade to an average beyond the stretch the map was built for
    let rel = (q - F.cloud_shadow.xy) * F.cloud_shadow.z;
    let e = max(abs(rel.x), abs(rel.y)) * 2.0;
    let beyond = vec2f(F.cloud_shadow.w, (1.0 - F.cloud_shadow.w) * 0.3);
    return mix(v, beyond, smoothstep(0.9, 1.0, e));
}

fn cloud_shadow(wp: vec3f) -> vec2f {
    return cloud_shadow_from(t_cloud_shadow, wp);
}

// Direct sun at wp, after the clouds.
fn sun_light(wp: vec3f) -> vec3f {
    return SKY.sun.rgb * cloud_shadow(wp).x;
}

// Share of the sky seen from the ground at p (baked in sky_vis.wgsl from the heightmap and
// the forest canopy).
fn sky_vis(p: vec2f) -> f32 {
    return textureSampleLevel(t_normal, s_clamp, p / F.world.x, 0.0).w;
}

// Light from the sky and the ground, plus what clouds above wp scatter down, for a point
// that sees `vis` of the sky. The part of the sky that is blocked by hills and trees is
// replaced by what those send back: lit ground, partly in shade.
fn ambient_vis(n: vec3f, wp: vec3f, vis: f32) -> vec3f {
    let up = n.y * 0.5 + 0.5;
    let bounce = SKY.amb_ground.rgb;
    let cs = cloud_shadow(wp);
    // Where trees block the sky, what is seen instead is the underside of their leaves, lit
    // through by the sun and the sky above them: green light, most of the light on a forest
    // floor. (A stand-in for the light bounced and transmitted through the canopy.)
    let leafy = smoothstep(0.3, 0.7, data_at(wp.xz).x);
    let sun_above = SKY.sun.rgb * max(F.sun_dir.y, 0.0) * cs.x;
    let through = (sun_above / PI + SKY.amb_up.rgb) * CANOPY_TRANSMIT;
    let blocked = mix(bounce * 0.5, through, leafy);
    let from_above = SKY.amb_up.rgb * vis + blocked * (1.0 - vis);
    let sky = mix(bounce, from_above, up) * F.sun_color.w;
    let cloud = SKY.sun.rgb * max(F.sun_dir.y, 0.0) * cs.y * up * vis;
    return sky + cloud;
}

// Share of the light falling on a canopy that comes out underneath, through and between the
// leaves (green: leaves pass green and absorb the rest).
const CANOPY_TRANSMIT: vec3f = vec3f(0.07, 0.13, 0.03);

fn ambient(n: vec3f, wp: vec3f) -> vec3f {
    return ambient_vis(n, wp, sky_vis(wp.xz));
}

// Aerial perspective: the air and ground haze between the camera and wp, from the volume
// built each frame from the same atmosphere as the sky.
fn apply_fog(col: vec3f, wp: vec3f) -> vec3f {
    let clip = F.view_proj * vec4f(wp, 1.0);
    let ndc = clip.xy / clip.w;
    let w = sqrt(saturate(distance(wp, F.cam_pos.xyz) / AP_DIST));
    let c = vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, w);
    let scat = textureSampleLevel(t_ap_scat, s_clamp, c, 0.0).rgb;
    let trans = textureSampleLevel(t_ap_trans, s_clamp, c, 0.0).rgb;
    return col * trans + scat;
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

// ---------------------------------------------------------------------------
// Meadow colour, shared by the terrain and the grass blades so the real grass near the
// camera and the ground further out agree.

// Low-frequency grass colour at p: lush and dry stretches (drier on raised, open or steeper
// ground, lusher in hollows), and 5-20 m patches of darker clumps and paler grass.
fn meadow_tint(p: vec2f, dat: vec4f, n: vec3f, vis: f32) -> vec3f {
    let lush = mix(vec3f(0.030, 0.066, 0.012), vec3f(0.060, 0.100, 0.020), dat.w);
    let dry_c = vec3f(0.120, 0.112, 0.046);
    let slope = 1.0 - n.y;
    let dryness = fbm(p * 0.0042 + vec2f(13.0, -7.0), 3) + slope * 1.5 - (1.0 - vis) * 4.0;
    let dry = smoothstep(-0.05, 0.3, dryness);
    var c = mix(lush, dry_c, dry * 0.55);
    let m1 = gnoise(p * 0.06);
    let m2 = gnoise(p * 0.17 + vec2f(3.7, 1.1));
    c *= 0.86 + 0.24 * m1;
    c = mix(c, c * vec3f(0.65, 0.82, 0.55), smoothstep(0.15, 0.55, m2) * 0.55);
    c = mix(c, vec3f(0.115, 0.125, 0.04), smoothstep(0.3, 0.7, -m2) * 0.3);
    return c;
}

// Wild flowers grow in drifts of one kind: buttercups, daisies or violet clover.
fn flower_drift(p: vec2f) -> f32 {
    return smoothstep(-0.15, 0.45, gnoise(p * 0.12 + vec2f(4.0, 1.5)));
}

fn flower_color(p: vec2f) -> vec3f {
    let kind = gnoise(p * 0.03 + vec2f(-5.0, 9.0));
    var c = vec3f(0.80, 0.58, 0.05);
    c = mix(c, vec3f(0.78, 0.78, 0.72), smoothstep(-0.05, 0.15, kind));
    c = mix(c, vec3f(0.34, 0.15, 0.48), smoothstep(0.35, 0.5, kind));
    return c;
}
