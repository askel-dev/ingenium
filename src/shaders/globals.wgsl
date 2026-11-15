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
    // The canopy light volume (canopy.rs), per level: xz of its first voxel (m), voxel size (m),
    // first row in the stacked textures; [4]: y range above the terrain (m), z unused, w levels
    // in use (0: off).
    canopy: array<vec4f, 5>,
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
// Scanned materials (materials.rs): a = albedo rgb + height, b = normal xy + roughness + AO.
@group(0) @binding(15) var t_mat_a: texture_2d_array<f32>;
@group(0) @binding(16) var t_mat_b: texture_2d_array<f32>;
@group(0) @binding(17) var s_mat: sampler;
// The foliage's normals (foliage.rs): normal xy (+x along u, +y along v), roughness, translucency.
// (t_leaf, binding 7, holds its albedo and coverage.)
@group(0) @binding(18) var t_leaf_n: texture_2d_array<f32>;
// The canopy light volume (canopy.wgsl): rows 0-119 the SH of the sky the leaves let through,
// rows 120-239 the leaves' light and the sun's transmittance. Repeats in x and z, clamps in y.
@group(0) @binding(19) var t_canopy: texture_3d<f32>;
@group(0) @binding(20) var s_canopy: sampler;
// unfiltered, for gathering the heightmap (height_fast)
@group(0) @binding(21) var s_point: sampler;

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

// height_at from one gather of the four texels instead of four loads
fn height_fast(p: vec2f) -> f32 {
    let uv = p / F.world.z - 0.5;
    let fl = floor(uv);
    let f = uv - fl;
    let g = textureGather(0, t_height, s_point, (fl + 1.0) / F.world.y);
    return mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y);
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

// macro terrain normal (the map holds its x and z; z and w the sky visibility, sky_vis.wgsl)
fn normal_at(p: vec2f) -> vec3f {
    let m = textureSampleLevel(t_normal, s_clamp, p / F.world.x, 0.0).xy;
    return normalize(vec3f(m.x, sqrt(max(1.0 - dot(m, m), 0.0)), m.y));
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

// The same for the terrain alone, without the forest.
fn sky_vis_ground(p: vec2f) -> f32 {
    return textureSampleLevel(t_normal, s_clamp, p / F.world.x, 0.0).z;
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
    return ambient_c(n, wp, canopy_sample(wp, n));
}

// ---------------------------------------------------------------------------
// The canopy light volume (canopy.rs, canopy.wgsl): a clipmap of 4 levels of 128 x 128 voxels
// (1, 2, 4, 8 m across, at most C_DY tall) around the camera, following the terrain from C_Y0
// to C_Y1 m above it. The levels are stacked in y, and wrap around in x and z: world voxel i is
// texel i mod 128.

const C_LEVELS: u32 = 4u;
const C_N: f32 = 128.0;
const C_NY: f32 = 120.0;
const C_DY: f32 = 2.0;
const C_Y0: f32 = -4.0;
const C_Y1: f32 = 44.0;
// A level hands over to the next between these distances from its centre (in half-widths),
// short of its edge, where its texels wrap around.
const C_BLEND0: f32 = 0.78;
const C_BLEND1: f32 = 0.92;
// Beyond the volume the leaves that block the sky are taken as this much sunlit and sky-lit.
const C_FALLBACK_LIT: f32 = 0.45;
// Leaves deep in a canopy are lit less by the sun and sky than by the light the leaves around
// them scatter, which the volume doesn't follow: the share of the rest that reaches them anyway.
const CANOPY_MS: f32 = 0.45;
// Share of the light falling on a canopy that the leaves that block a point's sky send on to
// it, through them and off them (with C_FALLBACK_LIT, CANOPY_TRANSMIT beyond the volume).
const CANOPY_SCATTER: vec3f = CANOPY_TRANSMIT * 2.2;

// How far p is from level l's centre, in half-widths (beyond 1: outside it).
fn c_edge(l: u32, p: vec2f) -> f32 {
    let L = F.canopy[l];
    let half = 0.5 * C_N * L.z;
    let d = abs(p - (L.xy + half)) / half;
    return max(d.x, d.y);
}

// The finest level from `first` on that holds p within `edge` half-widths; C_LEVELS if none.
fn c_level_at(p: vec2f, first: u32, edge: f32) -> u32 {
    var l = first;
    while (l < C_LEVELS && c_edge(l, p) >= edge) {
        l++;
    }
    return l;
}

// Level l's voxel height (m).
fn c_dy(l: u32) -> f32 {
    return min(F.canopy[l].z, C_DY);
}

// Level l's texture coordinates at p (h: the terrain height under p): x and z wrap, y in texels
// of the stacked rows, kept half a texel inside the level's own.
fn c_coords(l: u32, p: vec3f, h: f32) -> vec3f {
    let L = F.canopy[l];
    let dy = c_dy(l);
    let rows = (C_Y1 - C_Y0) / dy;
    let y = clamp((p.y - h - C_Y0) / dy, 0.5, rows - 0.5) + L.w;
    return vec3f(p.x / (L.z * C_N), y, p.z / (L.z * C_N));
}

// 0.6 of level l's voxel out along n
fn c_offset(l: u32, n: vec3f) -> vec3f {
    return n * vec3f(F.canopy[l].z, c_dy(l), F.canopy[l].z) * 0.6;
}

// What the shading needs from the volume at a point.
struct Canopy {
    sh: vec4f,     // SH-L1 (1, x, y, z) of the sky the leaves let through, over the upper hemisphere
    lit: vec3f,    // x sunlit, y sky-lit share of the leaves that block the sky; z sun transmittance
    w: f32,        // 1 inside the volume, fading to 0 beyond it
    vis_g: f32,    // share of the sky the terrain leaves open (the 2D bake)
    vis_fb: f32,   // share of the rest the forest leaves open, beyond the volume (the 2D bake)
}

fn c_fetch(l: u32, p: vec3f, h: f32) -> array<vec4f, 2> {
    let c = c_coords(l, p, h);
    let a = textureSampleLevel(t_canopy, s_canopy, vec3f(c.x, c.y / (2.0 * C_NY), c.z), 0.0);
    let b = textureSampleLevel(t_canopy, s_canopy, vec3f(c.x, (c.y + C_NY) / (2.0 * C_NY), c.z), 0.0);
    return array<vec4f, 2>(a, b);
}

// The volume at wp, for a surface facing n: looked up 0.6 voxels out along n, so that a surface
// isn't shaded by the leaves or bark it is part of. Beyond the volume (or with it off), the 2D
// bake's guess.
fn canopy_sample(wp: vec3f, n: vec3f) -> Canopy {
    var c: Canopy;
    let bake = textureSampleLevel(t_normal, s_clamp, wp.xz / F.world.x, 0.0);
    c.vis_g = bake.z;
    c.vis_fb = saturate(bake.w / max(bake.z, 0.05));
    c.sh = vec4f(0.0);
    c.lit = vec3f(1.0);
    c.w = 0.0;
    let count = u32(F.canopy[4].w);
    if (count == 0u) {
        return c;
    }
    let l = c_level_at(wp.xz, 0u, C_BLEND1);
    if (l >= count) {
        return c;
    }
    let t = smoothstep(C_BLEND0, C_BLEND1, c_edge(l, wp.xz));
    let p = wp + c_offset(l, n);
    let h = height_fast(p.xz);
    var s = c_fetch(l, p, h);
    var w = 1.0;
    if (t > 0.0) {
        if (l + 1u < count) {
            let s2 = c_fetch(l + 1u, wp + c_offset(l + 1u, n), h);
            s = array<vec4f, 2>(mix(s[0], s2[0], t), mix(s[1], s2[1], t));
        } else {
            w = 1.0 - t;
        }
    }
    c.sh = s[0];
    c.lit = s[1].xyz;
    c.w = w;
    return c;
}

// Share of the sky (canopy only, not the terrain) seen by a surface facing n: the irradiance
// from the SH over that of an open upper hemisphere, pi/2 (1 + n.y).
fn canopy_vis(c: Canopy, n: vec3f) -> f32 {
    let e = PI * 0.282095 * c.sh.x + (TAU / 3.0) * 0.488603 * dot(c.sh.yzw, n);
    let open = 0.5 * PI * (1.0 + n.y);
    let v = saturate((e + 0.1) / (open + 0.1));
    return mix(c.vis_fb, v, c.w);
}

// Sunlight that gets through the leaves, for beyond the shadow cascades.
fn canopy_sun(c: Canopy) -> f32 {
    return mix(1.0, c.lit.z, c.w);
}

// Light from the sky and the ground, plus what clouds above wp scatter down (as ambient_vis),
// with the sky blocked by the terrain and by the leaves told apart: the terrain sends back
// ground light, the leaves what they let through and scatter of the sun and sky on them.
fn ambient_c(n: vec3f, wp: vec3f, c: Canopy) -> vec3f {
    let up = n.y * 0.5 + 0.5;
    let bounce = SKY.amb_ground.rgb;
    let cs = cloud_shadow(wp);
    let vc = canopy_vis(c, n);
    let vg = c.vis_g;
    let sun_above = SKY.sun.rgb * max(F.sun_dir.y, 0.0) * cs.x;
    let lit = mix(vec2f(C_FALLBACK_LIT), mix(c.lit.xy, vec2f(1.0), CANOPY_MS), c.w);
    let leaves = (sun_above * lit.x / PI + SKY.amb_up.rgb * lit.y) * CANOPY_SCATTER;
    let from_above = SKY.amb_up.rgb * vg * vc + bounce * 0.5 * (1.0 - vg) + leaves * vg * (1.0 - vc);
    let sky = mix(bounce, from_above, up) * F.sun_color.w;
    let cloud = SKY.sun.rgb * max(F.sun_dir.y, 0.0) * cs.y * up * vg * vc;
    return sky + cloud;
}

// The whole share of the sky a surface facing n sees: terrain and canopy.
fn canopy_sky(c: Canopy, n: vec3f) -> f32 {
    return c.vis_g * canopy_vis(c, n);
}

// The `K` views: the sky a surface sees (as a share of what an open, upward surface gets),
// and the sun that gets through the leaves.
fn canopy_debug(col: vec3f, c: Canopy, n: vec3f) -> vec3f {
    let mode = u32(F.debug.y + 0.5);
    if (mode == 2u) {
        return vec3f(canopy_sky(c, n) * (0.5 + 0.5 * n.y)) * 0.12;
    }
    if (mode == 3u) {
        return vec3f(canopy_sun(c)) * 0.12;
    }
    return col;
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

// The sun's shadow at wp; `far` is what to use beyond the cascades (the sun the canopy volume
// lets through, canopy_sun).
fn shadow_at(wp: vec3f, n: vec3f, far: f32) -> f32 {
    let d = distance(wp, F.cam_pos.xyz);
    if (d < F.cascade.x) {
        let s0 = shadow_sample(0, wp + n * F.cascade.z * 1.5);
        let s1 = shadow_sample(1, wp + n * F.cascade.w * 1.5);
        return mix(s0, s1, smoothstep(F.cascade.x * 0.8, F.cascade.x, d));
    }
    if (d < F.cascade.y) {
        let s = shadow_sample(1, wp + n * F.cascade.w * 1.5);
        return mix(s, far, smoothstep(F.cascade.y * 0.75, F.cascade.y, d));
    }
    return far;
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

// The grass blades (and the flowers among them) thin out between these distances (m) from
// the camera (grass_gen.wgsl, grass.rs); the ground paints the flowers beyond.
const GRASS_NEAR: f32 = 10.4;
const GRASS_FAR: f32 = 26.0;

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

// ---------------------------------------------------------------------------
// Scanned materials. Layers, in the order of materials.rs; the sizes (m) are what each scan
// covers, from the Poly Haven metadata.

const MAT_LEAVES: i32 = 3;
const MAT_MOSS: i32 = 4;
const MAT_GRASS: i32 = 5;
const MAT_SOIL: i32 = 6;
const MAT_ROCK: i32 = 7;
// coarsest mip: the layer's average (albedo, height, roughness, AO)
const MAT_MEAN_LOD: f32 = 11.0;

struct MatSample {
    a: vec4f, // albedo, height
    b: vec4f, // normal x, y in -1..1 (+x along u, +y along v), roughness, AO
}

fn mat_fetch(layer: i32, uv: vec2f, dx: vec2f, dy: vec2f) -> MatSample {
    var m: MatSample;
    m.a = textureSampleGrad(t_mat_a, s_mat, uv, layer, dx, dy);
    let b = textureSampleGrad(t_mat_b, s_mat, uv, layer, dx, dy);
    m.b = vec4f(b.xy * 2.0 - 1.0, b.zw);
    return m;
}

fn mat_mean(layer: i32) -> MatSample {
    var m: MatSample;
    m.a = textureSampleLevel(t_mat_a, s_mat, vec2f(0.5), layer, MAT_MEAN_LOD);
    let b = textureSampleLevel(t_mat_b, s_mat, vec2f(0.5), layer, MAT_MEAN_LOD);
    m.b = vec4f(b.xy * 2.0 - 1.0, b.zw);
    return m;
}

fn mat_mix(x: MatSample, y: MatSample, t: f32) -> MatSample {
    var m: MatSample;
    m.a = mix(x.a, y.a, t);
    m.b = mix(x.b, y.b, t);
    return m;
}

// Hex-tiling (after Mikkelsen 2022, "Practical real-time hex-tiling"): the plane is cut into
// a triangle grid, every grid vertex takes the texture at a random rotation and offset, and a
// point blends the three around it, favouring the one whose surface stands highest there. No
// tile repeats, and the seams follow the scan's own relief. uv in tiles, dx/dy its gradients;
// the returned normal is turned back to the uv axes.
fn mat_hex(layer: i32, uv: vec2f, dx: vec2f, dy: vec2f, seed: i32) -> MatSample {
    // grid cells about 0.6 tiles across
    let st = uv * 1.9;
    let skew = vec2f(st.x - 0.57735027 * st.y, 1.15470054 * st.y);
    let base = vec2i(floor(skew));
    let f = fract(skew);
    let z = 1.0 - f.x - f.y;
    var w: vec3f;
    var v0: vec2i;
    var v1: vec2i;
    var v2: vec2i;
    if (z > 0.0) {
        w = vec3f(z, f.y, f.x);
        v0 = base;
        v1 = base + vec2i(0, 1);
        v2 = base + vec2i(1, 0);
    } else {
        w = vec3f(-z, 1.0 - f.y, 1.0 - f.x);
        v0 = base + vec2i(1, 1);
        v1 = base + vec2i(1, 0);
        v2 = base + vec2i(0, 1);
    }
    // Sharpened up front, so a grid vertex whose share is negligible here isn't sampled at all
    // (most of the plane is near one vertex; only the seams take two or three lookups).
    var wb = w * w;
    wb *= wb;
    wb /= wb.x + wb.y + wb.z;
    let vs = array<vec2i, 3>(v0, v1, v2);
    var m: MatSample;
    m.a = vec4f(0.0);
    m.b = vec4f(0.0);
    var total = 0.0;
    for (var i = 0; i < 3; i++) {
        if (wb[i] < 0.05) {
            continue;
        }
        let h = hash_i2(vs[i] + vec2i(seed * 7919, layer * 104729));
        let ang = u01(h) * TAU;
        let cs = vec2f(cos(ang), sin(ang));
        let rot = mat2x2f(cs.x, cs.y, -cs.y, cs.x);
        let off = vec2f(u01(pcg(h + 1u)), u01(pcg(h + 2u)));
        var t = mat_fetch(layer, rot * uv + off, rot * dx, rot * dy);
        // back from the rotated texture's axes (rot is orthonormal: its inverse is its transpose)
        t.b = vec4f(transpose(rot) * t.b.xy, t.b.zw);
        // the higher surface wins the seam
        let k = pow(wb[i] * (t.a.w + 0.35), 3.0);
        m.a += t.a * k;
        m.b += t.b * k;
        total += k;
    }
    m.a /= total;
    m.b /= total;
    return m;
}

// Sun specular (GGX, Smith, Schlick, F0 = 0.04 for a dielectric), as a factor on the sun light
// the way the diffuse terms use it (albedo * sun * n.l), so it carries the pi of the BRDF.
fn sun_specular(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
    let nl = dot(n, l);
    let nv = max(dot(n, v), 1e-3);
    if (nl <= 0.0) {
        return 0.0;
    }
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let a = max(rough * rough, 0.02);
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let kk = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g = nl / (nl * (1.0 - kk) + kk) * nv / (nv * (1.0 - kk) + kk);
    let fr = 0.04 + 0.96 * pow(1.0 - saturate(dot(h, v)), 5.0);
    return d * g * fr / (4.0 * nv) * PI;
}

// Light that a crevice still lets in from the sun, from the map's AO (Brinkman's micro-shadow
// approximation): a fully open texel is untouched, a deep one only lit when facing the sun.
fn micro_shadow(ao: f32, nl: f32) -> f32 {
    return saturate(nl + 2.0 * ao * ao - 1.0);
}
