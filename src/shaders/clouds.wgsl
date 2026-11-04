// Volumetric clouds. A cloud layer wrapped around a curved planet is raymarched into a
// reduced-resolution buffer (rgb = light scattered towards the camera, a = transmittance). Each
// frame marches one pixel of every 2x2 block and reprojects the other three from the previous
// frame, so every pixel is refreshed every 4 frames. The sky pass composites the result.

struct CloudPass {
    prev_vp_rot: mat4x4f, // previous frame's proj * view rotation (camera at origin)
    cam_delta: vec4f,     // xyz camera movement since the previous frame, w = 1 if history is valid
    size: vec4f,          // buffer width, height, 1/width, 1/height
    params: vec4f,        // x frame index, y angle covered by one buffer pixel (radians),
                          // z 1 = refresh a quarter of the cloud shadow map, 0 = all of it
}

@group(1) @binding(0) var<uniform> C: CloudPass;
@group(1) @binding(1) var t_shape: texture_3d<f32>;
@group(1) @binding(2) var t_detail: texture_3d<f32>;
@group(1) @binding(3) var t_weather: texture_2d<f32>;
@group(1) @binding(4) var s_wrap: sampler;
@group(1) @binding(5) var t_history: texture_2d<f32>;
@group(1) @binding(6) var o_cloud: texture_storage_2d<rgba16float, write>;

const EARTH_R: f32 = 6360000.0;
const SHAPE_TILE: f32 = 3000.0;    // metres covered by one repeat of the shape volume
const DETAIL_TILE: f32 = 400.0;
const WEATHER_TILE: f32 = 40000.0;
const SHAPE_TEXEL: f32 = SHAPE_TILE / 128.0;
const DETAIL_LOD_BIAS: f32 = 1.907; // log2(shape texel / detail texel)
const MAX_START: f32 = 60000.0;    // rays that reach the layer further away than this see none
const MAX_LEN: f32 = 24000.0;      // longest stretch of the layer marched along one ray
const EXTINCTION: f32 = 0.07;      // per metre at density 1
const SUN_GAIN: f32 = 4.0;
const AMB_GAIN: f32 = 2.4;
const AERIAL: f32 = 1.0 / 32000.0; // haze over cloud distance

fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / (b - a);
}

// Density at world position p, alt = its height above the curved ground. `detail` adds the
// high-frequency erosion; the light march leaves it out.
fn cloud_density(p: vec3f, alt: f32, lod: f32, detail: bool) -> f32 {
    let hf = (alt - F.clouds.x) / (F.clouds.y - F.clouds.x);
    if (hf <= 0.0 || hf >= 1.0) {
        return 0.0;
    }
    let time = F.cam_pos.w;
    // the wind carries the clouds, and their tops lean downwind
    let xz = p.xz - F.wind.xy * (time * F.cloud_type.z + hf * 500.0);
    let w = textureSampleLevel(t_weather, s_wrap, xz / WEATHER_TILE, 0.0);
    let c = F.clouds.z;
    let cover = smoothstep(0.85 - c, 1.2 - c, w.r) * min(c * 1.6, 1.0);
    let ctype = mix(F.cloud_type.x, F.cloud_type.y, w.g);
    // flat base; the top rises from a thin stratus band to a tall cumulus tower
    let top = mix(0.22, 1.0, ctype);
    let prof = smoothstep(0.0, 0.07, hf) * (1.0 - smoothstep(top * 0.45, top, hf));
    if (cover * prof <= 0.0) {
        return 0.0;
    }
    // the noise rises slowly through the layer, so the clouds churn as they drift
    let sp = vec3f(xz.x, alt - time * 1.2, xz.y) / SHAPE_TILE;
    let s = textureSampleLevel(t_shape, s_wrap, sp, lod);
    let low = dot(s.gba, vec3f(0.625, 0.25, 0.125));
    let base = remap(s.r, low - 1.0, 1.0, 0.0, 1.0) * prof;
    var d = saturate(remap(base, 1.0 - cover, 1.0, 0.0, 1.0)) * cover;
    if (d <= 0.0 || !detail) {
        return d * F.clouds.w;
    }
    let dp = vec3f(xz.x - time * 2.0 * F.wind.x, alt - time * 3.0, xz.y - time * 2.0 * F.wind.y) / DETAIL_TILE;
    let dn = textureSampleLevel(t_detail, s_wrap, dp, lod + DETAIL_LOD_BIAS).rgb;
    let hi = dot(dn, vec3f(0.625, 0.25, 0.125));
    // wispy at the base, billowing towards the top
    let erode = mix(hi, 1.0 - hi, saturate(hf * 5.0));
    d = saturate(remap(d, erode * 0.45, 1.0, 0.0, 1.0));
    return d * F.clouds.w;
}

fn hg(mu: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// Strong forward lobe (the silver lining) plus a weak back lobe. k < 1 flattens it for the
// multiple-scattering octaves.
fn cloud_phase(mu: f32, k: f32) -> f32 {
    return mix(hg(mu, -0.25 * k), hg(mu, 0.75 * k), 0.65);
}

// Optical depth from p towards the sun: four doubling steps through the cloud (on coarser
// mips as they spread) and one long probe for shadow from tall clouds further sunward.
fn sun_optical_depth(p: vec3f, alt: f32, lod: f32) -> f32 {
    let L = F.sun_dir.xyz;
    var od = 0.0;
    var s = 0.0;
    var ds = 40.0;
    for (var i = 0; i < 4; i++) {
        let x = s + ds * 0.5;
        od += cloud_density(p + L * x, alt + L.y * x, max(lod, 0.0) + 1.0 + f32(i), false) * ds;
        s += ds;
        ds *= 2.0;
    }
    od += cloud_density(p + L * 1500.0, alt + L.y * 1500.0, max(lod, 0.0) + 4.0, false) * 900.0;
    return od * EXTINCTION;
}

// Distances along a ray from the camera where it reaches altitude h above the curved ground:
// cam.y + dy t + k t^2 = h with k = (1 - dy^2) / 2R. Sorted; x > y when it never does.
fn alt_hits(dy: f32, h: f32) -> vec2f {
    let k = max((1.0 - dy * dy) / (2.0 * EARTH_R), 1e-12);
    let c = F.cam_pos.y - h;
    let disc = dy * dy - 4.0 * k * c;
    if (disc < 0.0) {
        return vec2f(1.0, -1.0);
    }
    var q = -0.5 * (dy + select(-1.0, 1.0, dy >= 0.0) * sqrt(disc));
    if (abs(q) < 1e-9) {
        q = -1e-9;
    }
    let a = q / k;
    let b = c / q;
    return vec2f(min(a, b), max(a, b));
}

// The stretch [x, y] of the ray inside the cloud layer; empty when y <= x.
fn cloud_segment(dy: f32) -> vec2f {
    let y = F.cam_pos.y;
    let bot = alt_hits(dy, F.clouds.x);
    let top = alt_hits(dy, F.clouds.y);
    let hits_bot_from_above = bot.x <= bot.y && bot.x > 0.0;
    var t0 = 0.0;
    var t1 = 0.0;
    if (y < F.clouds.x) {
        // below: curvature always brings the ray up to the layer eventually
        t0 = bot.y;
        t1 = top.y;
    } else if (y < F.clouds.y) {
        t1 = select(top.y, bot.x, hits_bot_from_above);
    } else {
        if (top.x > top.y || top.x < 0.0) {
            return vec2f(0.0, -1.0);
        }
        t0 = top.x;
        t1 = select(top.y, bot.x, hits_bot_from_above);
    }
    let ground = alt_hits(dy, 0.0);
    if (ground.x <= ground.y && ground.x > 0.0) {
        t1 = min(t1, ground.x);
    }
    if (t0 > MAX_START) {
        return vec2f(0.0, -1.0);
    }
    return vec2f(t0, min(t1, t0 + MAX_LEN));
}

struct March {
    scat: vec3f,
    trans: f32,
    depth: f32, // transmittance-weighted distance to the cloud (layer entry when there is none)
}

fn march(dir: vec3f, jitter: f32) -> March {
    var o: March;
    o.scat = vec3f(0.0);
    o.trans = 1.0;
    let seg = cloud_segment(dir.y);
    o.depth = max(seg.x, 1000.0);
    let len = seg.y - seg.x;
    if (len <= 0.0) {
        o.depth = 10000.0;
        return o;
    }
    let steps = clamp(len / 50.0, 40.0, 96.0);
    let dt = len / steps;
    let n = i32(ceil(steps)) + 8;
    let k = (1.0 - dir.y * dir.y) / (2.0 * EARTH_R);
    let mu = dot(dir, F.sun_dir.xyz);
    let ph = vec3f(cloud_phase(mu, 1.0), cloud_phase(mu, 0.5), cloud_phase(mu, 0.25));
    // sunlight as it reaches the middle of the layer: it reddens at sunset before the ground does
    let sun = sun_light_at(mix(F.clouds.x, F.clouds.y, 0.3)) * SUN_GAIN;
    let sky_amb = SKY.amb_up.rgb * AMB_GAIN;
    let ground_amb = SKY.amb_ground.rgb * 0.8; // bounced up from the valley

    var T = 1.0;
    var scat = vec3f(0.0);
    var wsum = 0.0;
    var dsum = 0.0;
    var t = seg.x + dt * jitter;
    // empty space is crossed in double steps with the cheap density; the first hit steps back
    // and switches to full steps until the ray has been clear for a while
    var fine = 0;
    for (var i = 0; i < n; i++) {
        if (t > seg.y) {
            break;
        }
        let alt = F.cam_pos.y + dir.y * t + k * t * t;
        let p = F.cam_pos.xyz + dir * t;
        let lod = log2(max(t * C.params.y, 1.0) / SHAPE_TEXEL);
        if (fine == 0) {
            if (cloud_density(p, alt, lod, false) > 0.0) {
                fine = 6;
                t = max(t - dt, seg.x);
            } else {
                t += dt * 2.0;
            }
            continue;
        }
        let d = cloud_density(p, alt, lod, true);
        fine -= 1;
        if (d > 0.002) {
            fine = 6;
            let od = sun_optical_depth(p, alt, lod);
            // multiple scattering as octaves with weaker extinction and a flatter phase
            let ms = exp(-od) * ph.x + 0.6 * exp(-od * 0.3) * ph.y + 0.45 * exp(-od * 0.1) * ph.z;
            let hf = saturate((alt - F.clouds.x) / (F.clouds.y - F.clouds.x));
            // skylight reaches thin and high parts; deep in a thick cloud little of it is left
            let sky_vis = mix(0.15, 1.0, exp(-od * 0.1)) * mix(0.45, 1.0, hf);
            let S = sun * ms + sky_amb * sky_vis + ground_amb;
            let ts = exp(-d * EXTINCTION * dt);
            let a = T * (1.0 - ts);
            scat += S * a;
            wsum += a;
            dsum += a * t;
            T *= ts;
            if (T < 0.02) {
                T = 0.0; // call it opaque, or the sun disk shines through thick cloud
                break;
            }
        }
        t += dt;
    }
    o.scat = scat;
    o.trans = T;
    if (wsum > 1e-4) {
        o.depth = dsum / wsum;
    }
    return o;
}

fn ray_dir(px: vec2f) -> vec3f {
    let uv = (px + 0.5) * C.size.zw;
    let pn = F.inv_view_proj_rot * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    return normalize(pn.xyz / pn.w);
}

// Interleaved gradient noise, shifted every frame.
fn ign(px: vec2f, frame: u32) -> f32 {
    let p = px + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

// Last frame's value for the point `depth` metres along dir, or -1 when it was off screen.
fn reproject(dir: vec3f, depth: f32) -> vec4f {
    let q = dir * depth + C.cam_delta.xyz;
    let clip = C.prev_vp_rot * vec4f(q, 1.0);
    if (clip.w <= 0.0) {
        return vec4f(-1.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        return vec4f(-1.0);
    }
    return textureSampleLevel(t_history, s_clamp, uv, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn march_clouds(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(C.size.xy);
    let base = id.xy * 2u;
    if (any(base >= size)) {
        return;
    }
    // which pixel of the 2x2 block gets a fresh ray this frame
    let frame = u32(C.params.x);
    let f = frame % 4u;
    let pick = vec2u(select(0u, 1u, f == 1u || f == 2u), select(0u, 1u, f == 1u || f == 3u));
    let mp = min(base + pick, size - 1u);
    let mdir = ray_dir(vec2f(mp));
    let m = march(mdir, ign(vec2f(mp), frame));
    // distant clouds fade into the sky behind them
    let fade = 1.0 - exp(-m.depth * AERIAL);
    let fresh = vec4f(mix(m.scat, sky_radiance(mdir) * (1.0 - m.trans), fade), m.trans);

    let history = C.cam_delta.w > 0.5;
    for (var i = 0u; i < 4u; i++) {
        let px = base + vec2u(i & 1u, i >> 1u);
        if (any(px >= size)) {
            continue;
        }
        var out = fresh;
        if (history) {
            let h = reproject(ray_dir(vec2f(px)), m.depth);
            if (h.a >= 0.0) {
                out = select(h, mix(h, fresh, 0.5), all(px == mp));
            }
        }
        textureStore(o_cloud, px, clamp(out, vec4f(0.0), vec4f(64.0, 64.0, 64.0, 1.0)));
    }
}

// ---------------------------------------------------------------------------
// Cloud shadow map: for each texel, a point on the cloud base, the sun's transmittance marched
// from it towards the sun through the layer (direct, and scattered down as diffuse light).
// Texel i holds the world point (i + 0.5) * texel + k * extent nearest the map's centre, so
// the map wraps around and stays put as the camera moves. Ground (and air) at any height looks it up where
// its own sun ray crosses the cloud base (cloud_shadow_uv), so one map serves every altitude.

@group(1) @binding(7) var o_shadow: texture_storage_2d<rgba8unorm, write>;

const SHADOW_STEPS: i32 = 24;

@compute @workgroup_size(8, 8, 1)
fn cloud_shadow_map(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(o_shadow);
    var px = id.xy;
    if (C.params.z > 0.5) {
        // one 8x8 tile of each 2x2 block of tiles per frame (whole tiles keep each
        // workgroup's noise reads close together)
        let f = u32(C.params.x) % 4u;
        px = ((id.xy / 8u) * 2u + vec2u(f & 1u, f >> 1u)) * 8u + id.xy % 8u;
    }
    if (any(px >= size)) {
        return;
    }
    let extent = 1.0 / F.cloud_shadow.z;
    let p0 = (vec2f(px) + 0.5) * (extent / f32(size.x));
    let q = p0 + extent * round((F.cloud_shadow.xy - p0) / extent);
    let L = F.sun_dir.xyz;
    let Ls = normalize(vec3f(L.x, max(L.y, CLOUD_SHADOW_MIN_MU), L.z));
    let len = min((F.clouds.y - F.clouds.x) / Ls.y, 14000.0);
    let dt = len / f32(SHADOW_STEPS);
    // about one shape texel per map texel
    let lod = log2(max(1.0 / (F.cloud_shadow.z * f32(size.x)), SHAPE_TEXEL) / SHAPE_TEXEL);
    var od = 0.0;
    for (var i = 0; i < SHADOW_STEPS; i++) {
        let s = (f32(i) + 0.5) * dt;
        let p = vec3f(q.x, F.clouds.x, q.y) + Ls * s;
        // height above the curved ground, as the view march sees it
        let hd = p.xz - F.cam_pos.xz;
        let alt = p.y + dot(hd, hd) / (2.0 * EARTH_R);
        // the base shape without the detail erosion is fatter than the clouds we see; take
        // off the erosion's average (0.45 x a mean detail value of about 0.5)
        od += max(cloud_density(p, alt, lod, false) - 0.22 * F.clouds.w, 0.0) / 0.78 * dt;
    }
    od *= EXTINCTION;
    // r: the direct beam. g: sunlight the cloud scatters down to the ground as diffuse light,
    // from the two-stream total transmittance 1 / (1 + 3/4 (1 - g) tau) of a scattering layer
    // with g = 0.85. Without it everything under a cloud would only get the blue sky's light.
    let direct = exp(-od);
    let total = 1.0 / (1.0 + 0.1125 * od);
    textureStore(o_shadow, px, vec4f(direct, max(total - direct, 0.0), 0.0, 1.0));
}
