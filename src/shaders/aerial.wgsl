// Aerial perspective volume (Hillaire 2020, section 5.5): a froxel grid over the screen. Each
// column marches its view ray through the same medium as the sky-view table (air, aerosols and
// the ground haze) and stores, per depth slice, the light scattered towards the camera so far
// (o_scat) and how much of what lies behind still gets through (o_trans). Surfaces then apply
// it with two texture reads (apply_fog in globals.wgsl). The sun's part is shadowed by the
// cloud shadow map, so the haze under a cloud is darker too, and beyond the first couple of
// hundred metres by the terrain shadow map (sun_height.wgsl): the air in a valley the mountains
// shade is dark, and the sun shines in shafts between the peaks.
//
// Near the camera that is far too coarse for shafts between trees. `aerial_near` marches a finer
// grid over the first NEAR_DIST metres through the far shadow cascade instead, and stores the
// sunlight its shadows take out of the haze (o_shade), which apply_fog then subtracts. Under the
// canopy the air is mostly dark, and lit where the sun gets through the gaps.

@group(1) @binding(0) var t_ms: texture_2d<f32>;
@group(1) @binding(1) var t_cs: texture_2d<f32>;
@group(1) @binding(2) var o_scat: texture_storage_3d<rgba16float, write>;
@group(1) @binding(3) var o_trans: texture_storage_3d<rgba16float, write>;
@group(1) @binding(4) var o_shade: texture_storage_3d<rgba16float, write>;

const SUBSTEPS: i32 = 4;
const NEAR_SUBSTEPS: i32 = 2;

fn multi(r: f32, mu_s: f32) -> vec3f {
    let uv = vec2f(mu_s * 0.5 + 0.5, (r - PLANET_R) / (ATMO_R - PLANET_R));
    return textureSampleLevel(t_ms, s_clamp, (uv * (MS_LUT - 1.0) + 0.5) / MS_LUT, 0.0).rgb;
}

fn sun_trans(r: f32, mu_s: f32) -> vec3f {
    if (ray_hits_ground(r, mu_s)) {
        return vec3f(0.0);
    }
    return textureSampleLevel(t_trans, s_clamp, trans_uv(r, mu_s), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn aerial(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(o_scat);
    if (any(id.xy >= size.xy)) {
        return;
    }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size.xy);
    let pn = F.inv_view_proj_rot * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let dir = normalize(pn.xyz / pn.w);
    let sun = F.sun_dir.xyz;
    let cos_t = dot(dir, sun);
    let ph_r = rayleigh_phase(cos_t);
    let ph_m = mie_phase(cos_t);
    let haze = F.atmos.x;
    let ov = F.atmos.y;
    let origin = vec3f(0.0, cam_radius(), 0.0);

    var T = vec3f(1.0);
    var L = vec3f(0.0);
    var t_prev = 0.0;
    let n = size.z;
    for (var s = 0u; s < n; s++) {
        let x = (f32(s) + 0.5) / f32(n);
        let t_end = AP_DIST * x * x;
        let dt = (t_end - t_prev) / f32(SUBSTEPS);
        for (var k = 0; k < SUBSTEPS; k++) {
            let t = t_prev + (f32(k) + 0.5) * dt;
            // planet-centred position in km, for the tables
            let pk = origin + dir * (t * 0.001);
            let r = length(pk);
            let h = r - PLANET_R;
            let mu_s = dot(pk / r, sun);
            let m = medium(h, haze);
            let wp = F.cam_pos.xyz + dir * t;
            let fh = forest_haze_km(wp);
            let fd = fog_km(h, F.fog) + fh;
            let cs = cloud_shadow_from(t_cs, wp);
            // the terrain's shadow, and the forest canopy's for the air under it
            let shade = mix(1.0, far_ground_vis(sun_height_vis(wp)), ap_far_shadow(t));
            let sun_t = sun_trans(r, mu_s) * exp(-fog_sun_od(h, mu_s, F.fog));
            let scat = m.rayleigh + vec3f(m.mie + fd);
            // direct sun with its phase function; the clouds' diffuse light from above
            // (radiance E / pi over the upper hemisphere, scattered isotropically)
            var S = (m.rayleigh * ph_r + vec3f((m.mie + fd) * ph_m)) * sun_t * cs.x * shade;
            S += scat * sun_t * max(mu_s, 0.0) * cs.y / (2.0 * PI);
            // light scattered around the sky; under a cloud deck it comes from the grey cloud
            // base and the ground instead, so take it from the ambient light there
            // (under the canopy the haze among the trees sees only part of the sky)
            let ms = (m.rayleigh + vec3f(m.mie + (fd - fh * (1.0 - sky_vis(wp.xz))) * FOG_MS)) * multi(r, mu_s);
            let deck = scat * (SKY.amb_up.rgb + SKY.amb_ground.rgb) * 0.5 / F.sun_color.rgb;
            S += mix(ms, deck, ov);
            let ext = m.ext + vec3f(fd);
            let st = exp(-ext * (dt * 0.001));
            L += T * S * (1.0 - st) / max(ext, vec3f(1e-9));
            T *= st;
        }
        t_prev = t_end;
        textureStore(o_scat, vec3u(id.xy, s), vec4f(L * F.sun_color.rgb, 1.0));
        textureStore(o_trans, vec3u(id.xy, s), vec4f(T, 1.0));
    }
}

// Sunlight in cascade ci at wp, averaged over a square `size` metres across (a froxel), so the
// leaves' fine shadows don't sparkle in the march.
fn cascade_lit(ci: i32, wp: vec3f, size: f32) -> f32 {
    let lp = F.shadow_vp[ci] * vec4f(wp, 1.0);
    let uv = lp.xy * vec2f(0.5, -0.5) + 0.5;
    if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0)) || lp.z <= 0.0 || lp.z >= 1.0) {
        return 1.0;
    }
    let texel_m = select(F.cascade.w, F.cascade.z, ci == 0);
    let r = max(size / texel_m, 1.0) * 0.35 / F.world.w;
    var s = 0.0;
    for (var i = 0; i < 4; i++) {
        let o = vec2f(f32(i & 1) * 2.0 - 1.0, f32(i >> 1) * 2.0 - 1.0);
        s += textureSampleCompareLevel(t_shadow, s_shadow, uv + vec2f(o.x - o.y * 0.5, o.y + o.x * 0.5) * r, ci, lp.z);
    }
    return s * 0.25;
}

// Sunlight at wp in the air near the camera, t along the view ray: the near cascade's leaves,
// then the far cascade (every tree as its impostor), and the terrain's shadow from further
// towards the sun. `size`: the froxel's width there.
fn near_sunlight(wp: vec3f, t: f32, size: f32) -> f32 {
    var s: f32;
    if (t < F.cascade.x) {
        s = mix(cascade_lit(0, wp, size), cascade_lit(1, wp, size), smoothstep(F.cascade.x * 0.8, F.cascade.x, t));
    } else {
        s = cascade_lit(1, wp, size);
    }
    return min(s, sun_height_vis(wp).x);
}

@compute @workgroup_size(8, 8, 1)
fn aerial_near(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(o_shade);
    if (any(id.xy >= size.xy)) {
        return;
    }
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size.xy);
    let pn = F.inv_view_proj_rot * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let dir = normalize(pn.xyz / pn.w);
    let sun = F.sun_dir.xyz;
    let cos_t = dot(dir, sun);
    let ph_r = rayleigh_phase(cos_t);
    let ph_m = mie_phase(cos_t);
    let origin = vec3f(0.0, cam_radius(), 0.0);
    // where in each step the samples sit, different per column so the steps don't band
    let jit = fract(52.9829189 * fract(dot(vec2f(id.xy), vec2f(0.06711056, 0.00583715))));
    // a froxel's width per metre of distance: the angle to the next row's ray
    let pn2 = F.inv_view_proj_rot * vec4f(uv.x * 2.0 - 1.0, 1.0 - (uv.y + 1.0 / f32(size.y)) * 2.0, 1.0, 1.0);
    let spread = length(normalize(pn2.xyz / pn2.w) - dir);

    var T = vec3f(1.0);
    var R = vec3f(0.0);
    var R_air = vec3f(0.0); // without the haze among the trees, which the sky leaves out
    var t_prev = 0.0;
    let n = size.z;
    for (var s = 0u; s < n; s++) {
        let x = (f32(s) + 0.5) / f32(n);
        let t_end = NEAR_DIST * x * x;
        let dt = (t_end - t_prev) / f32(NEAR_SUBSTEPS);
        for (var k = 0; k < NEAR_SUBSTEPS; k++) {
            let t = t_prev + (f32(k) + jit) * dt;
            let pk = origin + dir * (t * 0.001);
            let r = length(pk);
            let h = r - PLANET_R;
            let mu_s = dot(pk / r, sun);
            let m = medium(h, F.atmos.x);
            let fd = fog_km(h, F.fog);
            let wp = F.cam_pos.xyz + dir * t;
            let fh = forest_haze_km(wp);
            // the sunlight the coarse volume counted here and the shadow takes away
            let dark = (1.0 - near_sunlight(wp, t, t * spread)) * (1.0 - ap_far_shadow(t));
            let sun_t = sun_trans(r, mu_s) * exp(-fog_sun_od(h, mu_s, F.fog)) * cloud_shadow(wp).x * dark;
            let S_air = (m.rayleigh * ph_r + vec3f((m.mie + fd) * ph_m)) * sun_t;
            let ext = m.ext + vec3f(fd + fh);
            let st = exp(-ext * (dt * 0.001));
            let k = T * (1.0 - st) / max(ext, vec3f(1e-9));
            R += k * (S_air + vec3f(fh * ph_m) * sun_t);
            R_air += k * S_air;
            T *= st;
        }
        t_prev = t_end;
        // a: the share of it the sky (which has no haze among the trees) takes off
        textureStore(o_shade, vec3u(id.xy, s), vec4f(R * F.sun_color.rgb, luma(R_air) / max(luma(R), 1e-12)));
    }
}
