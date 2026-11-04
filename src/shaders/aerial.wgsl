// Aerial perspective volume (Hillaire 2020, section 5.5): a froxel grid over the screen. Each
// column marches its view ray through the same medium as the sky-view table (air, aerosols and
// the ground haze) and stores, per depth slice, the light scattered towards the camera so far
// (o_scat) and how much of what lies behind still gets through (o_trans). Surfaces then apply
// it with two texture reads (apply_fog in globals.wgsl). The sun's part is shadowed by the
// cloud shadow map, so the haze under a cloud is darker too.

@group(1) @binding(0) var t_ms: texture_2d<f32>;
@group(1) @binding(1) var t_cs: texture_2d<f32>;
@group(1) @binding(2) var o_scat: texture_storage_3d<rgba16float, write>;
@group(1) @binding(3) var o_trans: texture_storage_3d<rgba16float, write>;

const SUBSTEPS: i32 = 2;

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
            let fd = fog_km(h, F.fog);
            let cs = cloud_shadow_from(t_cs, F.cam_pos.xyz + dir * t);
            let sun_t = sun_trans(r, mu_s) * exp(-fog_sun_od(h, mu_s, F.fog));
            let scat = m.rayleigh + vec3f(m.mie + fd);
            // direct sun with its phase function; the clouds' diffuse light from above
            // (radiance E / pi over the upper hemisphere, scattered isotropically)
            var S = (m.rayleigh * ph_r + vec3f((m.mie + fd) * ph_m)) * sun_t * cs.x;
            S += scat * sun_t * max(mu_s, 0.0) * cs.y / (2.0 * PI);
            // light scattered around the sky; under a cloud deck it comes from the grey cloud
            // base and the ground instead, so take it from the ambient light there
            let ms = (m.rayleigh + vec3f(m.mie + fd * FOG_MS)) * multi(r, mu_s);
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
