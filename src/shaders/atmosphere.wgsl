// Atmosphere lookup tables (Hillaire 2020), built by compute passes:
//   transmittance  256x64  sun light left after crossing the atmosphere      (when haze changes)
//   multi-scatter   32x32  light scattered more than once, per unit sun      (when haze changes)
//   sky-view       192x108 sky radiance around the camera                    (every frame)
//   irradiance        1    sun at the camera, ambient sky and ground light   (every frame)
// Shaders then read the sky from the sky-view table instead of marching it per pixel.

struct Atmo {
    sun_dir: vec4f,   // xyz towards the sun
    sun_illum: vec4f, // rgb sun illuminance above the atmosphere, w = haze
    params: vec4f,    // x camera radius (km), y overcast 0..1, z ground albedo
    fog: vec4f,       // ground haze, see fog_km
}

struct SkyOut {
    sun: vec4f,        // rgb sun illuminance reaching the camera
    amb_up: vec4f,     // irradiance / pi from the sky on an upward surface
    amb_ground: vec4f, // light bounced up from the ground, as radiance
    horizon: vec4f,    // rgb average radiance at the horizon, w = illuminance on flat ground
}

@group(0) @binding(0) var<uniform> A: Atmo;
@group(0) @binding(1) var s_lin: sampler;
@group(0) @binding(2) var t_trans: texture_2d<f32>;
@group(0) @binding(3) var t_ms: texture_2d<f32>;
@group(0) @binding(4) var t_sky: texture_2d<f32>;
@group(0) @binding(5) var o_trans: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var o_ms: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var o_sky: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> o_irr: SkyOut;

fn haze() -> f32 {
    return A.sun_illum.w;
}

fn trans(r: f32, mu: f32) -> vec3f {
    return textureSampleLevel(t_trans, s_lin, trans_uv(r, mu), 0.0).rgb;
}

fn multi(r: f32, mu_s: f32) -> vec3f {
    let uv = vec2f(mu_s * 0.5 + 0.5, (r - PLANET_R) / (ATMO_R - PLANET_R));
    return textureSampleLevel(t_ms, s_lin, (uv * (MS_LUT - 1.0) + 0.5) / MS_LUT, 0.0).rgb;
}

// Sun light arriving at radius r with sun cosine mu_s, zero in the planet's shadow.
fn sun_trans(r: f32, mu_s: f32) -> vec3f {
    if (ray_hits_ground(r, mu_s)) {
        return vec3f(0.0);
    }
    return trans(r, mu_s);
}

@compute @workgroup_size(8, 8, 1)
fn transmittance(@builtin(global_invocation_id) id: vec3u) {
    if (any(vec2f(id.xy) >= TRANS_LUT)) {
        return;
    }
    let rm = trans_rmu((vec2f(id.xy) + 0.5) / TRANS_LUT);
    let r = rm.x;
    let mu = rm.y;
    let len = ray_sphere(r, mu, ATMO_R);
    let n = 40;
    let dt = len / f32(n);
    var od = vec3f(0.0);
    for (var i = 0; i < n; i++) {
        let t = (f32(i) + 0.5) * dt;
        let h = sqrt(r * r + t * t + 2.0 * r * mu * t) - PLANET_R;
        od += medium(h, haze()).ext * dt;
    }
    textureStore(o_trans, id.xy, vec4f(exp(-od), 1.0));
}

// Hillaire's multiple scattering: from each point, the second-order light gathered over the
// sphere (L2) and the fraction re-scattered (f_ms); the infinite series sums to L2 / (1 - f_ms).
@compute @workgroup_size(8, 8, 1)
fn multiscatter(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= vec2u(u32(MS_LUT)))) {
        return;
    }
    let uv = (vec2f(id.xy) + 0.5) / MS_LUT;
    let mu_s = uv.x * 2.0 - 1.0;
    let r = PLANET_R + uv.y * (ATMO_R - PLANET_R);
    let sun = vec3f(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    let albedo = A.params.z;
    var L2 = vec3f(0.0);
    var fms = vec3f(0.0);
    let N = 8;
    for (var j = 0; j < N; j++) {
        for (var k = 0; k < N; k++) {
            // uniform directions over the sphere
            let ct = 1.0 - 2.0 * (f32(j) + 0.5) / f32(N);
            let st = sqrt(max(1.0 - ct * ct, 0.0));
            let ph = 2.0 * PI * (f32(k) + 0.5) / f32(N);
            let dir = vec3f(st * cos(ph), ct, st * sin(ph));
            let mu = dir.y;
            let ground = ray_hits_ground(r, mu);
            let len = select(ray_sphere(r, mu, ATMO_R), ray_sphere(r, mu, PLANET_R), ground);
            let steps = 20;
            let dt = len / f32(steps);
            var T = vec3f(1.0);
            var L = vec3f(0.0);
            var f = vec3f(0.0);
            for (var i = 0; i < steps; i++) {
                let t = (f32(i) + 0.5) * dt;
                let p = vec3f(0.0, r, 0.0) + dir * t;
                let pr = length(p);
                let m = medium(pr - PLANET_R, haze());
                let scat = m.rayleigh + vec3f(m.mie);
                let st_ = exp(-m.ext * dt);
                let integ = (1.0 - st_) / max(m.ext, vec3f(1e-9));
                let ts = sun_trans(pr, dot(p / pr, sun));
                L += T * scat * ts * (1.0 / (4.0 * PI)) * integ;
                f += T * scat * integ;
                T *= st_;
            }
            if (ground) {
                let p = vec3f(0.0, r, 0.0) + dir * len;
                let up = normalize(p);
                let mus = dot(up, sun);
                L += T * sun_trans(PLANET_R, mus) * max(mus, 0.0) * albedo / PI;
            }
            L2 += L;
            fms += f;
        }
    }
    let w = 1.0 / f32(N * N);
    L2 *= w;
    fms *= w;
    textureStore(o_ms, id.xy, vec4f(L2 / (1.0 - fms), 1.0));
}

// Single scattering marched from the camera, plus the multi-scatter table. The ground haze
// lives in the first few kilometres, so that stretch gets its own dense set of steps.
fn sky_march(r: f32, dir: vec3f, sun: vec3f) -> vec3f {
    let mu = dir.y;
    let ground = ray_hits_ground(r, mu);
    let len = select(ray_sphere(r, mu, ATMO_R), ray_sphere(r, mu, PLANET_R), ground);
    if (len <= 0.0) {
        return vec3f(0.0);
    }
    let cos_t = dot(dir, sun);
    let ph_r = rayleigh_phase(cos_t);
    let ph_m = mie_phase(cos_t);
    // the cloud deck keeps the sun off the haze under an overcast sky
    let fog_sun = 1.0 - 0.8 * A.params.y;
    let near = min(len, 40.0);
    var T = vec3f(1.0);
    var L = vec3f(0.0);
    for (var seg = 0; seg < 2; seg++) {
        let a = select(near, 0.0, seg == 0);
        let b = select(len, near, seg == 0);
        if (b <= a) {
            continue;
        }
        let steps = select(20, 24, seg == 0);
        var t_prev = a;
        for (var i = 0; i < steps; i++) {
            // denser steps near the start of each stretch, where the air is thickest
            let x = (f32(i) + 0.5) / f32(steps);
            let t = a + (b - a) * x * x;
            let x1 = (f32(i) + 1.0) / f32(steps);
            let t1 = a + (b - a) * x1 * x1;
            let dt = t1 - t_prev;
            t_prev = t1;
            let p = vec3f(0.0, r, 0.0) + dir * t;
            let pr = length(p);
            let h = pr - PLANET_R;
            let m = medium(h, haze());
            let fd = fog_km(h, A.fog);
            let mus = dot(p / pr, sun);
            let ts = sun_trans(pr, mus) * exp(-fog_sun_od(h, mus, A.fog));
            let S = (m.rayleigh * ph_r + vec3f((m.mie + fd * fog_sun) * ph_m)) * ts + (m.rayleigh + vec3f(m.mie + fd * FOG_MS)) * multi(pr, mus);
            let ext = m.ext + vec3f(fd);
            let st_ = exp(-ext * dt);
            L += T * S * (1.0 - st_) / max(ext, vec3f(1e-9));
            T *= st_;
        }
    }
    if (ground) {
        let p = vec3f(0.0, r, 0.0) + dir * len;
        let mus = dot(normalize(p), sun);
        L += T * sun_trans(PLANET_R, mus) * max(mus, 0.0) * A.params.z / PI;
    }
    return L * A.sun_illum.rgb;
}

// The sun in the sky-view frame: up = y, sun in the xy plane.
fn local_sun() -> vec3f {
    let mu_s = clamp(A.sun_dir.y, -1.0, 1.0);
    return vec3f(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
}

fn view_dir(elev: f32, azim: f32) -> vec3f {
    return vec3f(cos(elev) * cos(azim), sin(elev), cos(elev) * sin(azim));
}

@compute @workgroup_size(8, 8, 1)
fn skyview(@builtin(global_invocation_id) id: vec3u) {
    if (any(vec2f(id.xy) >= SKY_LUT)) {
        return;
    }
    let r = A.params.x;
    let ea = sky_elev_azim((vec2f(id.xy) + 0.5) / SKY_LUT, r);
    let L = sky_march(r, view_dir(ea.x, ea.y), local_sun());
    textureStore(o_sky, id.xy, vec4f(L, 1.0));
}

fn sky_lut(elev: f32, azim: f32, r: f32) -> vec3f {
    return textureSampleLevel(t_sky, s_lin, sky_uv(elev, azim, r), 0.0).rgb;
}

fn lum(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(1, 1, 1)
fn irradiance() {
    let r = A.params.x;
    let ov = A.params.y;
    let mu_s = A.sun_dir.y;
    // fraction of the sun's disk above the (dipped) horizon
    let elev_s = asin(clamp(mu_s, -1.0, 1.0));
    let beta = horizon_dip(r);
    let vis = smoothstep(-beta - SUN_ANGULAR_R, -beta + SUN_ANGULAR_R, elev_s);
    let sun = A.sun_illum.rgb * trans(r, mu_s) * vis;

    // cosine-weighted sky over the upper hemisphere, and a ring at the horizon
    var e_up = vec3f(0.0);
    var hor = vec3f(0.0);
    let NE = 16;
    let NA = 16;
    let de = 0.5 * PI / f32(NE);
    let da = PI / f32(NA);
    for (var a = 0; a < NA; a++) {
        let az = (f32(a) + 0.5) * da;
        for (var e = 0; e < NE; e++) {
            let el = (f32(e) + 0.5) * de;
            e_up += sky_lut(el, az, r) * sin(el) * cos(el) * de * da;
        }
        hor += sky_lut(0.02, az, r);
    }
    e_up *= 2.0; // azimuth covered 0..pi, the sky is symmetric
    hor /= f32(NA);

    // under overcast the cloud deck replaces the blue sky with a brighter grey and blocks the sun
    let sun_ground = sun * (1.0 - 0.8 * ov);
    let grey_up = vec3f(lum(e_up + sun * max(mu_s, 0.0) * 0.35));
    e_up = mix(e_up, grey_up, ov);
    hor = mix(hor, vec3f(lum(grey_up) / PI * 0.8), ov);

    let albedo = A.params.z;
    let on_flat = sun_ground * max(mu_s, 0.0) + e_up;
    var o: SkyOut;
    o.sun = vec4f(sun, 1.0);
    o.amb_up = vec4f(e_up / PI, 0.0);
    o.amb_ground = vec4f(on_flat * albedo / PI, 0.0);
    o.horizon = vec4f(hor, lum(on_flat));
    o_irr = o;
}
