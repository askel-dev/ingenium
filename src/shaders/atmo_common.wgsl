// Physically based atmosphere (Hillaire 2020, "A Scalable and Production Ready Sky and
// Atmosphere Rendering Technique"): the medium, and the lookup-table parameterisations shared by
// the LUT passes (atmosphere.wgsl) and every shader that reads the tables (globals.wgsl).
// Distances in km. `haze` scales the aerosol (Mie) density: 1 = clear, 3+ = hazy.

const PLANET_R: f32 = 6360.0;
const ATMO_R: f32 = 6460.0;
const RAYLEIGH_SCAT: vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_SCAT: f32 = 3.996e-3;
const MIE_EXT: f32 = 4.40e-3;
const OZONE_ABS: vec3f = vec3f(0.650e-3, 1.881e-3, 0.085e-3);
const MIE_G: f32 = 0.8;
const SUN_ANGULAR_R: f32 = 0.004651; // 0.2665 degrees
const TRANS_LUT: vec2f = vec2f(256.0, 64.0);
const MS_LUT: f32 = 32.0;
const SKY_LUT: vec2f = vec2f(192.0, 108.0);

struct Medium {
    rayleigh: vec3f,
    mie: f32,
    ext: vec3f,
}

fn medium(h: f32, haze: f32) -> Medium {
    let hh = max(h, 0.0);
    let r = exp(-hh / 8.0);
    let m = exp(-hh / 1.2) * haze;
    let o = max(0.0, 1.0 - abs(hh - 25.0) / 15.0);
    var s: Medium;
    s.rayleigh = RAYLEIGH_SCAT * r;
    s.mie = MIE_SCAT * m;
    s.ext = RAYLEIGH_SCAT * r + MIE_EXT * m + OZONE_ABS * o;
    return s;
}

fn rayleigh_phase(mu: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

// Cornette-Shanks, a Henyey-Greenstein variant with a better back lobe.
fn mie_phase(mu: f32) -> f32 {
    let g2 = MIE_G * MIE_G;
    let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + mu * mu) / pow(max(1.0 + g2 - 2.0 * MIE_G * mu, 1e-4), 1.5);
}

// Distance from radius r along a ray with cosine mu to the sphere of radius R; < 0 when missed.
fn ray_sphere(r: f32, mu: f32, R: f32) -> f32 {
    let d = r * r * (mu * mu - 1.0) + R * R;
    if (d < 0.0) {
        return -1.0;
    }
    let s = sqrt(d);
    let t0 = -r * mu - s;
    if (t0 > 0.0) {
        return t0;
    }
    return -r * mu + s;
}

fn ray_hits_ground(r: f32, mu: f32) -> bool {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + PLANET_R * PLANET_R >= 0.0;
}

// Transmittance LUT (Bruneton): x = view zenith cosine, y = altitude, both remapped so the
// horizon and the ground get most of the texels.
fn trans_uv(r: f32, mu: f32) -> vec2f {
    let H = sqrt(ATMO_R * ATMO_R - PLANET_R * PLANET_R);
    let rho = sqrt(max(r * r - PLANET_R * PLANET_R, 0.0));
    let disc = r * r * (mu * mu - 1.0) + ATMO_R * ATMO_R;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let d_min = ATMO_R - r;
    let d_max = rho + H;
    let x_mu = (d - d_min) / (d_max - d_min);
    let x_r = rho / H;
    // keep the samples on texel centres
    return (vec2f(x_mu, x_r) * (TRANS_LUT - 1.0) + 0.5) / TRANS_LUT;
}

fn trans_rmu(uv: vec2f) -> vec2f {
    let x = (uv * TRANS_LUT - 0.5) / (TRANS_LUT - 1.0);
    let H = sqrt(ATMO_R * ATMO_R - PLANET_R * PLANET_R);
    let rho = H * x.y;
    let r = sqrt(rho * rho + PLANET_R * PLANET_R);
    let d_min = ATMO_R - r;
    let d_max = rho + H;
    let d = d_min + x.x * (d_max - d_min);
    var mu = 1.0;
    if (d > 0.0) {
        mu = clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2f(r, mu);
}

// Sky-view LUT: x = azimuth from the sun (0 .. pi, the sky is symmetric about the sun's
// vertical), y = elevation with the horizon in the middle and sqrt spacing towards it.
// `beta` is how far the true horizon dips below horizontal at this altitude.
fn horizon_dip(r: f32) -> f32 {
    return acos(clamp(sqrt(max(r * r - PLANET_R * PLANET_R, 0.0)) / r, -1.0, 1.0));
}

fn sky_uv(elev: f32, azim: f32, r: f32) -> vec2f {
    let beta = horizon_dip(r);
    var v: f32;
    if (elev >= -beta) {
        v = 0.5 + 0.5 * sqrt(saturate((elev + beta) / (0.5 * PI + beta)));
    } else {
        v = 0.5 - 0.5 * sqrt(saturate((-beta - elev) / (0.5 * PI - beta)));
    }
    let u = sqrt(saturate(azim / PI));
    return (vec2f(u, v) * (SKY_LUT - 1.0) + 0.5) / SKY_LUT;
}

fn sky_elev_azim(uv: vec2f, r: f32) -> vec2f {
    let x = (uv * SKY_LUT - 0.5) / (SKY_LUT - 1.0);
    let beta = horizon_dip(r);
    var elev: f32;
    if (x.y >= 0.5) {
        let t = (x.y - 0.5) * 2.0;
        elev = t * t * (0.5 * PI + beta) - beta;
    } else {
        let t = (0.5 - x.y) * 2.0;
        elev = -beta - t * t * (0.5 * PI - beta);
    }
    return vec2f(elev, x.x * x.x * PI);
}

// Ground haze: an extra aerosol layer on top of the standard atmosphere, densest at the valley
// floor. fog = (extinction per metre at height fog.z, falloff per metre, fog.z, unused). It is
// part of the sky-view table and the aerial-perspective volume alike, so a distant ridge fades
// into exactly the sky behind it.
const FOG_MS: f32 = 2.0; // the haze's own multiple scattering, which the table leaves out

fn fog_km(h: f32, fog: vec4f) -> f32 {
    return fog.x * 1000.0 * exp(-fog.y * (max(h, 0.0) * 1000.0 - fog.z));
}

// Optical depth of the haze above altitude h (km) towards the sun (flat-layer approximation).
fn fog_sun_od(h: f32, mu_s: f32, fog: vec4f) -> f32 {
    return fog.x / max(fog.y, 1e-6) * exp(-fog.y * (max(h, 0.0) * 1000.0 - fog.z)) / max(mu_s, 0.05);
}
