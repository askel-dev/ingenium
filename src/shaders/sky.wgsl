// Sky, sun disk and a drifting cloud layer, drawn behind everything (reversed-Z depth 0).

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    let ndc = uv * 2.0 - 1.0;
    var o: VOut;
    o.clip = vec4f(ndc, 0.0, 1.0);
    o.ndc = ndc;
    return o;
}

fn cloud_density(p: vec2f) -> f32 {
    let shape = fbm(p * 0.00042, 4);
    let cover = mix(0.3, -0.1, F.fog.w);
    let core = smoothstep(cover, cover + 0.22, shape);
    // erode the edges with finer noise so they billow instead of streaking
    let detail = fbm(p * 0.0021 + vec2f(3.0, 7.0), 4);
    return clamp(core * 1.25 - (1.0 - core) * 0.6 + detail * 0.35 * (1.0 - core * 0.5), 0.0, 1.0);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let pn = F.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let dir = normalize(pn.xyz / pn.w - F.cam_pos.xyz);
    var col = sky_radiance(dir);
    let L = F.sun_dir.xyz;
    let mu = dot(dir, L);
    col += F.sun_color.rgb * smoothstep(0.99965, 0.9998, mu) * 30.0;

    if (dir.y > 0.003) {
        let t = (2200.0 - F.cam_pos.y) / dir.y;
        let drift = F.wind.xy * F.cam_pos.w * 8.0;
        let cp = F.cam_pos.xz + dir.xz * t + drift;
        let d = cloud_density(cp);
        let d_sun = cloud_density(cp + L.xz * 350.0);
        let lit = clamp(1.0 - (d_sun - d * 0.5) * 1.2, 0.0, 1.0);
        let shade = mix(F.sky_horizon.rgb * 0.75 + vec3f(0.06), F.sun_color.rgb * 0.33 + F.sky_zenith.rgb * 0.25, lit);
        let silver = F.sun_color.rgb * pow(max(mu, 0.0), 12.0) * (1.0 - d) * 0.6;
        let cloud_col = shade * (1.0 - 0.25 * d) + silver;
        let fade = smoothstep(0.0, 0.18, dir.y);
        col = mix(col, cloud_col, d * fade * 0.95);
        // haze towards the horizon
        col = mix(col, F.sky_horizon.rgb, (1.0 - smoothstep(0.0, 0.12, dir.y)) * 0.5);
    }
    return vec4f(col, 1.0);
}
