// CDLOD terrain: one 33x33 grid mesh instanced per quadtree patch, morphed
// towards the next-coarser grid as it approaches its LOD range so there are no cracks.

@group(1) @binding(0) var<uniform> P: Pass;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @location(0) quad: vec4f) -> VOut {
    var g = vec2f(f32(vid % 33u), f32(vid / 33u));
    let origin = quad.xy;
    let size = quad.z;
    let level = u32(quad.w);
    let p0 = origin + g * (size / 32.0);
    let d = distance(p0, P.ref_cam.xz);
    let m = F.morph[level];
    let k = clamp((d - m.x) / (m.y - m.x), 0.0, 1.0);
    g = g - fract(g * 0.5) * 2.0 * k;
    let p = origin + g * (size / 32.0);
    let h = height_at(p) + detail_h(p) * (1.0 - smoothstep(80.0, 140.0, d));
    let wp = vec3f(p.x, h, p.y);
    var o: VOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let p = in.wp.xz;
    let dat = data_at(p);
    let dist = distance(in.wp, F.cam_pos.xyz);

    // macro normal from the baked map + procedural micro detail up close
    var n = normal_at(p);
    let df = 1.0 - smoothstep(60.0, 140.0, dist);
    let e = 0.25;
    let hx = detail_h(p + vec2f(e, 0.0)) - detail_h(p - vec2f(e, 0.0));
    let hz = detail_h(p + vec2f(0.0, e)) - detail_h(p - vec2f(0.0, e));
    n = normalize(n + vec3f(-hx, 0.0, -hz) / (2.0 * e) * df);

    let forest = dat.x;
    let rock = dat.y;
    let flowers = dat.z;
    let vari = dat.w;
    let nz = fbm(p * 0.08, 3) * 0.5 + 0.5;
    let nz2 = gnoise(p * 0.9) * 0.5 + 0.5;
    let nz3 = gnoise(p * 0.013) * 0.5 + 0.5;

    var grass = mix(vec3f(0.032, 0.062, 0.012), vec3f(0.072, 0.105, 0.022), vari);
    grass = mix(grass, vec3f(0.12, 0.13, 0.04), nz3 * 0.35);
    grass *= 0.8 + 0.35 * nz;
    grass = mix(grass, vec3f(0.30, 0.26, 0.09), flowers * 0.18 * nz2);

    var floor_c = mix(vec3f(0.10, 0.07, 0.04), vec3f(0.19, 0.13, 0.075), nz2) * (0.8 + 0.3 * nz);
    floor_c = mix(floor_c, vec3f(0.06, 0.11, 0.03), smoothstep(0.55, 0.8, nz) * 0.7);
    var albedo = mix(grass, floor_c, smoothstep(0.35, 0.75, forest));

    let rock_c = mix(vec3f(0.11, 0.105, 0.10), vec3f(0.22, 0.21, 0.19), nz2) * (0.75 + 0.35 * nz);
    albedo = mix(albedo, rock_c, smoothstep(0.4, 0.7, rock + (nz - 0.5) * 0.3));

    let L = F.sun_dir.xyz;
    let ndl = max(dot(n, L), 0.0);
    let sh = shadow_at(in.wp, n);
    // far away there is no shadow map: darken dense forest floor instead
    let canopy = mix(1.0, 0.35, smoothstep(0.45, 0.85, forest) * smoothstep(F.cascade.y * 0.6, F.cascade.y, dist));
    let amb = ambient(n) * mix(1.0, 0.5, smoothstep(0.3, 0.9, forest));
    var col = albedo * (F.sun_color.rgb * ndl * sh * canopy + amb);
    col = apply_fog(col, in.wp);
    return vec4f(col, 1.0);
}

// ---------------------------------------------------------------------------
// Far-forest canopy shell: the terrain grid lifted by forest density and shaded
// like treetops. It stands in for individual trees beyond ~1.3 km, so a million
// distant trees cost nothing per instance (the "proxy without instance data").

struct CanopyOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) ground: f32,
}

fn shell_height(p: vec2f) -> f32 {
    let forest = data_at(p).x;
    let crowns = gnoise(p * 0.19) * 0.5 + gnoise(p * 0.43) * 0.25;
    return 15.5 * smoothstep(0.2, 0.7, forest) * (0.85 + 0.25 * crowns);
}

@vertex
fn vs_canopy(@builtin(vertex_index) vid: u32, @location(0) quad: vec4f) -> CanopyOut {
    var g = vec2f(f32(vid % 33u), f32(vid / 33u));
    let origin = quad.xy;
    let size = quad.z;
    let level = u32(quad.w);
    let p0 = origin + g * (size / 32.0);
    let d = distance(p0, P.ref_cam.xz);
    let m = F.morph[level];
    let k = clamp((d - m.x) / (m.y - m.x), 0.0, 1.0);
    g = g - fract(g * 0.5) * 2.0 * k;
    let p = origin + g * (size / 32.0);
    let ground = height_at(p);
    let wp = vec3f(p.x, ground + shell_height(p), p.y);
    var o: CanopyOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    o.ground = ground;
    return o;
}

@fragment
fn fs_canopy(in: CanopyOut) -> @location(0) vec4f {
    let p = in.wp.xz;
    let dat = data_at(p);
    let dist = distance(in.wp, F.cam_pos.xyz);
    let lift = in.wp.y - in.ground;
    // fade in where individual trees are thinning out, and only where there is forest
    let alpha = smoothstep(0.3, 0.45, dat.x) * smoothstep(F.lod.w * 0.62, F.lod.w * 0.8, dist) * smoothstep(1.0, 4.0, lift);

    // crown-scale bumps for shading
    let e = 0.6;
    let c0 = gnoise(p * 0.19) * 0.5 + gnoise(p * 0.43) * 0.25;
    let cx = gnoise((p + vec2f(e, 0.0)) * 0.19) * 0.5 + gnoise((p + vec2f(e, 0.0)) * 0.43) * 0.25;
    let cz = gnoise((p + vec2f(0.0, e)) * 0.19) * 0.5 + gnoise((p + vec2f(0.0, e)) * 0.43) * 0.25;
    let tn = normal_at(p);
    let n = normalize(tn + vec3f(-(cx - c0), 0.0, -(cz - c0)) / e * 6.0);

    let h = in.ground;
    let spruce = smoothstep(250.0, 380.0, h + fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3) * 140.0);
    var albedo = mix(vec3f(0.04, 0.08, 0.014), vec3f(0.022, 0.042, 0.02), spruce);
    let birchy = smoothstep(0.12, 0.32, fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3)) * (1.0 - spruce);
    albedo = mix(albedo, vec3f(0.06, 0.10, 0.02), birchy * 0.6);
    let tint = gnoise(p * 0.07) * 0.5 + 0.5;
    albedo *= mix(0.75, 1.1, tint);
    // gaps between crowns are darker
    let gap = smoothstep(-0.1, 0.35, c0);

    let L = F.sun_dir.xyz;
    let wrap = max((dot(n, L) + 0.45) / 1.45, 0.0);
    var col = albedo * (F.sun_color.rgb * wrap * mix(0.45, 1.0, gap) + ambient(n) * mix(0.5, 1.0, gap));
    col = apply_fog(col, in.wp);
    return vec4f(col, alpha);
}
