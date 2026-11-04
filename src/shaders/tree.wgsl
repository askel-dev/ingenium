// Instanced tree rendering. Vertex buffer 0 = tree mesh, vertex buffer 1 = the
// culled list of tree indices written by cull.wgsl (one u32 per instance).

@group(1) @binding(0) var<storage, read> trees: array<vec4u>;
@group(2) @binding(0) var<uniform> P: Pass;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) attr: vec4f,   // x kind (0 bark, 1 leaf card, 2 canopy hull), y sway, z ao
    @location(4) tree: u32,
}

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) info: vec4u,   // species, kind, seed, lod
    @location(4) misc: vec4f,                      // ao, sway, local height, stand tint
}

@vertex
fn vs(v: VIn) -> VOut {
    let t = trees[v.tree];
    let base = bitcast<vec3f>(t.xyz);
    let packed = t.w;
    let species = packed & 3u;
    let rot = f32((packed >> 2u) & 255u) / 255.0 * TAU;
    let scale = 0.75 + f32((packed >> 10u) & 255u) / 255.0 * 0.55;
    let seed = packed >> 18u;
    let c = cos(rot);
    let s = sin(rot);
    let lp0 = v.pos * scale;
    let lp = vec3f(c * lp0.x - s * lp0.z, lp0.y, s * lp0.x + c * lp0.z);
    let n = vec3f(c * v.nrm.x - s * v.nrm.z, v.nrm.y, s * v.nrm.x + c * v.nrm.z);

    let time = F.cam_pos.w;
    let sway = v.attr.y;
    let phase = f32(seed) * 0.013 + base.x * 0.02 + base.z * 0.017;
    let gust = sin(time * 0.55 + phase) * 0.5 + 0.5;
    let wdir = vec3f(F.wind.x, 0.0, F.wind.y);
    let s2 = sway * sway;
    var off = wdir * s2 * (0.2 + 0.55 * gust) * F.wind.z * scale;
    off += vec3f(sin(time * 1.6 + phase * 3.0), 0.0, cos(time * 1.25 + phase * 2.0)) * s2 * 0.07 * F.wind.z * scale;
    let is_leaf = select(0.0, 1.0, v.attr.x > 0.5 && v.attr.x < 1.5);
    let fl = dot(v.pos, vec3f(3.1, 1.7, 2.3));
    off += vec3f(sin(time * 6.0 + fl), cos(time * 5.3 + fl * 1.3), sin(time * 4.7 + fl * 0.7)) * 0.025 * is_leaf * F.wind.z;

    let wp = base + lp + off;

    // LOD estimate for the debug tint (mirrors cull.wgsl closely enough)
    let center = base + vec3f(0.0, 8.0 * scale, 0.0);
    let jitter = 1.0 + (u01(pcg(packed)) - 0.5) * 0.25;
    let d = distance(P.ref_cam.xyz, center) / scale * jitter;
    var lod = 3u;
    if (d < F.lod.x) { lod = 0u; } else if (d < F.lod.y) { lod = 1u; } else if (d < F.lod.z) { lod = 2u; }

    var o: VOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    o.nrm = n;
    o.uv = v.uv;
    o.info = vec4u(species, u32(v.attr.x + 0.5), seed, lod);
    // neighbouring trees look alike (same soil, same age), so part of the colour varies by stand
    let stand = gnoise(base.xz * 0.012 + vec2f(3.3, -8.1)) * 0.6 + gnoise(base.xz * 0.05) * 0.4;
    o.misc = vec4f(v.attr.z, sway, v.pos.y, stand);
    return o;
}

fn bark(species: u32, uv: vec2f) -> vec3f {
    let n1 = gnoise(vec2f(uv.x * 7.0, uv.y * 1.3));
    let n2 = gnoise(vec2f(uv.x * 23.0, uv.y * 5.0));
    if (species == 1u) {
        let marks = smoothstep(0.3, 0.6, gnoise(vec2f(uv.x * 3.0, uv.y * 7.0)));
        let c = vec3f(0.62, 0.60, 0.56) * (0.88 + 0.12 * n2);
        return mix(c, vec3f(0.05, 0.045, 0.04), marks * 0.85);
    }
    let ridges = abs(n1);
    var base = vec3f(0.16, 0.14, 0.12);
    if (species == 2u) {
        base = vec3f(0.20, 0.12, 0.08);
    }
    return base * (0.5 + 0.7 * ridges + 0.2 * n2);
}

fn canopy_color(species: u32) -> vec3f {
    switch species {
        case 1u: { return vec3f(0.075, 0.12, 0.028); }
        case 2u: { return vec3f(0.03, 0.055, 0.026); }
        default: { return vec3f(0.055, 0.10, 0.02); }
    }
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
    let aw = max(fwidth(tex.a), 1e-4);
    let species = in.info.x;
    let kind = in.info.y;
    let seed = in.info.z;
    // Per-tree colour: brightness, a hue between blue-green and yellow-green, the odd tree
    // that is paler or turning early, and a stand-wide shift shared with its neighbours.
    let r1 = u01(pcg(seed * 7u + 3u));
    let r2 = u01(pcg(seed * 13u + 5u));
    let r3 = u01(pcg(seed * 29u + 11u));
    let stand = in.misc.w;
    var tint = vec3f(mix(0.66, 1.12, r1));
    tint *= mix(vec3f(0.84, 1.0, 1.18), vec3f(1.22, 1.06, 0.68), saturate(r2 * 0.8 + stand * 0.35 + 0.1));
    tint *= mix(0.86, 1.12, stand * 0.5 + 0.5);
    tint = mix(tint, tint * vec3f(1.55, 1.2, 0.5), smoothstep(0.93, 1.0, r3) * 0.75);
    var n = normalize(in.nrm);
    let L = F.sun_dir.xyz;
    let V = normalize(F.cam_pos.xyz - in.wp);
    let ao = in.misc.x;

    var albedo: vec3f;
    var alpha = 1.0;
    if (kind == 1u) {
        albedo = tex.rgb * tint;
        alpha = clamp((tex.a - 0.5) / aw + 0.5, 0.0, 1.0);
    } else if (kind == 0u) {
        albedo = bark(species, in.uv);
    } else {
        let q = in.wp * 1.4;
        let nz = gnoise(q.xz + q.y * 0.7) * 0.5 + gnoise(q.xz * 2.3 - q.y) * 0.3;
        albedo = canopy_color(species) * tint * (0.95 + 0.45 * nz);
    }

    var sh = shadow_at(in.wp, n);
    if (kind == 2u) {
        sh = mix(sh, 1.0, 0.4);
    }
    // the baked sky visibility is for the ground: up in the crown much more of the sky is open
    let up = saturate(in.misc.z / 12.0);
    let vis = mix(sky_vis(in.wp.xz), 1.0, select(up * 0.85, 0.75, kind == 2u));
    let amb = ambient_vis(n, in.wp, vis);
    let sun = sun_light(in.wp);
    var col: vec3f;
    if (kind == 0u) {
        col = albedo * (sun * max(dot(n, L), 0.0) * sh + amb * ao);
    } else {
        let wrap = max((dot(n, L) + 0.45) / 1.45, 0.0);
        let trans = pow(max(dot(-V, L), 0.0), 5.0) * 0.3;
        col = albedo * (sun * (wrap * 0.95 + trans) * sh * mix(0.45, 1.0, ao) + amb * ao);
        // Leaves are waxy: a soft, broad highlight from the sun (the normals are rounded over
        // the crown, so it lands where the crown faces between sun and eye) and a sky sheen
        // at grazing angles. Uncoloured, as on any dielectric.
        // (a rough surface: the Fresnel rise towards grazing angles is mostly lost)
        let H = normalize(L + V);
        let ndv = saturate(dot(n, V));
        let fres = 0.04 + 0.16 * pow(1.0 - ndv, 4.0);
        let spec = pow(max(dot(n, H), 0.0), 12.0) * fres * max(dot(n, L), 0.0);
        let sheen = SKY.amb_up.rgb * F.sun_color.w * vis * fres * 0.35;
        col += (sun * spec * sh + sheen * ao) * select(1.0, 0.5, kind == 2u);
    }
    if (F.debug.x > 0.5) {
        col = mix(col, lod_tint(in.info.w) * 0.3, 0.6);
    }
    col = apply_fog(col, in.wp);
    return vec4f(col, alpha);
}

@fragment
fn fs_shadow(in: VOut) {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
    if (in.info.y == 1u && tex.a < 0.5) {
        discard;
    }
}
