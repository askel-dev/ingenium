// Instanced tree rendering. Vertex buffer 0 = tree mesh, vertex buffer 1 = the
// culled list of tree indices written by cull.wgsl (one u32 per instance).
//
// Near trees are meshes (vs, fs); beyond them every tree is an impostor (vs_imp, fs_imp): a
// quad facing the camera (or the sun, in the shadow passes) that shows the views of the tree's
// mesh baked from the three directions nearest the one it is seen from (impostor.rs).

@group(1) @binding(0) var<storage, read> trees: array<vec4u>;
@group(1) @binding(1) var t_imp_a: texture_2d_array<f32>;   // albedo, coverage
@group(1) @binding(2) var t_imp_b: texture_2d_array<f32>;   // normal (octahedral, tree space), ao, leafness
@group(1) @binding(3) var<uniform> IMP: ImpKinds;
@group(2) @binding(0) var<uniform> P: Pass;

struct ImpKinds {
    kinds: array<vec4f, 12>,   // per species * 4 + variant: x frame centre height, y frame half size (m)
}

struct VIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) attr: vec4f,   // x kind (0 bark, 1 leaf card, 2 canopy hull, 3 fallen wood, 4 rock, 5 cut wood), y sway, z ao, w bark girth
    @location(4) tree: u32,
}

struct VOut {
    // invariant: the prepass and the shading pass must compute bit-identical depth
    @builtin(position) @invariant clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) info: vec4u,   // species, kind, seed, lod
    @location(4) misc: vec4f,                      // ao, bark girth (m), local height, stand tint
    @location(5) bark: vec3f,                      // bark texture coordinates (tiles), wraps around
    @location(6) @interpolate(flat) layer: u32,    // foliage atlas layer (leaf cards)
}

// The LOD the culling picked (as cull.wgsl does, from the same bounding-sphere centre), for
// the debug tint. The forest-floor plants have no kind of their own here; theirs is unused.
fn lod_estimate(i: Inst) -> u32 {
    let center = i.base + vec3f(0.0, IMP.kinds[min(min(i.species, 2u) * 4u + i.variant, 11u)].x * i.scale, 0.0);
    let jitter = 1.0 + (u01(pcg(i.seed * 977u + i.species)) - 0.5) * 0.25;
    let d = distance(P.ref_cam.xyz, center) / i.scale * jitter;
    return select(select(2u, 1u, d < F.lod.y), 0u, d < F.lod.x);
}

@vertex
fn vs(v: VIn) -> VOut {
    let i = inst(trees[v.tree]);
    let lp = inst_vec(i, v.pos);
    let n = inst_normal(i, v.nrm);

    let time = F.cam_pos.w;
    let sway = v.attr.y;
    let phase = f32(i.seed) * 0.013 + i.base.x * 0.02 + i.base.z * 0.017;
    let gust = sin(time * 0.55 + phase) * 0.5 + 0.5;
    let wdir = vec3f(F.wind.x, 0.0, F.wind.y);
    let s2 = sway * sway;
    var off = wdir * s2 * (0.2 + 0.55 * gust) * F.wind.z * i.scale;
    off += vec3f(sin(time * 1.6 + phase * 3.0), 0.0, cos(time * 1.25 + phase * 2.0)) * s2 * 0.07 * F.wind.z * i.scale;
    let is_leaf = select(0.0, 1.0, v.attr.x > 0.5 && v.attr.x < 1.5);
    let fl = dot(v.pos, vec3f(3.1, 1.7, 2.3));
    off += vec3f(sin(time * 6.0 + fl), cos(time * 5.3 + fl * 1.3), sin(time * 4.7 + fl * 0.7)) * 0.025 * is_leaf * F.wind.z;

    var wp = i.base + lp + off;
    if (v.attr.x > 2.5 && v.attr.x < 3.5) {
        // fallen wood lies on the ground and follows it
        wp.y += ground_h(wp.xz) - ground_h(i.base.xz);
    }

    var o: VOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    o.nrm = n;
    o.uv = v.uv;
    o.info = vec4u(i.species, u32(v.attr.x + 0.5), i.seed, lod_estimate(i));
    o.misc = vec4f(v.attr.z, v.attr.w, v.pos.y, stand_of(i.base));
    o.bark = bark_uv(i.species, v.uv, v.attr.w, i.seed);
    o.layer = select(0u, u32(v.attr.w + 0.5), v.attr.x > 0.5 && v.attr.x < 1.5);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.layer));
    let texn = textureSample(t_leaf_n, s_repeat, in.uv, i32(in.layer));
    let aw = max(fwidth(tex.a), 1e-4);
    let species = in.info.x;
    let kind = in.info.y;
    let tint = tree_tint(in.info.z, in.misc.w);
    let n = normalize(in.nrm);
    let L = F.sun_dir.xyz;
    let V = normalize(F.cam_pos.xyz - in.wp);
    let ao = in.misc.x;
    // derivatives for the bark's texture lookups and tangent frame (taken here, where every
    // pixel of the quad still runs)
    let dwx = dpdx(in.wp);
    let dwy = dpdy(in.wp);
    let dbx = dpdx(in.bark.xy);
    let dby = dpdy(in.bark.xy);
    let dux = dpdx(in.uv);
    let duy = dpdy(in.uv);

    // Beyond the near cascade the trees cast their impostors, a quad through the middle of the
    // tree: the shadow is looked up a little towards the sun, out of the tree's own.
    let dist = distance(in.wp, F.cam_pos.xyz);
    let lift = select(0.0, 2.5, kind == 1u || kind == 2u) * smoothstep(F.cascade.x * 0.75, F.cascade.x, dist);
    var sh = shadow_at(in.wp + L * lift, n);
    // the baked sky visibility is for the ground: up in the crown much more of the sky is open
    let up = saturate(in.misc.z / 12.0);
    let vis = mix(sky_vis(in.wp.xz), 1.0, select(up * 0.85, 0.75, kind == 2u));

    var col: vec3f;
    var alpha = 1.0;
    if (kind == 1u) {
        alpha = clamp((tex.a - 0.5) / aw + 0.5, 0.0, 1.0);
        // Each leaf's own normal (the scan's relief plus the tilt it was placed at in its
        // cluster), in the card's frame, around the rounded crown normal: every leaf of a card
        // catches the light on its own.
        let ns = perturb(n, texn.xy * 2.0 - 1.0, dwx, dwy, dux, duy);
        col = shade_leaf(tex.rgb * tint, ns, n, ao, texn.w, texn.z, sh, vis, in.wp, V);
    } else if (kind == 0u || kind == 3u) {
        let m = mat_fetch(i32(min(species, 2u)), in.bark.xy, dbx, dby);
        var albedo = m.a.rgb;
        var rough = m.b.z;
        if (species == 1u) {
            let bb = birch_bark(albedo, in.bark, max(length(dbx), length(dby)), in.misc.y);
            albedo = bb.rgb;
            rough = mix(rough, 0.55, bb.a);
        }
        var ns = perturb(n, m.b.xy, dwx, dwy, dbx, dby);
        // a normal tilted away from the eye on a silhouette shows its back: bend it back
        ns = normalize(ns + n * max(-dot(ns, V), 0.0) * 1.1);
        var tex_ao = m.b.w;
        if (kind == 3u) {
            // moss on the upper side of what has lain on the ground a while
            let moss = smoothstep(0.45, 0.95, n.y * 0.8 + gnoise(in.wp.xz * 1.9) * 0.55) * 0.75;
            albedo = mix(albedo * 0.85, MOSS, moss);
            rough = mix(rough, 0.95, moss);
            ns = normalize(mix(ns, n, moss * 0.6));
            tex_ao = mix(tex_ao, 1.0, moss * 0.5);
        }
        col = shade_bark(albedo, ns, ao, tex_ao, rough, sh, vis, in.wp, V);
    } else if (kind == 4u) {
        // rock: the rock scan projected along the three axes by world position
        let tile = 0.45;
        var w3 = pow(abs(n), vec3f(4.0));
        w3 /= w3.x + w3.y + w3.z;
        let mx = mat_fetch(MAT_ROCK, in.wp.zy * tile, dwx.zy * tile, dwy.zy * tile);
        let my = mat_fetch(MAT_ROCK, in.wp.xz * tile, dwx.xz * tile, dwy.xz * tile);
        let mz = mat_fetch(MAT_ROCK, in.wp.xy * tile, dwx.xy * tile, dwy.xy * tile);
        var albedo = mx.a.rgb * w3.x + my.a.rgb * w3.y + mz.a.rgb * w3.z;
        var rough = mx.b.z * w3.x + my.b.z * w3.y + mz.b.z * w3.z;
        var tex_ao = mx.b.w * w3.x + my.b.w * w3.y + mz.b.w * w3.z;
        var ns = normalize(perturb(n, mx.b.xy, dwx, dwy, dwx.zy * tile, dwy.zy * tile) * w3.x
            + perturb(n, my.b.xy, dwx, dwy, dwx.xz * tile, dwy.xz * tile) * w3.y
            + perturb(n, mz.b.xy, dwx, dwy, dwx.xy * tile, dwy.xy * tile) * w3.z);
        let moss = smoothstep(0.55, 1.0, n.y * 0.85 + gnoise(in.wp.xz * 1.3) * 0.5 + gnoise(in.wp.xz * 4.1) * 0.15) * 0.75;
        albedo = mix(albedo, MOSS, moss);
        rough = mix(rough, 0.95, moss);
        ns = normalize(mix(ns, n, moss * 0.5));
        col = shade_bark(albedo, ns, ao, tex_ao, rough, sh, vis, in.wp, V);
    } else if (kind == 5u) {
        // cut or broken wood: growth rings, weathered grey, the bark's dark rim
        let rr = length(in.uv);
        let rings = 0.5 + 0.5 * sin(rr * 40.0 + gnoise(in.uv * 3.0) * 4.0);
        var albedo = mix(vec3f(0.30, 0.21, 0.12), vec3f(0.21, 0.145, 0.08), rings * 0.6);
        albedo = mix(albedo, vec3f(0.16, 0.15, 0.13), 0.4 + 0.3 * gnoise(in.uv * 5.0));
        albedo = mix(albedo, vec3f(0.07, 0.055, 0.04), smoothstep(0.86, 0.98, rr));
        col = shade_bark(albedo, n, ao, 1.0, 0.85, sh, vis, in.wp, V);
    } else {
        // The inner hull (spruce LOD1) has no leaves of its own: the leaves' average colour,
        // lit like them, and a soft highlight where it faces between sun and eye, as the
        // leaves' highlights average out to.
        let q = in.wp * 1.4;
        let nz = gnoise(q.xz + q.y * 0.7) * 0.5 + gnoise(q.xz * 2.3 - q.y) * 0.3;
        let albedo = canopy_color(species) * tint * (0.95 + 0.45 * nz);
        sh = mix(sh, 1.0, 0.4);
        col = shade_leaf(albedo, n, n, ao, 0.6, 0.6, sh, vis, in.wp, V);
    }
    if (F.debug.x > 0.5) {
        col = mix(col, lod_tint(in.info.w) * 0.3, 0.6);
    }
    col = apply_fog(col, in.wp);
    return vec4f(col, alpha);
}

// Leaf coverage, as `fs` computes it: the alpha edge sharpened to about a pixel wide.
fn leaf_alpha(in: VOut) -> f32 {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.layer));
    let aw = max(fwidth(tex.a), 1e-4);
    return select(1.0, clamp((tex.a - 0.5) / aw + 0.5, 0.0, 1.0), in.info.y == 1u);
}

// The depth prepass with MSAA: the alpha only drives alpha-to-coverage (no colour is written).
@fragment
fn fs_depth(in: VOut) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, leaf_alpha(in));
}

// The depth prepass without MSAA: an alpha test.
@fragment
fn fs_depth_test(in: VOut) -> @location(0) vec4f {
    if (leaf_alpha(in) < 0.5) {
        discard;
    }
    return vec4f(0.0);
}

@fragment
fn fs_shadow(in: VOut) {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.layer));
    if (in.info.y == 1u && tex.a < 0.5) {
        discard;
    }
}

// ---------------------------------------------------------------------------------------------
// Impostors

struct VImp {
    @builtin(position) @invariant clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) uv01: vec4f,                         // frame coordinates (-1..1) in frames 0 and 1
    @location(2) uv2q: vec4f,                         // in frame 2; the quad's own coordinates
    @location(3) @interpolate(flat) w: vec4f,         // the frames' weights; w = lift for the shadow lookup
    @location(4) @interpolate(flat) frames: vec4u,    // the frames' indices; w = atlas layer
    @location(5) @interpolate(flat) info: vec4u,      // species, seed, packed rotation, mirror
    @location(6) misc: vec4f,                         // height above the base (m), stand tint
    @location(7) @interpolate(flat) right: vec4f,     // the quad's axes in the world; w = stretch across
    @location(8) @interpolate(flat) up: vec4f,        // w = stretch up
}

@vertex
fn vs_imp(v: VIn) -> VImp {
    let i = inst(trees[v.tree]);
    let layer = i.species * 4u + i.variant;
    let K = IMP.kinds[layer];
    let size3 = inst_size(i);
    let center = i.base + vec3f(0.0, K.x * size3.y, 0.0);
    // towards the camera, or the sun in the shadow passes
    var V = F.sun_dir.xyz;
    if (P.params.x < 0.5) {
        V = normalize(F.cam_pos.xyz - center);
    }
    let r0 = cross(vec3f(0.0, 1.0, 0.0), V);
    let right = select(normalize(r0), vec3f(1.0, 0.0, 0.0), dot(r0, r0) < 1e-8);
    let up = cross(V, right);
    let size = K.y * i.scale * max(i.stretch.x, i.stretch.y) * 1.04;
    let corner = center + (right * v.pos.x + up * v.pos.y) * size;

    // The view ray through this corner, in the tree's own space around the frame centre
    // (where the frames were baked): from the camera, or along the sun's rays.
    var o: vec3f;
    var dir: vec3f;
    if (P.params.x < 0.5) {
        o = inst_local(i, F.cam_pos.xyz - center);
        dir = inst_local(i, corner - center) - o;
    } else {
        o = inst_local(i, corner - center);
        dir = inst_local(i, -V);
    }
    // The three frames around the view direction on the hemi-octahedral grid, and their
    // weights (a view from below the horizon takes the horizon's).
    var dl = inst_local(i, V);
    dl = normalize(vec3f(dl.x, max(dl.y, 0.0), dl.z));
    let g = (hemi_oct(dl) * 0.5 + 0.5) * f32(IMP_GRID - 1u);
    let gi = clamp(floor(g), vec2f(0.0), vec2f(f32(IMP_GRID - 2u)));
    let f = g - gi;
    var gs: array<vec2f, 3>;
    var w: vec3f;
    if (f.x + f.y < 1.0) {
        gs = array<vec2f, 3>(gi, gi + vec2f(1.0, 0.0), gi + vec2f(0.0, 1.0));
        w = vec3f(1.0 - f.x - f.y, f.x, f.y);
    } else {
        gs = array<vec2f, 3>(gi + vec2f(1.0), gi + vec2f(0.0, 1.0), gi + vec2f(1.0, 0.0));
        w = vec3f(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    // where the ray meets each frame's plane, in that frame's coordinates
    var uv: array<vec2f, 3>;
    var idx: array<u32, 3>;
    for (var k = 0; k < 3; k++) {
        let dk = frame_dir(gs[k]);
        let rk = frame_right(dk);
        let uk = cross(dk, rk);
        let t = -dot(o, dk) / dot(dir, dk);
        let hit = o + dir * t;
        uv[k] = vec2f(dot(hit, rk), dot(hit, uk)) / K.y;
        idx[k] = u32(gs[k].y) * IMP_GRID + u32(gs[k].x);
    }

    var out: VImp;
    out.clip = P.view_proj * vec4f(corner, 1.0);
    out.wp = corner;
    out.uv01 = vec4f(uv[0], uv[1]);
    out.uv2q = vec4f(uv[2], v.pos.xy);
    out.w = vec4f(w, size * 0.5);
    out.frames = vec4u(idx[0], idx[1], idx[2], layer);
    let rot_q = u32(round(fract(atan2(i.s, i.c) / TAU) * 65536.0)) & 0xFFFFu;
    out.info = vec4u(i.species, i.seed, rot_q, select(0u, 1u, i.mirror < 0.0));
    out.misc = vec4f(corner.y - i.base.y, stand_of(i.base), 0.0, 0.0);
    out.right = vec4f(right, i.stretch.x);
    out.up = vec4f(up, i.stretch.y);
    return out;
}

struct ImpSample {
    albedo: vec3f,
    alpha: f32,
    n: vec3f,        // tree space
    ao: f32,
    leaf: f32,
}

fn imp_frame(idx: u32, fuv: vec2f) -> vec2f {
    let g = vec2f(f32(idx % IMP_GRID), f32(idx / IMP_GRID));
    let c = clamp(fuv, vec2f(-1.0), vec2f(1.0));
    return (g + vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5)) / f32(IMP_GRID);
}

fn imp_sample(in: VImp, full: bool) -> ImpSample {
    let layer = i32(in.frames.w);
    let uvs = array<vec2f, 3>(in.uv01.xy, in.uv01.zw, in.uv2q.xy);
    var s: ImpSample;
    var wa = 0.0;
    var n = vec3f(0.0);
    var col = vec3f(0.0);
    var ao = 0.0;
    var leaf = 0.0;
    // Both maps hold their values multiplied by the coverage, so that filtering doesn't pull
    // in the empty texels around a leaf: divided by it again here.
    for (var k = 0; k < 3; k++) {
        let auv = imp_frame(in.frames[k], uvs[k]);
        let a = textureSample(t_imp_a, s_repeat, auv, layer);
        let b = textureSample(t_imp_b, s_repeat, auv, layer);
        let wk = in.w[k];
        wa += wk * a.a;
        if (full) {
            let ik = 1.0 / max(a.a, 1e-3);
            col += a.rgb * wk;
            n += oct_decode(b.xy * ik) * (wk * a.a);
            ao += b.z * wk;
            leaf += b.w * wk;
        }
    }
    let inv = 1.0 / max(wa, 1e-4);
    s.alpha = wa;
    s.albedo = col * inv;
    s.n = n * inv;
    s.ao = ao * inv;
    s.leaf = leaf * inv;
    return s;
}

// The blended frames' coverage, its edge sharpened to about a pixel wide as for the leaves.
fn imp_alpha(a: f32) -> f32 {
    return clamp((a - 0.5) / max(fwidth(a), 1e-4) + 0.5, 0.0, 1.0);
}

@fragment
fn fs_imp(in: VImp) -> @location(0) vec4f {
    let s = imp_sample(in, true);
    let alpha = imp_alpha(s.alpha);
    let ang = f32(in.info.z) / 65536.0 * TAU;
    var i: Inst;
    i.c = cos(ang);
    i.s = sin(ang);
    i.mirror = select(1.0, -1.0, in.info.w == 1u);
    i.stretch = vec2f(in.right.w, in.up.w);
    i.scale = 1.0;
    let ns = inst_normal(i, s.n);
    let V = normalize(F.cam_pos.xyz - in.wp);
    let right = in.right.xyz;
    let up = in.up.xyz;
    let fwd = cross(right, up);
    // the crown as a whole is rounded: a sphere's normal over the quad
    let q = in.uv2q.zw * 1.3;
    let nc = normalize(right * q.x + up * q.y + fwd * sqrt(max(1.0 - dot(q, q), 0.1)));
    let n = normalize(ns + nc * 0.6);
    let L = F.sun_dir.xyz;
    // the baked sky visibility is for the ground: up in the crown much more of the sky is open
    let up_frac = saturate(in.misc.x / 12.0);
    let vis = mix(sky_vis(in.wp.xz), 1.0, up_frac * 0.85);
    // out of the tree's own shadow (its impostor faces the sun through its middle)
    var sh = shadow_at(in.wp + L * in.w.w, n);
    sh = mix(sh, 1.0, 0.3);
    let tint = tree_tint(in.info.y, in.misc.y);
    let ao = s.ao;
    let leaf = shade_leaf(s.albedo * tint, ns, n, ao, 0.65, 0.5, sh, vis, in.wp, V);
    let bark = shade_bark(s.albedo, ns, ao, 1.0, 0.8, sh, vis, in.wp, V);
    var col = mix(bark, leaf, saturate(s.leaf));
    if (F.debug.x > 0.5) {
        col = mix(col, lod_tint(2u) * 0.3, 0.6);
    }
    col = apply_fog(col, in.wp);
    return vec4f(col, alpha);
}

@fragment
fn fs_imp_depth(in: VImp) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, imp_alpha(imp_sample(in, false).alpha));
}

@fragment
fn fs_imp_depth_test(in: VImp) -> @location(0) vec4f {
    if (imp_alpha(imp_sample(in, false).alpha) < 0.5) {
        discard;
    }
    return vec4f(0.0);
}

@fragment
fn fs_imp_shadow(in: VImp) {
    if (imp_sample(in, false).alpha < 0.5) {
        discard;
    }
}
