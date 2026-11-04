// Instanced tree rendering. Vertex buffer 0 = tree mesh, vertex buffer 1 = the culled list of
// tree indices written by cull.wgsl (one u32 per instance).
//
// Near trees are meshes (`vs`, `fs`). From LOD2 on, a tree is a quad facing the camera that
// shows its impostor (impostor.rs), blended from the four baked views nearest to the
// direction it is seen from (`vs_impostor`, `fs_impostor`). Both are lit the same way.

@group(1) @binding(0) var<storage, read> trees: array<vec4u>;
@group(1) @binding(2) var t_imp_color: texture_2d_array<f32>;
@group(1) @binding(3) var t_imp_data: texture_2d_array<f32>;
@group(2) @binding(0) var<uniform> P: Pass;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) attr: vec4f,   // x kind (0 bark, 1 leaf card, 2 inner cone), y sway, z ao, w shape slot (-1: undergrowth)
    @location(4) tree: u32,
}

// A tree record (scatter.wgsl), unpacked.
struct Tree {
    base: vec3f,
    packed: u32,
    species: u32,
    cs: vec2f, // cosine and sine of its rotation about y
    scale: f32,
    seed: u32,
}

fn unpack_tree(t: vec4u) -> Tree {
    let rot = f32((t.w >> 2u) & 255u) / 255.0 * TAU;
    return Tree(bitcast<vec3f>(t.xyz), t.w, t.w & 3u, vec2f(cos(rot), sin(rot)), tree_scale(t.w), t.w >> 18u);
}

// Neighbouring trees look alike (same soil, same age), so part of the colour varies by stand.
fn stand_tint(base: vec3f) -> f32 {
    return gnoise(base.xz * 0.012 + vec2f(3.3, -8.1)) * 0.6 + gnoise(base.xz * 0.05) * 0.4;
}

// LOD estimate for the debug tint (mirrors cull.wgsl closely enough)
fn lod_estimate(center: vec3f, t: Tree) -> u32 {
    let jitter = 1.0 + (u01(pcg(t.packed)) - 0.5) * 0.25;
    let d = distance(P.ref_cam.xyz, center) / t.scale * jitter;
    var lod = 3u;
    if (d < F.lod.x) { lod = 0u; } else if (d < F.lod.y) { lod = 1u; } else if (d < F.lod.z) { lod = 2u; }
    return lod;
}

// Per-tree colour: brightness, a hue between blue-green and yellow-green, the odd tree that is
// paler or turning early, and a stand-wide shift shared with its neighbours.
fn tree_tint(seed: u32, stand: f32) -> vec3f {
    let r1 = u01(pcg(seed * 7u + 3u));
    let r2 = u01(pcg(seed * 13u + 5u));
    let r3 = u01(pcg(seed * 29u + 11u));
    var tint = vec3f(mix(0.66, 1.12, r1));
    tint *= mix(vec3f(0.84, 1.0, 1.18), vec3f(1.22, 1.06, 0.68), saturate(r2 * 0.8 + stand * 0.35 + 0.1));
    tint *= mix(0.86, 1.12, stand * 0.5 + 0.5);
    return mix(tint, tint * vec3f(1.55, 1.2, 0.5), smoothstep(0.93, 1.0, r3) * 0.75);
}

// Leaves: wrapped diffuse light, light through the leaves from behind, a soft waxy highlight
// and a sky sheen. `sun_vis` is the share of the sunlight that gets to them, `card` 1 for leaf
// cards and 0 for the spruce's inner cone (opaque, no light through it).
fn shade_leaves(albedo: vec3f, n: vec3f, wp: vec3f, vis: f32, sun_vis: f32, ao: f32, card: f32) -> vec3f {
    let L = F.sun_dir.xyz;
    let V = normalize(F.cam_pos.xyz - wp);
    let sun = sun_light(wp);
    let amb = ambient_vis(n, wp, vis);
    let wrap = max((dot(n, L) + 0.45) / 1.45, 0.0);
    let trans = pow(max(dot(-V, L), 0.0), 5.0) * 0.3;
    var col = albedo * (sun * (wrap * 0.95 + trans) * sun_vis * mix(0.45, 1.0, ao) + amb * ao);
    // Leaves are waxy: a soft, broad highlight from the sun (the normals are rounded over the
    // crown and its clumps, so it lands where they face between sun and eye) and a sky sheen at
    // grazing angles. Uncoloured, as on any dielectric. (a rough surface: the Fresnel rise
    // towards grazing angles is mostly lost)
    let H = normalize(L + V);
    let ndv = saturate(dot(n, V));
    let fres = 0.04 + 0.16 * pow(1.0 - ndv, 4.0);
    let spec = pow(max(dot(n, H), 0.0), 12.0) * fres * max(dot(n, L), 0.0);
    let sheen = SKY.amb_up.rgb * F.sun_color.w * vis * fres * 0.35;
    col += (sun * spec * sun_vis + sheen * ao) * mix(0.5, 1.0, card);
    // Leaves are thin: light falling on one side comes out of the other, yellower. The normals
    // point outwards, so the leaves under a crown glow with the sun and sky on their upper side,
    // as a canopy does seen from below.
    let tcol = albedo * vec3f(1.15, 1.2, 0.55);
    let through = sun * max(dot(-n, L), 0.0) * sun_vis * 0.5 + ambient_vis(-n, wp, vis) * ao * 0.6;
    return col + tcol * through * card;
}

fn shade_bark(albedo: vec3f, n: vec3f, wp: vec3f, vis: f32, sun_vis: f32, ao: f32) -> vec3f {
    return albedo * (sun_light(wp) * max(dot(n, F.sun_dir.xyz), 0.0) * sun_vis + ambient_vis(n, wp, vis) * ao);
}

// ---------------------------------------------------------------------------
// Meshes

struct VOut {
    // invariant: the prepass and the shading pass must compute bit-identical depth
    @builtin(position) @invariant clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) info: vec4u,   // species, kind, seed, lod
    @location(4) misc: vec4f,                      // ao, sway, local height, stand tint
    @location(5) own: vec4f,                       // x = sunlight left after the tree's own crown, yzw = its impostor's centre
}

@vertex
fn vs(v: VIn) -> VOut {
    let t = unpack_tree(trees[v.tree]);
    let lp = rot_y(v.pos * t.scale, t.cs);
    let n = rot_y(v.nrm, t.cs);

    let time = F.cam_pos.w;
    let sway = v.attr.y;
    let phase = f32(t.seed) * 0.013 + t.base.x * 0.02 + t.base.z * 0.017;
    let gust = sin(time * 0.55 + phase) * 0.5 + 0.5;
    let wdir = vec3f(F.wind.x, 0.0, F.wind.y);
    let s2 = sway * sway;
    var off = wdir * s2 * (0.2 + 0.55 * gust) * F.wind.z * t.scale;
    off += vec3f(sin(time * 1.6 + phase * 3.0), 0.0, cos(time * 1.25 + phase * 2.0)) * s2 * 0.07 * F.wind.z * t.scale;
    let is_leaf = select(0.0, 1.0, v.attr.x > 0.5 && v.attr.x < 1.5);
    let fl = dot(v.pos, vec3f(3.1, 1.7, 2.3));
    off += vec3f(sin(time * 6.0 + fl), cos(time * 5.3 + fl * 1.3), sin(time * 4.7 + fl * 0.7)) * 0.025 * is_leaf * F.wind.z;

    let wp = t.base + lp + off;

    var o: VOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    o.nrm = n;
    o.uv = v.uv;
    o.info = vec4u(t.species, u32(v.attr.x + 0.5), t.seed, lod_estimate(t.base + vec3f(0.0, 8.0 * t.scale, 0.0), t));
    o.misc = vec4f(v.attr.z, sway, v.pos.y, stand_tint(t.base));
    // The tree's own crown shades it (the undergrowth has no shape). In the near cascade the
    // leaves' own shadows do that, so it fades in beyond it.
    o.own = vec4f(1.0, wp);
    if (v.attr.w >= 0.0) {
        let s = SHAPES[u32(v.attr.w + 0.5)];
        let fade = smoothstep(F.cascade.x * 0.75, F.cascade.x, distance(wp, F.cam_pos.xyz));
        let lit = crown_light(v.pos, unrot_y(F.sun_dir.xyz, t.cs), s, t.scale);
        o.own = vec4f(mix(1.0, lit, fade), t.base + vec3f(0.0, s.imp.x * t.scale, 0.0));
    }
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
    let aw = max(fwidth(tex.a), 1e-4);
    let species = in.info.x;
    let kind = in.info.y;
    let tint = tree_tint(in.info.z, in.misc.w);
    let n = normalize(in.nrm);
    let L = F.sun_dir.xyz;
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

    // Beyond the near cascade every tree casts its impostor: a plane through its centre, facing
    // the sun. The leaves look the shadow up on the sun's side of their own tree's plane, so the
    // tree doesn't shade itself twice; its own crown's shade is `own.x`.
    let dist = distance(in.wp, F.cam_pos.xyz);
    let far = smoothstep(F.cascade.x * 0.75, F.cascade.x, dist);
    let lift = select(0.0, (max(dot(in.own.yzw - in.wp, L), 0.0) + 1.0) * far, kind != 0u);
    var sh = shadow_at(in.wp + L * lift, n);
    if (kind == 2u) {
        sh = mix(sh, 1.0, 0.4);
    }
    // the baked sky visibility is for the ground: up in the crown much more of the sky is open
    let up = saturate(in.misc.z / 12.0);
    let vis = mix(sky_vis(in.wp.xz), 1.0, select(up * 0.85, 0.75, kind == 2u));
    var col: vec3f;
    if (kind == 0u) {
        col = shade_bark(albedo, n, in.wp, vis, sh, ao);
    } else {
        col = shade_leaves(albedo, n, in.wp, vis, sh * in.own.x, ao, select(0.0, 1.0, kind == 1u));
    }
    if (F.debug.x > 0.5) {
        col = mix(col, lod_tint(in.info.w) * 0.3, 0.6);
    }
    col = apply_fog(col, in.wp);
    return vec4f(col, alpha);
}

// Leaf coverage, as `fs` computes it: the alpha edge sharpened to about a pixel wide.
fn leaf_alpha(in: VOut) -> f32 {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
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
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
    if (in.info.y == 1u && tex.a < 0.5) {
        discard;
    }
}

// ---------------------------------------------------------------------------
// Impostors

struct IOut {
    @builtin(position) @invariant clip: vec4f,
    @location(0) uv01: vec4f,                      // the quad point in views 0 and 1 (0..1 across a view)
    @location(1) uv23: vec4f,                      // and in views 2 and 3
    @location(2) @interpolate(flat) weights: vec4f,
    @location(3) @interpolate(flat) views: vec4u,  // the four views: x + y * IMP_FRAMES
    @location(4) @interpolate(flat) info: vec4u,   // species, slot, seed, lod
    @location(5) quad: vec3f,                      // the quad point (object space, before scaling)
    @location(6) @interpolate(flat) dir: vec4f,    // xyz direction it is seen from (object space), w = scale
    @location(7) @interpolate(flat) base: vec4f,   // xyz tree base, w = stand tint
    @location(8) @interpolate(flat) cs: vec2f,     // cosine and sine of its rotation
    // the directions the four views were baked from (object space)
    @location(9) @interpolate(flat) d0: vec3f,
    @location(10) @interpolate(flat) d1: vec3f,
    @location(11) @interpolate(flat) d2: vec3f,
    @location(12) @interpolate(flat) d3: vec3f,
}

// Where q, a point on the quad (relative to the impostor's centre), shows in the view from d:
// q moved along the direction the tree is seen from onto that view's plane.
fn imp_uv(q: vec3f, dir: vec3f, d: vec3f, half: f32) -> vec2f {
    let ax = imp_axes(d);
    let p = q - dir * (dot(q, d) / max(dot(dir, d), 0.2));
    return vec2f(dot(p, ax.r), -dot(p, ax.u)) / (2.0 * half) + 0.5;
}

@vertex
fn vs_impostor(v: VIn) -> IOut {
    let t = unpack_tree(trees[v.tree]);
    let slot = shape_slot(t.packed);
    let s = SHAPES[slot];
    let center = t.base + vec3f(0.0, s.imp.x * t.scale, 0.0);
    // seen from the camera, or in a shadow pass from the sun
    let shadow = P.params.x > 0.5;
    var view = F.sun_dir.xyz;
    if (!shadow) {
        view = normalize(F.cam_pos.xyz - center);
    }
    let dir = unrot_y(view, t.cs);
    // the quad faces the viewer, a little bigger than the tightly framed views
    let c = cross(vec3f(0.0, 1.0, 0.0), dir);
    let right = select(vec3f(1.0, 0.0, 0.0), normalize(c), dot(c, c) > 1e-10);
    let up = cross(dir, right);
    let q = (right * v.pos.x + up * v.pos.y) * (s.imp.y * 1.1);
    let quad = vec3f(0.0, s.imp.x, 0.0) + q;
    let wp = t.base + rot_y(quad * t.scale, t.cs);

    // the four views around dir, weighted by how close each is
    let g = (imp_encode(dir) * 0.5 + 0.5) * f32(IMP_FRAMES - 1u);
    let i0 = clamp(floor(g), vec2f(0.0), vec2f(f32(IMP_FRAMES - 2u)));
    let f = clamp(g - i0, vec2f(0.0), vec2f(1.0));
    var w = vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let c0 = vec2u(i0);
    var cells = array<vec2u, 4>(c0, c0 + vec2u(1u, 0u), c0 + vec2u(0u, 1u), c0 + vec2u(1u, 1u));
    if (shadow) {
        // a shadow only needs the nearest view
        var k = 0u;
        for (var i = 1u; i < 4u; i++) {
            if (w[i] > w[k]) {
                k = i;
            }
        }
        cells[0] = cells[k];
        w = vec4f(1.0, 0.0, 0.0, 0.0);
    }
    var uv: array<vec2f, 4>;
    var dirs: array<vec3f, 4>;
    var ids: vec4u;
    for (var i = 0u; i < 4u; i++) {
        dirs[i] = imp_frame_dir(cells[i]);
        uv[i] = imp_uv(q, dir, dirs[i], s.imp.y);
        ids[i] = cells[i].x + cells[i].y * IMP_FRAMES;
    }

    var o: IOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.uv01 = vec4f(uv[0], uv[1]);
    o.uv23 = vec4f(uv[2], uv[3]);
    o.weights = w;
    o.views = ids;
    o.info = vec4u(t.species, slot, t.seed, max(lod_estimate(center, t), 2u));
    o.quad = quad;
    o.dir = vec4f(dir, t.scale);
    o.base = vec4f(t.base, stand_tint(t.base));
    o.cs = t.cs;
    o.d0 = dirs[0];
    o.d1 = dirs[1];
    o.d2 = dirs[2];
    o.d3 = dirs[3];
    return o;
}

// The impostor under a pixel, each view weighted by its weight and coverage.
struct Gather {
    color: vec4f, // premultiplied colour, and coverage
    nrm: vec3f,   // normal (object space, not normalized)
    depth: f32,   // distance in front of the quad, towards the viewer (m, before scaling)
    ao: f32,
    mip: f32,     // the atlas mip level it is seen at
}

fn imp_gather(in: IOut, full: bool) -> Gather {
    let s = SHAPES[in.info.y];
    let layer = i32(s.imp.w);
    var uvs = array<vec2f, 4>(in.uv01.xy, in.uv01.zw, in.uv23.xy, in.uv23.zw);
    var dirs = array<vec3f, 4>(in.d0, in.d1, in.d2, in.d3);
    // the views all move across the quad at about the same rate
    let k = 1.0 / f32(IMP_FRAMES);
    let gx = dpdx(in.uv01.xy) * k;
    let gy = dpdy(in.uv01.xy) * k;
    let texels = max(length(gx), length(gy)) * f32(textureDimensions(t_imp_color).x);
    var g = Gather(vec4f(0.0), vec3f(0.0), 0.0, 0.0, max(log2(max(texels, 1e-6)), 0.0));
    for (var i = 0u; i < 4u; i++) {
        let uv = uvs[i];
        let w = select(0.0, in.weights[i], all(uv > vec2f(0.0)) && all(uv < vec2f(1.0)));
        if (w <= 0.0) {
            continue;
        }
        let cell = vec2u(in.views[i] % IMP_FRAMES, in.views[i] / IMP_FRAMES);
        let auv = (vec2f(cell) + uv) * k;
        let c = textureSampleGrad(t_imp_color, s_repeat, auv, layer, gx, gy);
        g.color += c * w;
        if (full && c.a > 1e-3) {
            let d = textureSampleGrad(t_imp_data, s_repeat, auv, layer, gx, gy) / c.a;
            let fd = dirs[i];
            let ax = imp_axes(fd);
            let nv = oct_decode(d.xy * 2.0 - 1.0);
            let wc = w * c.a;
            g.nrm += (ax.r * nv.x + ax.u * nv.y + fd * nv.z) * wc;
            g.depth += (d.z * 2.0 - 1.0) * s.imp.z * wc;
            g.ao += d.w * wc;
        }
    }
    let a = max(g.color.a, 1e-4);
    g.depth /= a;
    g.ao /= a;
    return g;
}

// Coverage as drawn. Far away a crown's leaves are smaller than a texel, so its views are
// partly covered all over: alpha-to-coverage turns that into the same share of the samples (a
// hard cut at one half would strip a sparse crown down to its bare branches). But the gaps
// between the leaves close up with distance, as more layers of them line up behind each
// other, so the coarser the mip level, the more the coverage is boosted; near the meshes'
// range it stays close to theirs.
fn imp_coverage(g: Gather) -> f32 {
    return saturate(g.color.a * mix(1.25, 2.0, saturate(g.mip / 3.0)) - 0.08);
}

// Without MSAA, and in the shadow map, a cut-out at a threshold that changes from pixel to
// pixel does the same.
fn imp_dither(frag: vec4f) -> f32 {
    return 0.05 + 0.9 * u01(hash_i2(vec2i(frag.xy)));
}

@fragment
fn fs_impostor(in: IOut) -> @location(0) vec4f {
    let g = imp_gather(in, true);
    let s = SHAPES[in.info.y];
    let scale = in.dir.w;
    let albedo0 = g.color.rgb / max(g.color.a, 1e-4);
    // the leaf back where it was baked, so the shadow, the crown's shade and the haze land right
    let p = in.quad + in.dir.xyz * g.depth;
    let wp = in.base.xyz + rot_y(p * scale, in.cs);
    let n = normalize(rot_y(g.nrm, in.cs) + vec3f(0.0, 1e-5, 0.0));
    // Leaves are green and bark isn't: only the leaves take the tree's tint and let light through.
    let leaf = smoothstep(0.38, 0.46, albedo0.g / max(albedo0.r + albedo0.g + albedo0.b, 1e-4));
    let albedo = albedo0 * mix(vec3f(1.0), tree_tint(in.info.z, in.base.w), leaf);
    // The tree casts this same impostor into the shadow map, as a plane through its centre facing
    // the sun: look the shadow up on the sun's side of it. Its own crown shades it instead.
    let L = F.sun_dir.xyz;
    let center = in.base.xyz + vec3f(0.0, s.imp.x * scale, 0.0);
    let lift = max(dot(center - wp, L), 0.0) + 1.0;
    let sun_vis = shadow_at(wp + L * lift, n) * crown_light(p, unrot_y(L, in.cs), s, scale);
    let vis = mix(sky_vis(wp.xz), 1.0, saturate(p.y / 12.0) * 0.85);
    var col = mix(shade_bark(albedo, n, wp, vis, sun_vis, g.ao), shade_leaves(albedo, n, wp, vis, sun_vis, g.ao, 1.0), leaf);
    if (F.debug.x > 0.5) {
        col = mix(col, lod_tint(in.info.w) * 0.3, 0.6);
    }
    col = apply_fog(col, wp);
    return vec4f(col, imp_coverage(g));
}

// The depth prepass with MSAA (alpha-to-coverage).
@fragment
fn fs_impostor_depth(in: IOut) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, imp_coverage(imp_gather(in, false)));
}

// The depth prepass without MSAA: an alpha test.
@fragment
fn fs_impostor_depth_test(in: IOut) -> @location(0) vec4f {
    if (imp_coverage(imp_gather(in, false)) < imp_dither(in.clip)) {
        discard;
    }
    return vec4f(0.0);
}

@fragment
fn fs_impostor_shadow(in: IOut) {
    if (imp_gather(in, false).color.a < imp_dither(in.clip)) {
        discard;
    }
}
