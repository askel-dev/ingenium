// Shared by the tree shader (tree.wgsl) and the impostor bake (impostor_bake.wgsl): decoding a
// tree record, the bark, the per-tree colour, the lighting of leaves and bark, and the impostor
// frames' directions.

// What each bark scan covers (m): broadleaf, birch, spruce (materials.rs layers 0-2).
const BARK_TILE = array<vec2f, 3>(vec2f(2.15, 2.15), vec2f(1.1, 2.2), vec2f(2.0, 2.0));

// moss, on fallen wood and rocks
const MOSS = vec3f(0.045, 0.07, 0.018);

// Impostors (impostor.rs): IMP_GRID x IMP_GRID frames per tree, the views from directions
// spread over the upper hemisphere (hemi-octahedral). Frame (i, j) is at index j * IMP_GRID + i.
const IMP_GRID = 8u;

// A tree record (scatter.wgsl): xyz = bitcast position, w = packed
//   bits 0-1 species, 2-9 rotation, 10-17 scale, 18-19 variant, 20-31 seed
struct Inst {
    base: vec3f,
    species: u32,
    variant: u32,
    seed: u32,
    c: f32,           // cos and sin of the rotation about the vertical
    s: f32,
    scale: f32,
    stretch: vec2f,   // x across, y up: no two trees of a variant have quite the same build
    mirror: f32,      // -1: the tree is mirrored
}

fn inst(t: vec4u) -> Inst {
    var i: Inst;
    i.base = bitcast<vec3f>(t.xyz);
    let packed = t.w;
    i.species = packed & 3u;
    let rot = f32((packed >> 2u) & 255u) / 255.0 * TAU;
    i.c = cos(rot);
    i.s = sin(rot);
    i.scale = 0.75 + f32((packed >> 10u) & 255u) / 255.0 * 0.55;
    i.variant = (packed >> 18u) & 3u;
    i.seed = packed >> 20u;
    let h = pcg(i.seed * 0x9E37u + 17u);
    i.stretch = vec2f(mix(0.9, 1.12, u01(h)), mix(0.9, 1.1, u01(pcg(h))));
    i.mirror = select(1.0, -1.0, (pcg(h ^ 0x51u) & 1u) == 1u);
    return i;
}

fn inst_size(i: Inst) -> vec3f {
    return vec3f(i.stretch.x, i.stretch.y, i.stretch.x) * i.scale;
}

// tree space -> world, relative to the tree's base
fn inst_vec(i: Inst, p: vec3f) -> vec3f {
    let q = p * inst_size(i) * vec3f(i.mirror, 1.0, 1.0);
    return vec3f(i.c * q.x - i.s * q.z, q.y, i.s * q.x + i.c * q.z);
}

fn inst_normal(i: Inst, n: vec3f) -> vec3f {
    let q = normalize(n * vec3f(i.mirror / i.stretch.x, 1.0 / i.stretch.y, 1.0 / i.stretch.x));
    return vec3f(i.c * q.x - i.s * q.z, q.y, i.s * q.x + i.c * q.z);
}

// world -> tree space (a vector, relative to whatever point it was measured from)
fn inst_local(i: Inst, w: vec3f) -> vec3f {
    let q = vec3f(i.c * w.x + i.s * w.z, w.y, -i.s * w.x + i.c * w.z);
    return q / inst_size(i) * vec3f(i.mirror, 1.0, 1.0);
}

// The bark wraps a whole number of times around the branch (no seam), as close as it can get
// to the scan's real size at the branch's reference girth, and keeps its aspect along the
// branch. A trunk gets life-size bark; a twig, whose girth is under one tile, a shrunken copy.
// Each tree starts the texture somewhere else.
fn bark_uv(species: u32, uv: vec2f, girth: f32, seed: u32) -> vec3f {
    let tile = BARK_TILE[min(species, 2u)];
    let k = max(round(girth / tile.x), 1.0);
    let stretch = girth / (k * tile.x);
    let off = vec2f(u01(pcg(seed * 3u + 1u)), u01(pcg(seed * 5u + 2u)) * 7.0);
    return vec3f(vec2f(uv.x * k, uv.y / (tile.y * max(stretch, 0.02))) + off, k);
}

// Value noise that repeats every `period` cells along x, for marks that wrap around a trunk.
fn wrap_noise(p: vec2f, period: i32) -> f32 {
    let fl = floor(p);
    let f = p - fl;
    let u = f * f * (3.0 - 2.0 * f);
    let i = vec2i(fl);
    let x0 = ((i.x % period) + period) % period;
    let x1 = (x0 + 1) % period;
    let a = u01(hash_i2(vec2i(x0, i.y)));
    let b = u01(hash_i2(vec2i(x1, i.y)));
    let c = u01(hash_i2(vec2i(x0, i.y + 1)));
    let d = u01(hash_i2(vec2i(x1, i.y + 1)));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Birch: the dark horizontal lenticels and the black blotches where branches were shed, over
// the pale scan. b in tiles (1.1 x 2.2 m), wrapping every `wraps` tiles around the trunk.
// Returns how dark (0..1) and fades to its average once a pixel spans the marks.
fn birch_marks(b: vec2f, wraps: f32, fw: f32) -> f32 {
    let n = i32(wraps + 0.5);
    // lenticels: dashes ~8 cm long, a few mm tall
    let lq = vec2f(b.x * 14.0, b.y * 140.0);
    let len = smoothstep(0.62, 0.8, wrap_noise(lq, 14 * n)) * smoothstep(0.3, 0.7, wrap_noise(lq * vec2f(1.0, 0.25) + 17.0, 14 * n));
    // blotches, wider than tall
    let bq = vec2f(b.x * 3.0, b.y * 9.0);
    // ragged edges: a finer octave shifts the threshold
    let bn = wrap_noise(bq, 3 * n) * 0.7 + wrap_noise(bq * 2.0 + 5.0, 6 * n) * 0.2 + wrap_noise(bq * 6.0 + 9.0, 18 * n) * 0.1;
    let blot = smoothstep(0.66, 0.7, bn);
    // a pixel spanning several lenticels sees their average cover instead
    let detail = 1.0 - smoothstep(0.004, 0.015, fw);
    return max(mix(0.05, len * 0.8, detail), blot * 0.9);
}

// Birch bark: the marks (above) over the pale scan, and only the trunk and big limbs white: the
// thin branches are dark red-brown. girth in m.
fn birch_bark(albedo: vec3f, b: vec3f, fw: f32, girth: f32) -> vec4f {
    let dark = birch_marks(b.xy, b.z, fw);
    var a = mix(albedo, vec3f(0.035, 0.03, 0.028), dark);
    let twig = 1.0 - smoothstep(0.22, 0.55, girth);
    a = mix(a, vec3f(0.055, 0.04, 0.032), twig);
    return vec4f(a, max(dark, twig));
}

// The hull's colour: what the leaf cards average to from a distance (the scanned leaves are a
// touch warmer than the painted ones were, and the hull and the canopy shell follow them).
fn canopy_color(species: u32) -> vec3f {
    switch species {
        case 1u: { return vec3f(0.081, 0.12, 0.028); }
        case 2u: { return vec3f(0.03, 0.055, 0.026); }
        default: { return vec3f(0.059, 0.10, 0.02); }
    }
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
    tint = mix(tint, tint * vec3f(1.55, 1.2, 0.5), smoothstep(0.93, 1.0, r3) * 0.75);
    return tint;
}

// neighbouring trees look alike (same soil, same age), so part of the colour varies by stand
fn stand_of(base: vec3f) -> f32 {
    return gnoise(base.xz * 0.012 + vec2f(3.3, -8.1)) * 0.6 + gnoise(base.xz * 0.05) * 0.4;
}

// A normal map's normal (xy, z up) in a surface's frame built from screen derivatives
// (Schüler 2013): no tangents needed. n = the surface normal, dwx/dwy = the position's and
// duv_x/duv_y the texture coordinates' derivatives.
fn perturb(n: vec3f, nm: vec2f, dwx: vec3f, dwy: vec3f, dux: vec2f, duy: vec2f) -> vec3f {
    let p1 = cross(dwy, n);
    let p2 = cross(n, dwx);
    let t = p1 * dux.x + p2 * duy.x;
    let b = p1 * dux.y + p2 * duy.y;
    let inv = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    let nz = sqrt(saturate(1.0 - dot(nm, nm)));
    return normalize(t * inv * nm.x + b * inv * nm.y + n * nz);
}

// Light on leaves: ns their own normal, n the crown's rounded one (which carries the shading
// of the crown as a whole), trans how much light they let through, sh the sun's shadow, vis
// the share of the sky they see.
fn shade_leaf(albedo: vec3f, ns: vec3f, n: vec3f, ao: f32, trans: f32, rough: f32, sh: f32, vis: f32, wp: vec3f, V: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    let sun = sun_light(wp);
    let wrap = max((dot(ns, L) + 0.45) / 1.45, 0.0);
    let fwd = pow(max(dot(-V, L), 0.0), 5.0) * 0.3 * mix(0.4, 1.25, trans);
    // the leaves' own normals for the sky too, but half way to the crown's
    let na = normalize(n + ns);
    var col = albedo * (sun * (wrap * 0.95 + fwd) * sh * mix(0.45, 1.0, ao) + ambient_vis(na, wp, vis) * ao);
    // Leaves are waxy: a highlight from the sun off each leaf (GGX, from the scan's roughness)
    // and a sky sheen at grazing angles; uncoloured, as on any dielectric.
    let ndv = saturate(dot(ns, V));
    let fres = 0.04 + 0.16 * pow(1.0 - ndv, 4.0);
    let sheen = SKY.amb_up.rgb * F.sun_color.w * vis * fres * 0.35;
    col += sun * sun_specular(ns, V, L, max(rough, 0.3)) * sh * mix(0.45, 1.0, ao) * 0.8 + sheen * ao;
    // Leaves are thin: light falling on one side comes out of the other, deeper and yellower
    // than the leaf's own colour, less through veins and twigs. The crown's normals point
    // outwards, so the leaves under a crown glow with the sun and sky on their upper side, as a
    // canopy does seen from below.
    let tcol = albedo * vec3f(1.1, 1.15, 0.6) * mix(0.3, 1.4, trans);
    let through = sun * max(dot(-ns, L), 0.0) * sh * 0.7 + ambient_vis(-n, wp, vis) * ao * 0.65;
    col += tcol * through;
    return col;
}

// Light on bark: ns the normal-mapped normal, tex_ao the scan's occlusion.
fn shade_bark(albedo: vec3f, ns: vec3f, ao: f32, tex_ao: f32, rough: f32, sh: f32, vis: f32, wp: vec3f, V: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    let sun = sun_light(wp);
    let nl = max(dot(ns, L), 0.0);
    // the crevices of the bark shade themselves, and hold less of the sky
    let direct = sun * sh * micro_shadow(tex_ao, nl);
    var col = albedo * (direct * nl + ambient_vis(ns, wp, vis) * ao * tex_ao);
    col += direct * sun_specular(ns, V, L, rough);
    return col;
}

// Direction (upper hemisphere) <-> [-1, 1]^2: the hemi-octahedral map the impostor frames are
// laid out on.
fn hemi_oct(d: vec3f) -> vec2f {
    let p = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
    return vec2f(p.x + p.y, p.x - p.y);
}

fn hemi_oct_dir(e: vec2f) -> vec3f {
    let t = vec2f(e.x + e.y, e.x - e.y) * 0.5;
    return normalize(vec3f(t.x, 1.0 - abs(t.x) - abs(t.y), t.y));
}

// The direction frame (i, j) was baked from (towards the camera that baked it)
fn frame_dir(g: vec2f) -> vec3f {
    return hemi_oct_dir(g / f32(IMP_GRID - 1u) * 2.0 - 1.0);
}

// A frame's right and up, seen from direction d (never straight up: no frame is at the zenith)
fn frame_right(d: vec3f) -> vec3f {
    return normalize(cross(vec3f(0.0, 1.0, 0.0), d));
}

// Full-sphere octahedral normals (y up), as impostor.rs packs them.
fn oct_decode(e: vec2f) -> vec3f {
    let f = e * 2.0 - 1.0;
    var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
    let t = max(-n.y, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.z += select(t, -t, n.z >= 0.0);
    return normalize(n);
}
