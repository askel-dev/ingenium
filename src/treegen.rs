//! Procedural tree meshes. Each species gets one branch skeleton, and four LODs are
//! built from that same skeleton so silhouettes stay consistent:
//!   LOD0  full branches + dense leaf cards
//!   LOD1  main limbs + fewer, larger cards + an opaque inner hull
//!   LOD2  trunk + a few huge cards + hull
//!   LOD3  just the canopy hull (union of spheres around each limb) + trunk stub

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::TAU;

pub const KIND_BARK: f32 = 0.0;
pub const KIND_LEAF: f32 = 1.0;
pub const KIND_HULL: f32 = 2.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TreeVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
    pub attr: [f32; 4], // kind, sway, ao, 0
}

#[derive(Default)]
pub struct Mesh {
    pub verts: Vec<TreeVertex>,
    pub idx: Vec<u32>,
}

impl Mesh {
    fn vert(&mut self, p: Vec3, n: Vec3, uv: Vec2, kind: f32, sway: f32, ao: f32) -> u32 {
        self.verts.push(TreeVertex { pos: p.into(), nrm: n.into(), uv: uv.into(), attr: [kind, sway, ao, 0.0] });
        self.verts.len() as u32 - 1
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.idx.extend_from_slice(&[a, b, c]);
    }
    pub fn tris(&self) -> u32 {
        (self.idx.len() / 3) as u32
    }
}

/// Trunk radius at the ground per species (broadleaf, birch, spruce), before the
/// per-tree scale. The player collides with trunks of this size.
pub const TRUNK_R: [f32; 3] = [0.42, 0.22, 0.34];

pub struct Species {
    pub name: &'static str,
    pub lods: Vec<Mesh>,
    pub center_y: f32,
    pub radius: f32,
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    fn count(&mut self, lo: u32, hi: u32) -> u32 {
        lo + ((hi - lo + 1) as f32 * self.f()) as u32
    }
    fn unit(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            let l = v.length_squared();
            if l > 1e-3 && l <= 1.0 {
                return v / l.sqrt();
            }
        }
    }
}

fn basis(d: Vec3) -> (Vec3, Vec3) {
    let a = if d.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = d.cross(a).normalize();
    let v = d.cross(u).normalize();
    (u, v)
}

struct Branch {
    pts: Vec<Vec3>,
    rads: Vec<f32>,
    level: u32,
    limb: usize,
}

impl Branch {
    fn length(&self) -> f32 {
        self.pts.windows(2).map(|w| w[0].distance(w[1])).sum()
    }
    /// position, direction, radius at fraction t of the length
    fn at(&self, t: f32) -> (Vec3, Vec3, f32) {
        let total = self.length();
        let mut s = t.clamp(0.0, 1.0) * total;
        for i in 0..self.pts.len() - 1 {
            let l = self.pts[i].distance(self.pts[i + 1]);
            if s <= l || i == self.pts.len() - 2 {
                let f = (s / l.max(1e-6)).clamp(0.0, 1.0);
                let p = self.pts[i].lerp(self.pts[i + 1], f);
                let d = (self.pts[i + 1] - self.pts[i]).normalize();
                let r = self.rads[i] + (self.rads[i + 1] - self.rads[i]) * f;
                return (p, d, r);
            }
            s -= l;
        }
        unreachable!()
    }
}

struct Params {
    height: f32,
    trunk_frac: f32,  // length of the trunk branch as a fraction of height
    trunk_r: f32,
    children: [(u32, u32); 3],
    child_start: [f32; 3],
    angle: [(f32, f32); 3],
    len_ratio: [f32; 3],
    excurrent: bool,
    tropism: [f32; 4],
    wiggle: [f32; 4],
    leaf_size: f32,
    leaves_per_twig: u32,
    crown_scale: Vec3, // ellipsoid radii multipliers for AO
}

struct Tree {
    branches: Vec<Branch>,
    leaves: Vec<(Vec3, Vec3, usize)>, // attach point, branch dir, limb
}

fn grow(rng: &mut Rng, p: &Params, tree: &mut Tree, start: Vec3, dir: Vec3, len: f32, radius: f32, level: u32, limb: usize) {
    let nseg = [10, 7, 5, 3][level as usize];
    let mut pts = vec![start];
    let mut rads = vec![radius];
    let mut d = dir;
    for i in 1..=nseg {
        d = (d + rng.unit() * p.wiggle[level as usize] + Vec3::Y * p.tropism[level as usize]).normalize();
        let last = *pts.last().unwrap();
        pts.push(last + d * (len / nseg as f32));
        let t = i as f32 / nseg as f32;
        rads.push(radius * (1.0 - 0.72 * t));
    }
    let br = Branch { pts, rads, level, limb };
    let me = tree.branches.len();
    tree.branches.push(br);
    let limb = if level == 1 { me } else { limb };
    if level >= 3 {
        return;
    }
    let (lo, hi) = p.children[level as usize];
    let n = rng.count(lo, hi);
    let start_t = p.child_start[level as usize];
    for c in 0..n {
        let t = (start_t + (1.0 - start_t) * (c as f32 + 0.3 + rng.f() * 0.5) / n as f32).min(0.97);
        let (pos, pd, pr) = tree.branches[me].at(t);
        let (u, v) = basis(pd);
        let phi = c as f32 * 2.39996 + rng.range(-0.4, 0.4);
        let side = u * phi.cos() + v * phi.sin();
        let (alo, ahi) = p.angle[level as usize];
        let ang = rng.range(alo, ahi);
        let mut cd = (pd * ang.cos() + side * ang.sin()).normalize();
        if level == 0 {
            // limbs off the trunk should spread outwards rather than downwards
            cd.y = cd.y.max(0.15);
            cd = cd.normalize();
        }
        let falloff = if level == 0 && p.excurrent { 1.0 - 0.75 * t } else { 1.0 - 0.35 * t };
        let clen = len * p.len_ratio[level as usize] * falloff * rng.range(0.8, 1.15);
        let crad = (pr * 0.62).max(0.012);
        grow(rng, p, tree, pos, cd, clen, crad, level + 1, limb);
    }
}

fn build_skeleton(p: &Params, seed: u64) -> Tree {
    let mut rng = Rng(seed);
    let mut tree = Tree { branches: Vec::new(), leaves: Vec::new() };
    let lean = Vec3::new(rng.range(-0.05, 0.05), 1.0, rng.range(-0.05, 0.05)).normalize();
    grow(&mut rng, p, &mut tree, Vec3::ZERO, lean, p.height * p.trunk_frac, p.trunk_r, 0, 0);
    // non-excurrent trees: the trunk forks into a crown of limbs at its top
    if !p.excurrent {
        let top = tree.branches[0].at(1.0);
        let n = rng.count(3, 4);
        for c in 0..n {
            let phi = c as f32 / n as f32 * TAU + rng.range(-0.3, 0.3);
            let ang = rng.range(0.6, 0.95);
            let d = Vec3::new(phi.cos() * ang.sin(), ang.cos(), phi.sin() * ang.sin()).normalize();
            let len = p.height * (1.0 - p.trunk_frac) * rng.range(0.85, 1.05);
            grow(&mut rng, p, &mut tree, top.0, d, len, top.2 * 0.75, 1, 0);
        }
    }
    // leaf attachment points along twigs (level 3) and the outer part of level 2
    let mut leaves = Vec::new();
    for b in tree.branches.iter() {
        let count = match b.level {
            3 => p.leaves_per_twig,
            2 => p.leaves_per_twig / 2,
            _ => 0,
        };
        for k in 0..count {
            let t = 0.25 + 0.75 * (k as f32 + rng.f()) / count as f32;
            let (pos, d, _) = b.at(t);
            leaves.push((pos, d, b.limb));
        }
    }
    tree.leaves = leaves;
    tree
}

struct Crown {
    center: Vec3,
    radii: Vec3,
    height: f32,
    spheres: Vec<(Vec3, f32)>,
}

fn crown_of(tree: &Tree, height: f32, scale: Vec3) -> Crown {
    let n = tree.leaves.len().max(1) as f32;
    let center = tree.leaves.iter().fold(Vec3::ZERO, |a, l| a + l.0) / n;
    let mut ext = Vec3::ZERO;
    for l in &tree.leaves {
        ext = ext.max((l.0 - center).abs());
    }
    let radii = (ext * scale).max(Vec3::splat(0.5));
    // one sphere per limb around its leaves
    let mut groups: HashMap<usize, Vec<Vec3>> = HashMap::new();
    for l in &tree.leaves {
        groups.entry(l.2).or_default().push(l.0);
    }
    let mut spheres = Vec::new();
    let mut keys: Vec<_> = groups.keys().copied().collect();
    keys.sort();
    for k in keys {
        let g = &groups[&k];
        let c = g.iter().fold(Vec3::ZERO, |a, p| a + *p) / g.len() as f32;
        let mut d: Vec<f32> = g.iter().map(|p| p.distance(c)).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r = d[(d.len() as f32 * 0.85) as usize % d.len()] + 0.6;
        spheres.push((c, r));
    }
    Crown { center, radii, height, spheres }
}

fn sway_of(p: Vec3, crown: &Crown) -> f32 {
    let horiz = Vec2::new(p.x, p.z).length() / crown.radii.x.max(crown.radii.z);
    let vert = (p.y / crown.height).clamp(0.0, 1.2);
    (horiz * 0.65 + vert * vert * 0.45).clamp(0.0, 1.0)
}

fn leaf_ao(p: Vec3, crown: &Crown) -> f32 {
    let q = (p - crown.center) / crown.radii;
    let depth = q.length();
    let height = (q.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let a = 0.3 + 0.7 * ((depth - 0.2) / 0.8).clamp(0.0, 1.0);
    (a * (0.7 + 0.3 * height)).clamp(0.15, 1.0)
}

fn emit_tube(mesh: &mut Mesh, b: &Branch, sides: usize, step: usize, crown: &Crown) {
    let mut idxs: Vec<usize> = (0..b.pts.len()).step_by(step.max(1)).collect();
    if *idxs.last().unwrap() != b.pts.len() - 1 {
        idxs.push(b.pts.len() - 1);
    }
    let mut prev_t = (b.pts[1] - b.pts[0]).normalize();
    let (mut nrm, _) = basis(prev_t);
    let mut along = 0.0;
    let mut rings: Vec<u32> = Vec::new();
    for (k, &i) in idxs.iter().enumerate() {
        let t = if i + 1 < b.pts.len() { (b.pts[i + 1] - b.pts[i]).normalize() } else { (b.pts[i] - b.pts[i - 1]).normalize() };
        // parallel transport the frame
        let axis = prev_t.cross(t);
        if axis.length() > 1e-5 {
            let ang = prev_t.dot(t).clamp(-1.0, 1.0).acos();
            nrm = glam::Quat::from_axis_angle(axis.normalize(), ang) * nrm;
        }
        nrm = (nrm - t * nrm.dot(t)).normalize();
        let bin = t.cross(nrm);
        prev_t = t;
        if k > 0 {
            along += b.pts[i].distance(b.pts[idxs[k - 1]]);
        }
        let r = b.rads[i];
        let start = mesh.verts.len() as u32;
        for s in 0..=sides {
            let a = s as f32 / sides as f32 * TAU;
            let off = nrm * a.cos() + bin * a.sin();
            let p = b.pts[i] + off * r;
            let ao = (0.45 + 0.55 * (p.y / crown.height).clamp(0.0, 1.0)) * if b.level == 0 { 1.0 } else { 0.8 };
            mesh.vert(p, off, Vec2::new(s as f32 / sides as f32, along), KIND_BARK, sway_of(p, crown), ao);
        }
        rings.push(start);
    }
    for w in rings.windows(2) {
        let (a, c) = (w[0], w[1]);
        for s in 0..sides as u32 {
            mesh.tri(a + s, c + s, a + s + 1);
            mesh.tri(a + s + 1, c + s, c + s + 1);
        }
    }
}

fn emit_card(mesh: &mut Mesh, attach: Vec3, axis: Vec3, side: Vec3, len: f32, width: f32, crown: &Crown) {
    let b0 = attach - side * width * 0.5;
    let b1 = attach + side * width * 0.5;
    let corners = [b0, b1, b1 + axis * len, b0 + axis * len];
    let uvs = [Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)];
    let card_n = axis.cross(side).normalize();
    let start = mesh.verts.len() as u32;
    for k in 0..4 {
        let p = corners[k];
        let sph = ((p - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let cn = if card_n.dot(sph) < 0.0 { -card_n } else { card_n };
        let n = (sph * 0.75 + cn * 0.25 + Vec3::Y * 0.15).normalize();
        mesh.vert(p, n, uvs[k], KIND_LEAF, sway_of(p, crown), leaf_ao(p, crown));
    }
    mesh.tri(start, start + 1, start + 2);
    mesh.tri(start, start + 2, start + 3);
}

/// Icosphere with `sub` subdivisions; returns (positions on unit sphere, triangles).
fn icosphere(sub: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..sub {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut mid = |a: u32, b: u32, v: &mut Vec<Vec3>| -> u32 {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                v.push(((v[a as usize] + v[b as usize]) * 0.5).normalize());
                v.len() as u32 - 1
            })
        };
        let mut nf = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let ab = mid(a, b, &mut v);
            let bc = mid(b, c, &mut v);
            let ca = mid(c, a, &mut v);
            nf.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    (v, f)
}

/// Lumpy canopy hull: each icosphere direction is pushed out to the union-of-spheres envelope.
fn emit_hull(mesh: &mut Mesh, crown: &Crown, sub: u32, shrink: f32, rng: &mut Rng) {
    let (dirs, faces) = icosphere(sub);
    let c = crown.center;
    let mean_r = crown.spheres.iter().map(|s| s.1).sum::<f32>() / crown.spheres.len().max(1) as f32;
    let mut pos: Vec<Vec3> = Vec::with_capacity(dirs.len());
    for d in &dirs {
        let mut tmax: f32 = 0.0;
        for &(sc, sr) in &crown.spheres {
            let oc = sc - c;
            let b = d.dot(oc);
            let disc = b * b - (oc.length_squared() - sr * sr);
            if disc >= 0.0 {
                tmax = tmax.max(b + disc.sqrt());
            }
        }
        if tmax <= 0.0 {
            tmax = mean_r * 0.4;
        }
        let jitter = 1.0 + rng.range(-0.06, 0.06);
        pos.push(c + *d * tmax * shrink * jitter);
    }
    // keep the hull above the trunk base
    for p in pos.iter_mut() {
        p.y = p.y.max(crown.height * 0.12);
    }
    let mut nrm = vec![Vec3::ZERO; pos.len()];
    for f in &faces {
        let n = (pos[f[1] as usize] - pos[f[0] as usize]).cross(pos[f[2] as usize] - pos[f[0] as usize]);
        for &i in f {
            nrm[i as usize] += n;
        }
    }
    let start = mesh.verts.len() as u32;
    for i in 0..pos.len() {
        let n = nrm[i].normalize_or(dirs[i]);
        let ao = 0.62 + 0.38 * (dirs[i].y * 0.5 + 0.5);
        mesh.vert(pos[i], n, Vec2::ZERO, KIND_HULL, sway_of(pos[i], crown) * 0.6, ao);
    }
    for f in &faces {
        mesh.tri(start + f[0], start + f[1], start + f[2]);
    }
}

fn place_cards(mesh: &mut Mesh, tree: &Tree, crown: &Crown, rng: &mut Rng, levels: &[u32], per_branch: u32, size: f32) {
    for b in tree.branches.iter().filter(|b| levels.contains(&b.level)) {
        for k in 0..per_branch {
            let t = 0.2 + 0.8 * (k as f32 + rng.f()) / per_branch as f32;
            let (pos, d, _) = b.at(t);
            let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
            let axis = (d * 0.45 + out * 0.55 + rng.unit() * 0.55).normalize();
            let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
            let s = size * rng.range(0.85, 1.15);
            emit_card(mesh, pos - axis * s * 0.15, axis, side, s, s, crown);
        }
    }
}

fn broadleaf(name: &'static str, p: Params, seed: u64) -> Species {
    let tree = build_skeleton(&p, seed);
    let crown = crown_of(&tree, p.height, p.crown_scale);
    let mut rng = Rng(seed ^ 0xABCDEF);
    let mut lods: Vec<Mesh> = (0..4).map(|_| Mesh::default()).collect();

    // LOD0
    for b in &tree.branches {
        let sides = [10, 6, 4, 3][b.level as usize];
        emit_tube(&mut lods[0], b, sides, 1, &crown);
    }
    for &(pos, d, _) in &tree.leaves {
        let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let axis = (d * 0.5 + out * 0.5 + rng.unit() * 0.6).normalize();
        let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
        let s = p.leaf_size * rng.range(0.8, 1.2);
        emit_card(&mut lods[0], pos - axis * s * 0.1, axis, side, s, s, &crown);
    }

    // LOD1
    for b in tree.branches.iter().filter(|b| b.level <= 1) {
        emit_tube(&mut lods[1], b, if b.level == 0 { 7 } else { 4 }, 2, &crown);
    }
    place_cards(&mut lods[1], &tree, &crown, &mut rng, &[3], 3, p.leaf_size * 1.4);
    place_cards(&mut lods[1], &tree, &crown, &mut rng, &[2], 2, p.leaf_size * 1.7);

    // LOD2
    emit_tube(&mut lods[2], &tree.branches[0], 4, 4, &crown);
    place_cards(&mut lods[2], &tree, &crown, &mut rng, &[2], 3, p.leaf_size * 2.6);
    emit_hull(&mut lods[2], &crown, 1, 0.74, &mut rng);

    // LOD3
    let trunk = Branch {
        pts: vec![Vec3::ZERO, tree.branches[0].at(0.6).0],
        rads: vec![p.trunk_r, p.trunk_r * 0.7],
        level: 0,
        limb: 0,
    };
    let _ = trunk;
    emit_hull(&mut lods[3], &crown, 1, 1.0, &mut rng);

    finish(name, lods)
}

fn spruce(seed: u64) -> Species {
    let mut rng = Rng(seed);
    let h = 24.0;
    let base_y = 2.2;
    let max_r = 4.4;
    let trunk = Branch {
        pts: (0..=12).map(|i| Vec3::new(rng.range(-0.04, 0.04), h * i as f32 / 12.0, rng.range(-0.04, 0.04))).collect(),
        rads: (0..=12).map(|i| TRUNK_R[2] * (1.0 - 0.92 * i as f32 / 12.0)).collect(),
        level: 0,
        limb: 0,
    };
    let crown = Crown {
        center: Vec3::new(0.0, h * 0.45, 0.0),
        radii: Vec3::new(max_r, h * 0.55, max_r),
        height: h,
        spheres: Vec::new(),
    };
    // whorls of branches, longest at the bottom
    struct Limb {
        b: Branch,
        len: f32,
    }
    let mut limbs: Vec<(f32, Limb)> = Vec::new();
    let mut y = base_y;
    let mut whorl = 0;
    while y < h - 0.5 {
        let frac = (h - y) / (h - base_y);
        let n = rng.count(5, 7);
        let off = rng.range(0.0, TAU);
        for k in 0..n {
            let phi = off + k as f32 / n as f32 * TAU + rng.range(-0.25, 0.25);
            let el = rng.range(-0.18, 0.06) - 0.28 * frac;
            let len = 0.35 + max_r * frac.powf(0.9) * rng.range(0.85, 1.1);
            let d0 = Vec3::new(phi.cos() * el.cos(), el.sin(), phi.sin() * el.cos());
            let start = Vec3::new(0.0, y, 0.0);
            let mut pts = vec![start];
            let mut d = d0;
            for _ in 0..3 {
                d = (d + Vec3::Y * 0.12).normalize();
                pts.push(*pts.last().unwrap() + d * len / 3.0);
            }
            let r0 = 0.03 + 0.05 * frac;
            limbs.push((y, Limb { b: Branch { pts, rads: vec![r0, r0 * 0.7, r0 * 0.45, r0 * 0.2], level: 1, limb: whorl }, len }));
        }
        y += rng.range(0.38, 0.5);
        whorl += 1;
    }

    let mut lods: Vec<Mesh> = (0..4).map(|_| Mesh::default()).collect();
    // LOD0: every branch, three needle cards each
    emit_tube(&mut lods[0], &trunk, 8, 1, &crown);
    for (_, l) in &limbs {
        emit_tube(&mut lods[0], &l.b, 3, 1, &crown);
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for (k, roll) in [(0.0f32, 0.0f32), (0.4, 0.25), (0.15, 1.2)] {
            let roll = roll + rng.range(-0.3, 0.3);
            let side = glam::Quat::from_axis_angle(d, roll) * flat;
            let card_len = l.len * 0.68;
            let w = (l.len * 0.55).clamp(0.45, 1.3);
            emit_card(&mut lods[0], l.b.at(k).0, d, side, card_len, w, &crown);
        }
    }
    // LOD1: every other whorl, two bigger cards per branch + inner cone
    emit_tube(&mut lods[1], &trunk, 5, 2, &crown);
    for (i, (_, l)) in limbs.iter().enumerate() {
        if (l.b.limb % 2) != 0 || i % 6 == 5 {
            continue;
        }
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for roll in [0.0f32, 1.3] {
            let side = glam::Quat::from_axis_angle(d, roll + rng.range(-0.3, 0.3)) * flat;
            emit_card(&mut lods[1], l.b.pts[0], d, side, l.len * 1.08, (l.len * 0.9).clamp(0.8, 2.4), &crown);
        }
    }
    emit_cone(&mut lods[1], h, base_y, max_r * 0.7, 10, &crown);
    // LOD2: sparse huge cards + cone
    emit_tube(&mut lods[2], &trunk, 4, 6, &crown);
    for (i, (_, l)) in limbs.iter().enumerate() {
        if l.b.limb % 4 != 0 || i % 2 == 1 {
            continue;
        }
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        let side = glam::Quat::from_axis_angle(d, rng.range(-0.4, 0.4)) * flat;
        emit_card(&mut lods[2], l.b.pts[0], d, side, l.len * 1.15, (l.len * 1.3).clamp(1.2, 3.6), &crown);
    }
    emit_cone(&mut lods[2], h, base_y, max_r * 0.82, 8, &crown);
    // LOD3: cone only
    let stub = Branch { pts: vec![Vec3::ZERO, Vec3::new(0.0, base_y + 1.0, 0.0)], rads: vec![0.34, 0.3], level: 0, limb: 0 };
    let _ = stub;
    emit_cone(&mut lods[3], h, base_y, max_r, 6, &crown);
    finish("spruce", lods)
}

fn emit_cone(mesh: &mut Mesh, h: f32, base_y: f32, r: f32, sides: usize, crown: &Crown) {
    let rings = [(base_y - 0.6, r * 0.55), (base_y + 0.6, r), (h * 0.45, r * 0.68), (h * 0.75, r * 0.32)];
    let mut starts = Vec::new();
    for &(y, rr) in &rings {
        let start = mesh.verts.len() as u32;
        for s in 0..sides {
            let a = s as f32 / sides as f32 * TAU;
            let p = Vec3::new(a.cos() * rr, y, a.sin() * rr);
            let n = Vec3::new(a.cos(), 0.45, a.sin()).normalize();
            let ao = 0.4 + 0.6 * (y / h);
            mesh.vert(p, n, Vec2::ZERO, KIND_HULL, sway_of(p, crown) * 0.5, ao);
        }
        starts.push(start);
    }
    let tip = mesh.vert(Vec3::new(0.0, h, 0.0), Vec3::Y, Vec2::ZERO, KIND_HULL, 0.6, 1.0);
    let s = sides as u32;
    for w in starts.windows(2) {
        let (a, b) = (w[0], w[1]);
        for k in 0..s {
            let k1 = (k + 1) % s;
            mesh.tri(a + k, b + k, a + k1);
            mesh.tri(a + k1, b + k, b + k1);
        }
    }
    let last = *starts.last().unwrap();
    for k in 0..s {
        mesh.tri(last + k, tip, last + (k + 1) % s);
    }
}

fn finish(name: &'static str, lods: Vec<Mesh>) -> Species {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for v in &lods[0].verts {
        let p = Vec3::from(v.pos);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let center_y = (lo.y + hi.y) * 0.5;
    let c = Vec3::new(0.0, center_y, 0.0);
    let radius = lods.iter().flat_map(|m| m.verts.iter()).map(|v| Vec3::from(v.pos).distance(c)).fold(0.0, f32::max) + 1.0;
    Species { name, lods, center_y, radius }
}

pub fn build_all() -> Vec<Species> {
    let oak = Params {
        height: 18.0,
        trunk_frac: 0.38,
        trunk_r: TRUNK_R[0],
        children: [(2, 3), (5, 7), (4, 6)],
        child_start: [0.55, 0.2, 0.2],
        angle: [(0.5, 0.9), (0.6, 1.1), (0.5, 1.0)],
        len_ratio: [0.9, 0.55, 0.5],
        excurrent: false,
        tropism: [0.02, 0.06, 0.03, -0.02],
        wiggle: [0.05, 0.16, 0.25, 0.3],
        leaf_size: 1.45,
        leaves_per_twig: 8,
        crown_scale: Vec3::new(1.1, 1.0, 1.1),
    };
    let birch = Params {
        height: 17.0,
        trunk_frac: 1.0,
        trunk_r: TRUNK_R[1],
        children: [(12, 16), (4, 6), (3, 5)],
        child_start: [0.32, 0.2, 0.25],
        angle: [(0.55, 0.85), (0.5, 0.9), (0.6, 1.1)],
        len_ratio: [0.4, 0.6, 0.55],
        excurrent: true,
        tropism: [0.0, 0.03, -0.12, -0.22],
        wiggle: [0.04, 0.14, 0.22, 0.3],
        leaf_size: 1.05,
        leaves_per_twig: 8,
        crown_scale: Vec3::new(1.05, 1.0, 1.05),
    };
    let species = vec![broadleaf("broadleaf", oak, 0x1234_5678_9ABC), broadleaf("birch", birch, 0xBEEF_F00D_1357), spruce(0x5EED_5EED_2468)];
    for s in &species {
        let tris: Vec<String> = s.lods.iter().map(|m| m.tris().to_string()).collect();
        println!("tree '{}': LOD tris [{}], bound r={:.1} m", s.name, tris.join(", "), s.radius);
    }
    species
}
