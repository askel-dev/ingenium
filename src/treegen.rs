//! Procedural tree meshes. Each species grows in four variants: two forest-grown skeletons,
//! an open-grown one (low, wide branching) for forest edges and lone meadow trees, and a young
//! tree for the edge's mantle. Each variant has two mesh LODs built from the same skeleton:
//!   LOD0  full branches + dense leaf cards, and a root flare where the trunk meets the ground
//!   LOD1  main limbs + fewer, larger cards
//! Further out a tree is an impostor (impostor.rs): pictures of its LOD0 from many directions.

use crate::foliage::{self, Cell};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::f32::consts::TAU;

pub const KIND_BARK: f32 = 0.0;
pub const KIND_LEAF: f32 = 1.0;
pub const KIND_HULL: f32 = 2.0;
/// bark that lies on the ground and follows it (tree.wgsl, `vs`): fallen logs and branches
pub const KIND_DRAPED: f32 = 3.0;
pub const KIND_ROCK: f32 = 4.0;
/// the cut end of a log or a stump
pub const KIND_WOOD: f32 = 5.0;

pub const SPECIES: usize = 3;
/// forest-grown A and B, open-grown, young
pub const VARIANTS: usize = 4;
pub const MESH_LODS: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TreeVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
    /// kind, sway, ao, and for bark the branch's girth (m) where the bark is sized, for leaves
    /// the foliage atlas layer
    pub attr: [f32; 4],
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

/// Trunk radius at the ground per species (broadleaf, birch, spruce) and variant, before the
/// per-tree scale (and without the root flare). The player collides with trunks of this size;
/// the young broadleaf is a clump of stems.
pub const TRUNK_R: [[f32; VARIANTS]; SPECIES] = [[0.42, 0.40, 0.50, 0.22], [0.22, 0.24, 0.26, 0.09], [0.34, 0.37, 0.40, 0.12]];

pub struct TreeKind {
    pub name: String,
    pub lods: Vec<Mesh>,
    pub center_y: f32,
    /// bounding sphere of every LOD around (0, center_y, 0), with a margin for the wind
    pub radius: f32,
    /// half the side of an impostor frame: how far LOD0 reaches from (0, center_y, 0)
    pub half: f32,
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

#[derive(Clone, Copy)]
struct Params {
    height: f32,
    trunk_frac: f32, // length of the trunk branch as a fraction of height
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
    /// limbs a non-excurrent trunk forks into at its top, and their angle from the vertical
    fork: (u32, u32),
    fork_angle: (f32, f32),
    /// more than one: a clump of stems from the ground instead of a trunk (hazel-like)
    stems: u32,
    lean: f32,
    /// how much wider the trunk is at the ground, and the buttresses it spreads into
    flare: f32,
    lobes: u32,
}

const BASE: Params = Params {
    height: 20.0,
    trunk_frac: 0.5,
    trunk_r: 0.4,
    children: [(2, 3), (5, 7), (4, 6)],
    child_start: [0.7, 0.25, 0.2],
    angle: [(0.5, 0.9), (0.5, 0.95), (0.5, 1.0)],
    len_ratio: [0.6, 0.45, 0.5],
    excurrent: false,
    tropism: [0.02, 0.06, 0.03, -0.02],
    wiggle: [0.05, 0.16, 0.25, 0.3],
    leaf_size: 1.3,
    leaves_per_twig: 6,
    crown_scale: Vec3::new(1.1, 1.0, 1.1),
    fork: (3, 4),
    fork_angle: (0.36, 0.62),
    stems: 1,
    lean: 0.05,
    flare: 0.0,
    lobes: 0,
};

struct Tree {
    branches: Vec<Branch>,
    leaves: Vec<(Vec3, Vec3)>, // attach point, branch dir
}

#[allow(clippy::too_many_arguments)]
fn grow(rng: &mut Rng, p: &Params, tree: &mut Tree, start: Vec3, dir: Vec3, len: f32, radius: f32, level: u32) {
    let nseg = [10, 7, 5, 3][level as usize];
    let mut pts = vec![start];
    let mut rads = vec![radius];
    let mut d = dir;
    // A trunk that forks into limbs keeps most of its girth up to the fork; everything else
    // tapers towards a thin tip.
    let taper = if level == 0 && !p.excurrent { 0.4 } else { 0.72 };
    for i in 1..=nseg {
        d = (d + rng.unit() * p.wiggle[level as usize] + Vec3::Y * p.tropism[level as usize]).normalize();
        let last = *pts.last().unwrap();
        pts.push(last + d * (len / nseg as f32));
        let t = i as f32 / nseg as f32;
        rads.push(radius * (1.0 - taper * t));
    }
    let me = tree.branches.len();
    tree.branches.push(Branch { pts, rads, level });
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
        grow(rng, p, tree, pos, cd, clen, crad, level + 1);
    }
}

fn build_skeleton(p: &Params, seed: u64) -> Tree {
    let mut rng = Rng(seed);
    let mut tree = Tree { branches: Vec::new(), leaves: Vec::new() };
    if p.stems > 1 {
        // a clump of stems leaning out from one stool, each branching on its own
        for k in 0..p.stems {
            let phi = k as f32 / p.stems as f32 * TAU + rng.range(-0.4, 0.4);
            let out = Vec3::new(phi.cos(), 0.0, phi.sin());
            let d = (Vec3::Y + out * rng.range(0.2, 0.45)).normalize();
            let len = p.height * rng.range(0.7, 1.0) / d.y;
            let r = p.trunk_r * rng.range(0.75, 1.1);
            grow(&mut rng, p, &mut tree, out * 0.1, d, len, r, 0);
        }
    } else {
        let lean = Vec3::new(rng.range(-p.lean, p.lean), 1.0, rng.range(-p.lean, p.lean)).normalize();
        grow(&mut rng, p, &mut tree, Vec3::ZERO, lean, p.height * p.trunk_frac, p.trunk_r, 0);
        // non-excurrent trees: the trunk forks into a crown of limbs at its top
        if !p.excurrent {
            let top = tree.branches[0].at(1.0);
            let n = rng.count(p.fork.0, p.fork.1);
            for c in 0..n {
                let phi = c as f32 / n as f32 * TAU + rng.range(-0.3, 0.3);
                let ang = rng.range(p.fork_angle.0, p.fork_angle.1);
                let d = Vec3::new(phi.cos() * ang.sin(), ang.cos(), phi.sin() * ang.sin()).normalize();
                let len = p.height * (1.0 - p.trunk_frac) * rng.range(0.85, 1.05) / (0.55 + 0.45 * ang.cos());
                grow(&mut rng, p, &mut tree, top.0, d, len, top.2 * 0.75, 1);
            }
        }
    }
    // leaf attachment points along twigs (level 3), the outer part of level 2 and the ends of
    // the main limbs
    let mut leaves = Vec::new();
    for b in tree.branches.iter() {
        let count = match b.level {
            3 => p.leaves_per_twig,
            2 => p.leaves_per_twig / 2,
            1 => p.leaves_per_twig / 3,
            _ => 0,
        };
        let t0 = if b.level == 1 { 0.6 } else { 0.25 };
        for k in 0..count {
            let t = t0 + (1.0 - t0) * (k as f32 + rng.f()) / count as f32;
            let (pos, d, _) = b.at(t);
            leaves.push((pos, d));
        }
    }
    tree.leaves = leaves;
    tree
}

struct Crown {
    center: Vec3,
    radii: Vec3,
    height: f32,
}

fn crown_of(tree: &Tree, height: f32, scale: Vec3) -> Crown {
    let n = tree.leaves.len().max(1) as f32;
    let center = tree.leaves.iter().fold(Vec3::ZERO, |a, l| a + l.0) / n;
    let mut ext = Vec3::ZERO;
    for l in &tree.leaves {
        ext = ext.max((l.0 - center).abs());
    }
    let radii = (ext * scale).max(Vec3::splat(0.5));
    Crown { center, radii, height }
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
    let a = 0.4 + 0.6 * ((depth - 0.2) / 0.8).clamp(0.0, 1.0);
    (a * (0.72 + 0.28 * height)).clamp(0.25, 1.0)
}

/// One ring of a tube: centre, direction, radius, distance along the branch.
struct Ring {
    p: Vec3,
    t: Vec3,
    r: f32,
    along: f32,
}

/// The rings of a branch at every `step`th point.
fn branch_rings(b: &Branch, step: usize) -> Vec<Ring> {
    let mut idxs: Vec<usize> = (0..b.pts.len()).step_by(step.max(1)).collect();
    if *idxs.last().unwrap() != b.pts.len() - 1 {
        idxs.push(b.pts.len() - 1);
    }
    let mut out = Vec::with_capacity(idxs.len());
    let mut along = 0.0;
    for (k, &i) in idxs.iter().enumerate() {
        let t = if i + 1 < b.pts.len() { (b.pts[i + 1] - b.pts[i]).normalize() } else { (b.pts[i] - b.pts[i - 1]).normalize() };
        if k > 0 {
            along += b.pts[i].distance(b.pts[idxs[k - 1]]);
        }
        out.push(Ring { p: b.pts[i], t, r: b.rads[i], along });
    }
    out
}

/// Root flare: the trunk widens towards the ground and spreads into buttresses over its
/// lowest metre or so, and carries on a little below the ground so it meets it on a slope.
/// Returns the rings and the radius multiplier per ring for the buttresses, as a function of
/// the angle around the trunk.
struct Flare {
    amount: f32,
    lobes: u32,
    phase: f32,
}

fn flared_rings(b: &Branch, step: usize, flare: &Flare) -> (Vec<Ring>, Vec<f32>) {
    let len = b.length();
    let (p0, d0, r0) = b.at(0.0);
    let mut rings = vec![Ring { p: p0 - d0 * 0.45, t: d0, r: r0, along: -0.45 }];
    let base = [0.0f32, 0.1, 0.25, 0.45, 0.75, 1.15, 1.7];
    for &s in &base {
        let (p, d, r) = b.at(s / len);
        rings.push(Ring { p, t: d, r, along: s });
    }
    let cut = *base.last().unwrap() + 0.4;
    for r in branch_rings(b, step) {
        if r.along > cut {
            rings.push(r);
        }
    }
    let lobe_amp: Vec<f32> = rings.iter().map(|r| if flare.lobes > 0 { 0.6 * flare.amount * (-r.along.max(0.0) / 0.5).exp() } else { 0.0 }).collect();
    for r in rings.iter_mut() {
        r.r *= 1.0 + flare.amount * (-r.along.max(0.0) / 0.5).exp();
    }
    (rings, lobe_amp)
}

/// A tube through `rings` (the bark texture wraps a whole number of times around it, sized for
/// the girth `girth`: tree.wgsl, `bark_uv`), with normals taken from the surface itself so
/// buttresses shade as what they are.
fn emit_rings(mesh: &mut Mesh, rings: &[Ring], lobes: Option<(&[f32], &Flare)>, sides: usize, level: u32, girth: f32, crown: &Crown) {
    let mut nrm = basis(rings[0].t).0;
    let mut prev_t = rings[0].t;
    let start = mesh.verts.len();
    let mut grid: Vec<Vec3> = Vec::with_capacity(rings.len() * (sides + 1));
    for (k, ring) in rings.iter().enumerate() {
        let t = ring.t;
        // parallel transport the frame
        let axis = prev_t.cross(t);
        if axis.length() > 1e-5 {
            let ang = prev_t.dot(t).clamp(-1.0, 1.0).acos();
            nrm = glam::Quat::from_axis_angle(axis.normalize(), ang) * nrm;
        }
        nrm = (nrm - t * nrm.dot(t)).normalize();
        let bin = t.cross(nrm);
        prev_t = t;
        for s in 0..=sides {
            let a = s as f32 / sides as f32 * TAU;
            let off = nrm * a.cos() + bin * a.sin();
            let mut r = ring.r;
            if let Some((amp, f)) = lobes {
                let l = (0.5 + 0.5 * (f.lobes as f32 * a + f.phase).cos()).powi(2);
                r *= 1.0 + amp[k] * (l * 1.4 - 0.35);
            }
            let p = ring.p + off * r;
            grid.push(p);
            let ao = (0.45 + 0.55 * (p.y / crown.height).clamp(0.0, 1.0)) * if level == 0 { 1.0 } else { 0.8 };
            let v = mesh.vert(p, off, Vec2::new(s as f32 / sides as f32, ring.along), KIND_BARK, sway_of(p, crown), ao);
            mesh.verts[v as usize].attr[3] = girth;
        }
    }
    // normals from the surface: across the rings and around them
    let w = sides + 1;
    for k in 0..rings.len() {
        for s in 0..=sides {
            let around = grid[k * w + (s + 1) % sides] - grid[k * w + (s + sides - 1) % sides];
            let (ka, kb) = (k.saturating_sub(1), (k + 1).min(rings.len() - 1));
            let along = grid[kb * w + s] - grid[ka * w + s];
            let n = around.cross(along).normalize_or(Vec3::ZERO);
            let v = &mut mesh.verts[start + k * w + s];
            let off = Vec3::from(v.nrm);
            if n != Vec3::ZERO {
                v.nrm = (if n.dot(off) < 0.0 { -n } else { n }).into();
            }
        }
    }
    let s0 = start as u32;
    for k in 0..rings.len() as u32 - 1 {
        let (a, c) = (s0 + k * w as u32, s0 + (k + 1) * w as u32);
        for s in 0..sides as u32 {
            mesh.tri(a + s, c + s, a + s + 1);
            mesh.tri(a + s + 1, c + s, c + s + 1);
        }
    }
}

/// The bark wraps around a branch sized for its girth a fifth of the way up (eye level on a
/// trunk), so the same scale holds along the branch.
fn girth_of(b: &Branch) -> f32 {
    TAU * b.rads[(b.rads.len() - 1) / 5]
}

fn emit_tube(mesh: &mut Mesh, b: &Branch, sides: usize, step: usize, crown: &Crown) {
    let rings = branch_rings(b, step);
    emit_rings(mesh, &rings, None, sides, b.level, girth_of(b), crown);
}

fn emit_trunk(mesh: &mut Mesh, b: &Branch, sides: usize, step: usize, crown: &Crown, flare: &Flare) {
    if flare.amount <= 0.0 {
        return emit_tube(mesh, b, sides, step, crown);
    }
    let (rings, amp) = flared_rings(b, step, flare);
    emit_rings(mesh, &rings, Some((&amp, flare)), sides, b.level, girth_of(b), crown);
}

/// A leaf card: `len` along `axis` from `attach`, `width` across, showing one cell of the
/// foliage atlas (its base at the attach point), mirrored or not.
#[allow(clippy::too_many_arguments)]
fn emit_card(mesh: &mut Mesh, attach: Vec3, axis: Vec3, side: Vec3, len: f32, width: f32, crown: &Crown, cell: &Cell, mirror: bool) {
    let b0 = attach - side * width * 0.5;
    let b1 = attach + side * width * 0.5;
    let corners = [b0, b1, b1 + axis * len, b0 + axis * len];
    let [u0, v0, u1, v1] = cell.rect;
    let (u0, u1) = if mirror { (u1, u0) } else { (u0, u1) };
    let uvs = [Vec2::new(u0, v1), Vec2::new(u1, v1), Vec2::new(u1, v0), Vec2::new(u0, v0)];
    let card_n = axis.cross(side).normalize();
    let start = mesh.verts.len() as u32;
    for k in 0..4 {
        let p = corners[k];
        let sph = ((p - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let cn = if card_n.dot(sph) < 0.0 { -card_n } else { card_n };
        let n = (sph * 0.75 + cn * 0.25 + Vec3::Y * 0.15).normalize();
        let v = mesh.vert(p, n, uvs[k], KIND_LEAF, sway_of(p, crown), leaf_ao(p, crown));
        mesh.verts[v as usize].attr[3] = cell.layer as f32;
    }
    mesh.tri(start, start + 1, start + 2);
    mesh.tri(start, start + 2, start + 3);
}

/// Leaves deep inside a crown are in the shade of the outer ones and die off, so a crown
/// is a shell of leaves around a mostly open inside. Light gets through it to the forest
/// floor, and from below you see the limbs.
fn keep_leaf(pos: Vec3, crown: &Crown, rng: &mut Rng) -> bool {
    let depth = ((pos - crown.center) / crown.radii).length();
    let t = ((depth - 0.35) / 0.5).clamp(0.0, 1.0);
    rng.f() < 0.1 + t * t * (3.0 - 2.0 * t)
}

/// One of `cells` at random, mirrored half the time.
fn pick_cell<'a>(rng: &mut Rng, cells: &'a [Cell]) -> (&'a Cell, bool) {
    let i = ((rng.f() * cells.len() as f32) as usize).min(cells.len() - 1);
    (&cells[i], rng.f() < 0.5)
}

#[allow(clippy::too_many_arguments)]
fn place_cards(mesh: &mut Mesh, tree: &Tree, crown: &Crown, rng: &mut Rng, levels: &[u32], per_branch: u32, size: f32, cells: &[Cell]) {
    for b in tree.branches.iter().filter(|b| levels.contains(&b.level)) {
        for k in 0..per_branch {
            let t = 0.2 + 0.8 * (k as f32 + rng.f()) / per_branch as f32;
            let (pos, d, _) = b.at(t);
            if !keep_leaf(pos, crown, rng) {
                continue;
            }
            let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
            let axis = (d * 0.45 + out * 0.55 + rng.unit() * 0.55).normalize();
            let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
            let s = size * rng.range(0.85, 1.15);
            let (cell, mirror) = pick_cell(rng, cells);
            emit_card(mesh, pos - axis * s * 0.15, axis, side, s, s, crown, cell, mirror);
        }
    }
}

/// `near` are the atlas cells of the full-detail cards, `far` those (with more, smaller leaves)
/// of the bigger cards further out.
fn broadleaf(name: String, p: Params, seed: u64, near: &[Cell], far: &[Cell]) -> TreeKind {
    let tree = build_skeleton(&p, seed);
    let crown = crown_of(&tree, p.height, p.crown_scale);
    let mut rng = Rng(seed ^ 0xABCDEF);
    let flare = Flare { amount: p.flare, lobes: p.lobes, phase: rng.range(0.0, TAU) };
    let mut lods: Vec<Mesh> = (0..MESH_LODS).map(|_| Mesh::default()).collect();
    let trunk_sides = if p.lobes > 0 { 3 * p.lobes as usize } else { 10 };

    // LOD0 (the twigs are in the scanned leaf clusters already)
    for b in tree.branches.iter().filter(|b| b.level < 3) {
        if b.level == 0 && p.stems <= 1 {
            emit_trunk(&mut lods[0], b, trunk_sides, 1, &crown, &flare);
        } else {
            let sides = [if p.stems > 1 { 5 } else { 10 }, 6, 4, 3][b.level as usize];
            emit_tube(&mut lods[0], b, sides, 1, &crown);
        }
    }
    for &(pos, d) in &tree.leaves {
        if !keep_leaf(pos, &crown, &mut rng) {
            continue;
        }
        let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let axis = (d * 0.5 + out * 0.5 + rng.unit() * 0.6).normalize();
        let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
        let s = p.leaf_size * rng.range(0.8, 1.2);
        let (cell, mirror) = pick_cell(&mut rng, near);
        emit_card(&mut lods[0], pos - axis * s * 0.1, axis, side, s, s, &crown, cell, mirror);
    }

    // LOD1: the trunk and main limbs, and fewer, bigger cards with more, smaller leaves on each
    for b in tree.branches.iter().filter(|b| b.level <= 1) {
        if b.level == 0 && p.stems <= 1 {
            let lflare = Flare { lobes: 0, ..flare };
            emit_trunk(&mut lods[1], b, 7, 2, &crown, &lflare);
        } else {
            emit_tube(&mut lods[1], b, if b.level == 0 { 5 } else { 4 }, 2, &crown);
        }
    }
    let per_twig = (p.leaves_per_twig as f32 * 0.42).round().max(2.0) as u32;
    place_cards(&mut lods[1], &tree, &crown, &mut rng, &[3], per_twig, p.leaf_size * 1.4, far);
    place_cards(&mut lods[1], &tree, &crown, &mut rng, &[1, 2], (per_twig * 2) / 3, p.leaf_size * 1.7, far);

    finish(name, lods)
}

struct SpruceForm {
    height: f32,
    /// lowest live branches
    base_y: f32,
    max_r: f32,
    trunk_r: f32,
    flare: f32,
}

fn spruce(name: String, f: SpruceForm, seed: u64) -> TreeKind {
    let mut rng = Rng(seed);
    let h = f.height;
    let base_y = f.base_y;
    let max_r = f.max_r;
    let trunk = Branch {
        pts: (0..=12).map(|i| Vec3::new(rng.range(-0.04, 0.04), h * i as f32 / 12.0, rng.range(-0.04, 0.04))).collect(),
        rads: (0..=12).map(|i| f.trunk_r * (1.0 - 0.92 * i as f32 / 12.0)).collect(),
        level: 0,
    };
    let crown = Crown { center: Vec3::new(0.0, (base_y + h) * 0.45, 0.0), radii: Vec3::new(max_r, (h - base_y) * 0.55, max_r), height: h };
    let flare = Flare { amount: f.flare, lobes: 0, phase: 0.0 };
    // whorls of branches, longest at the bottom
    struct Limb {
        b: Branch,
        len: f32,
        whorl: usize,
    }
    let mut limbs: Vec<Limb> = Vec::new();
    let mut y = base_y;
    let mut whorl = 0;
    while y < h - 0.5 {
        let frac = (h - y) / (h - base_y);
        let n = rng.count(5, 7);
        let off = rng.range(0.0, TAU);
        for k in 0..n {
            let phi = off + k as f32 / n as f32 * TAU + rng.range(-0.25, 0.25);
            let len = 0.35 + max_r * frac.powf(0.9) * rng.range(0.85, 1.1);
            // low branches droop, but not into the ground
            let mut el = rng.range(-0.18, 0.06) - 0.28 * frac;
            el = el.max(-((y - 0.15).max(0.0) / len).min(1.0).asin() * 0.8);
            let d0 = Vec3::new(phi.cos() * el.cos(), el.sin(), phi.sin() * el.cos());
            let start = Vec3::new(0.0, y, 0.0);
            let mut pts = vec![start];
            let mut d = d0;
            for _ in 0..3 {
                d = (d + Vec3::Y * 0.12).normalize();
                pts.push(*pts.last().unwrap() + d * len / 3.0);
            }
            let r0 = 0.03 + 0.05 * frac;
            limbs.push(Limb { b: Branch { pts, rads: vec![r0, r0 * 0.7, r0 * 0.45, r0 * 0.2], level: 1 }, len, whorl });
        }
        // young trees put on shorter shoots between whorls
        y += rng.range(0.38, 0.5) * (h / 24.0).clamp(0.55, 1.0);
        whorl += 1;
    }

    let mut lods: Vec<Mesh> = (0..MESH_LODS).map(|_| Mesh::default()).collect();
    // LOD0: every branch, three needle cards each
    emit_trunk(&mut lods[0], &trunk, 9, 1, &crown, &flare);
    for l in &limbs {
        emit_tube(&mut lods[0], &l.b, 3, 1, &crown);
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for (k, roll) in [(0.0f32, 0.0f32), (0.4, 0.25), (0.15, 1.2)] {
            let roll = roll + rng.range(-0.3, 0.3);
            let side = glam::Quat::from_axis_angle(d, roll) * flat;
            let card_len = l.len * 0.68;
            let w = (l.len * 0.55).clamp(0.45, 1.3);
            let (cell, mirror) = pick_cell(&mut rng, &foliage::SPRUCE[..2]);
            emit_card(&mut lods[0], l.b.at(k).0, d, side, card_len, w, &crown, cell, mirror);
        }
    }
    // LOD1: every other whorl, two bigger cards per branch + inner cone
    emit_trunk(&mut lods[1], &trunk, 5, 2, &crown, &flare);
    for (i, l) in limbs.iter().enumerate() {
        if (l.whorl % 2) != 0 || i % 6 == 5 {
            continue;
        }
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for roll in [0.0f32, 1.3] {
            let side = glam::Quat::from_axis_angle(d, roll + rng.range(-0.3, 0.3)) * flat;
            let (cell, mirror) = pick_cell(&mut rng, &foliage::SPRUCE[2..]);
            emit_card(&mut lods[1], l.b.pts[0], d, side, l.len * 1.08, (l.len * 0.9).clamp(0.8, 2.4), &crown, cell, mirror);
        }
    }
    emit_cone(&mut lods[1], h, base_y, max_r * 0.7, 10, &crown);
    finish(name, lods)
}

fn emit_cone(mesh: &mut Mesh, h: f32, base_y: f32, r: f32, sides: usize, crown: &Crown) {
    let span = h - base_y;
    let rings = [(base_y - 0.6, r * 0.55), (base_y + 0.6, r), (base_y + span * 0.42, r * 0.68), (base_y + span * 0.74, r * 0.32)];
    let mut starts = Vec::new();
    for &(y, rr) in &rings {
        let start = mesh.verts.len() as u32;
        for s in 0..sides {
            let a = s as f32 / sides as f32 * TAU;
            let p = Vec3::new(a.cos() * rr, y.max(0.05), a.sin() * rr);
            let n = Vec3::new(a.cos(), 0.45, a.sin()).normalize();
            let ao = 0.4 + 0.6 * (y / h).max(0.0);
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

fn finish(name: String, lods: Vec<Mesh>) -> TreeKind {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for v in &lods[0].verts {
        let p = Vec3::from(v.pos);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let center_y = (lo.y.max(0.0) + hi.y) * 0.5;
    let c = Vec3::new(0.0, center_y, 0.0);
    let reach = |m: &Mesh| m.verts.iter().map(|v| Vec3::from(v.pos).distance(c)).fold(0.0, f32::max);
    let half = reach(&lods[0]) * 1.02;
    let radius = lods.iter().map(reach).fold(0.0, f32::max) + 1.0;
    TreeKind { name, lods, center_y, radius, half }
}

/// Every species' four variants, species-major (index species * VARIANTS + variant).
pub fn build_all() -> Vec<TreeKind> {
    // Forest-grown oaks: they shed their lower branches in the shade, so the trunks stand bare
    // up to a crown that starts about halfway up.
    let oak = Params { trunk_r: TRUNK_R[0][0], height: 21.0, trunk_frac: 0.46, child_start: [0.72, 0.25, 0.2], leaves_per_twig: 9, leaf_size: 1.4, flare: 0.95, lobes: 5, ..BASE };
    let oak_b = Params {
        height: 23.5,
        trunk_r: TRUNK_R[0][1],
        trunk_frac: 0.53,
        child_start: [0.78, 0.25, 0.2],
        angle: [(0.45, 0.8), (0.5, 0.95), (0.5, 1.0)],
        fork: (3, 3),
        fork_angle: (0.3, 0.55),
        lean: 0.07,
        ..oak
    };
    // Open-grown: a short trunk that soon divides, limbs off it from low down and spreading
    // wide, a dense, broad crown that comes down to a few metres off the ground.
    let oak_open = Params {
        height: 15.0,
        trunk_r: TRUNK_R[0][2],
        trunk_frac: 0.34,
        children: [(3, 4), (7, 9), (4, 5)],
        child_start: [0.5, 0.12, 0.15],
        angle: [(1.0, 1.25), (0.55, 1.0), (0.5, 1.0)],
        len_ratio: [1.45, 0.45, 0.5],
        tropism: [0.0, 0.07, 0.04, -0.02],
        wiggle: [0.04, 0.12, 0.25, 0.3],
        leaves_per_twig: 10,
        leaf_size: 1.6,
        crown_scale: Vec3::new(1.15, 1.0, 1.15),
        fork: (5, 6),
        fork_angle: (0.5, 0.95),
        flare: 0.9,
        ..oak
    };
    // Young: a clump of stems, leafy to the ground (hazel-like), thickening forest edges.
    let oak_young = Params {
        height: 6.0,
        trunk_r: 0.06,
        children: [(3, 4), (3, 3), (2, 3)],
        child_start: [0.15, 0.25, 0.25],
        angle: [(0.5, 0.95), (0.5, 1.0), (0.5, 1.0)],
        len_ratio: [0.5, 0.55, 0.5],
        excurrent: true,
        tropism: [0.0, 0.02, 0.0, -0.04],
        wiggle: [0.08, 0.18, 0.28, 0.3],
        leaf_size: 1.1,
        leaves_per_twig: 8,
        crown_scale: Vec3::ONE,
        stems: 4,
        flare: 0.0,
        lobes: 0,
        ..oak
    };
    let birch = Params {
        height: 17.0,
        trunk_frac: 1.0,
        trunk_r: TRUNK_R[1][0],
        children: [(11, 14), (4, 6), (3, 5)],
        child_start: [0.45, 0.2, 0.25],
        angle: [(0.55, 0.85), (0.5, 0.9), (0.6, 1.1)],
        len_ratio: [0.4, 0.6, 0.55],
        excurrent: true,
        tropism: [0.0, 0.03, -0.12, -0.22],
        wiggle: [0.04, 0.14, 0.22, 0.3],
        leaf_size: 1.05,
        leaves_per_twig: 9,
        crown_scale: Vec3::new(1.05, 1.0, 1.05),
        flare: 0.45,
        ..BASE
    };
    let birch_b = Params { height: 19.0, trunk_r: TRUNK_R[1][1], child_start: [0.52, 0.2, 0.25], children: [(10, 13), (4, 6), (3, 5)], lean: 0.1, ..birch };
    let birch_open = Params {
        height: 15.0,
        trunk_r: TRUNK_R[1][2],
        children: [(16, 20), (4, 6), (3, 5)],
        child_start: [0.16, 0.2, 0.25],
        angle: [(0.75, 1.1), (0.5, 0.9), (0.6, 1.1)],
        len_ratio: [0.48, 0.6, 0.55],
        tropism: [0.0, 0.0, -0.18, -0.3],
        leaves_per_twig: 11,
        flare: 0.5,
        ..birch
    };
    let birch_young = Params { height: 7.5, trunk_r: TRUNK_R[1][3], children: [(10, 13), (3, 5), (3, 4)], child_start: [0.12, 0.2, 0.25], leaf_size: 0.8, flare: 0.2, ..birch };
    let names = ["broadleaf", "birch", "spruce"];
    let forms = ["A", "B", "open", "young"];
    let n = |s: usize, v: usize| format!("{} {}", names[s], forms[v]);
    let bl = (&foliage::BROADLEAF_NEAR[..], &foliage::BROADLEAF_FAR[..]);
    let bi = (&foliage::BIRCH_NEAR[..], &foliage::BIRCH_FAR[..]);
    let sp = |height, base_y, max_r, v: usize, flare| SpruceForm { height, base_y, max_r, trunk_r: TRUNK_R[2][v], flare };
    let kinds = vec![
        broadleaf(n(0, 0), oak, 0x1234_5678_9ABC, bl.0, bl.1),
        broadleaf(n(0, 1), oak_b, 0x2468_ACE0_1357, bl.0, bl.1),
        broadleaf(n(0, 2), oak_open, 0x0BEE_CAFE_4321, bl.0, bl.1),
        broadleaf(n(0, 3), oak_young, 0x4A2E_1F00_7777, bl.0, bl.1),
        broadleaf(n(1, 0), birch, 0xBEEF_F00D_1357, bi.0, bi.1),
        broadleaf(n(1, 1), birch_b, 0x7E57_B1C4_9911, bi.0, bi.1),
        broadleaf(n(1, 2), birch_open, 0x0FE2_B1C4_5150, bi.0, bi.1),
        broadleaf(n(1, 3), birch_young, 0x1B1C_0D0E_2233, bi.0, bi.1),
        spruce(n(2, 0), sp(24.0, 3.2, 4.3, 0, 0.5), 0x5EED_5EED_2468),
        spruce(n(2, 1), sp(27.0, 6.5, 3.8, 1, 0.5), 0x5EED_0A0B_9753),
        spruce(n(2, 2), sp(20.0, 0.35, 5.0, 2, 0.6), 0x5EED_C0DE_1111),
        spruce(n(2, 3), sp(7.5, 0.15, 2.1, 3, 0.2), 0x5EED_FACE_2222),
    ];
    for k in &kinds {
        let tris: Vec<String> = k.lods.iter().map(|m| m.tris().to_string()).collect();
        println!("tree '{}': LOD tris [{}], bound r={:.1} m, impostor half {:.1} m", k.name, tris.join(", "), k.radius, k.half);
    }
    kinds
}

/// Forest-floor plants, drawn with the tree shader (see undergrowth.rs). Each comes with the
/// species bits its records carry (they pick the bark scan and the tint in tree.wgsl; the leaf
/// cards carry their own atlas layer) and the radius of a sphere around it (for culling).
pub struct Plant {
    pub name: &'static str,
    pub mesh: Mesh,
    pub layer: u32,
    pub radius: f32,
}

fn plant_crown(height: f32, radius: f32) -> Crown {
    Crown { center: Vec3::new(0.0, height * 0.5, 0.0), radii: Vec3::new(radius, height * 0.5, radius).max(Vec3::splat(0.2)), height }
}

/// A fern: fronds rising from one crown and arching outwards, each a bent strip showing one
/// scanned frond (as wide as its atlas cell, so the frond keeps its proportions).
fn fern(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let mut m = Mesh::default();
    let fronds = 19;
    for f in 0..fronds {
        let phi = f as f32 / fronds as f32 * TAU + rng.range(-0.2, 0.2);
        let out = Vec3::new(phi.cos(), 0.0, phi.sin());
        let len = rng.range(0.7, 1.1);
        let (cell, mirror) = pick_cell(&mut rng, &foliage::FERN);
        let [u0, v0, u1, v1] = cell.rect;
        let (u0, u1) = if mirror { (u1, u0) } else { (u0, u1) };
        let width = len * (cell.rect[2] - cell.rect[0]) / (cell.rect[3] - cell.rect[1]) * 1.1;
        let mut el = rng.range(0.85, 1.25); // starting angle above the horizontal
        let segs = 5;
        let mut p = out * 0.04;
        let side = Vec3::Y.cross(out).normalize();
        let start = m.verts.len() as u32;
        for k in 0..=segs {
            let t = k as f32 / segs as f32;
            let dir = out * el.cos() + Vec3::Y * el.sin();
            // the frond is cupped a little across its width
            let n = (dir.cross(side).normalize() * -1.0 + Vec3::Y * 0.6).normalize();
            let n = if n.y < 0.0 { -n } else { n };
            let w = width * 0.5;
            let sway = 0.5 * t * t;
            let ao = 0.45 + 0.55 * t;
            let v = v1 + (v0 - v1) * t;
            for (x, u) in [(-1.0, u0), (1.0, u1)] {
                let i = m.vert(p + side * w * x + Vec3::Y * w * 0.12, n, Vec2::new(u, v), KIND_LEAF, sway, ao);
                m.verts[i as usize].attr[3] = cell.layer as f32;
            }
            p += dir * (len / segs as f32);
            el -= rng.range(0.22, 0.32); // arching over
        }
        for k in 0..segs as u32 {
            let a = start + k * 2;
            m.tri(a, a + 1, a + 3);
            m.tri(a, a + 3, a + 2);
        }
    }
    m
}

/// A shrub: a few stems from the ground, branching twice, with small leaf cards.
fn shrub(seed: u64, height: f32, card: f32) -> Mesh {
    let mut rng = Rng(seed);
    let p = Params {
        height,
        trunk_frac: 1.0,
        trunk_r: 0.025,
        children: [(3, 4), (3, 4), (2, 3)],
        child_start: [0.3, 0.3, 0.3],
        angle: [(0.5, 0.9), (0.6, 1.0), (0.5, 1.0)],
        len_ratio: [0.6, 0.55, 0.5],
        excurrent: false,
        tropism: [0.05, 0.05, 0.0, -0.05],
        wiggle: [0.1, 0.2, 0.3, 0.3],
        leaf_size: card,
        leaves_per_twig: 3,
        crown_scale: Vec3::ONE,
        ..BASE
    };
    let mut tree = Tree { branches: Vec::new(), leaves: Vec::new() };
    let stems = rng.count(3, 5);
    for k in 0..stems {
        let phi = k as f32 / stems as f32 * TAU + rng.range(-0.4, 0.4);
        let d = Vec3::new(phi.cos() * 0.5, 1.0, phi.sin() * 0.5).normalize();
        let len = height * rng.range(0.75, 1.0);
        grow(&mut rng, &p, &mut tree, Vec3::new(phi.cos(), 0.0, phi.sin()) * 0.05, d, len, 0.022, 1);
    }
    for b in tree.branches.iter() {
        let count = match b.level {
            3 => p.leaves_per_twig,
            2 => 1,
            _ => 0,
        };
        for k in 0..count {
            let t = 0.3 + 0.7 * (k as f32 + rng.f()) / count as f32;
            let (pos, d, _) = b.at(t);
            tree.leaves.push((pos, d));
        }
    }
    let crown = crown_of(&tree, height, Vec3::ONE);
    let mut m = Mesh::default();
    for b in tree.branches.iter().filter(|b| b.level == 1) {
        emit_tube(&mut m, b, 3, 2, &crown);
    }
    for &(pos, d) in &tree.leaves {
        let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let axis = (d * 0.5 + out * 0.5 + rng.unit() * 0.6).normalize();
        let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
        let s = card * rng.range(0.8, 1.2);
        let (cell, mirror) = pick_cell(&mut rng, &foliage::BROADLEAF_NEAR);
        emit_card(&mut m, pos - axis * s * 0.1, axis, side, s, s, &crown, cell, mirror);
    }
    for v in m.verts.iter_mut() {
        v.attr[1] *= 0.35; // stiff: short stems barely sway
    }
    m
}

/// Low ground cover (woodruff, bramble, seedlings): a clump of leaf cards just above the ground.
fn ground_cover(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let crown = plant_crown(0.35, 0.5);
    let mut m = Mesh::default();
    for k in 0..13 {
        let phi = k as f32 / 13.0 * TAU + rng.range(-0.3, 0.3);
        let out = Vec3::new(phi.cos(), 0.0, phi.sin());
        let el = rng.range(0.25, 0.7);
        let axis = (out * el.cos() + Vec3::Y * el.sin()).normalize();
        let side = Vec3::Y.cross(out).normalize();
        let s = rng.range(0.34, 0.5);
        let base = out * rng.range(0.0, 0.15) + Vec3::Y * rng.range(0.0, 0.08);
        let start = m.verts.len() as u32;
        let (cell, mirror) = pick_cell(&mut rng, &foliage::SPRIG);
        emit_card(&mut m, base, axis, side, s, s, &crown, cell, mirror);
        for v in &mut m.verts[start as usize..] {
            v.nrm = (Vec3::from(v.nrm) * 0.4 + Vec3::Y).normalize().into();
            v.attr[1] *= 0.25;
        }
    }
    m
}

/// A disc closing a tube: the cut end of a log or a stump, its texture coordinates the offset
/// from the centre in radii (for the growth rings, tree.wgsl).
fn emit_cap(mesh: &mut Mesh, c: Vec3, n: Vec3, r: f32, sides: usize, rng: &mut Rng) {
    let (u, v) = basis(n);
    let start = mesh.verts.len() as u32;
    let mid = mesh.vert(c, n, Vec2::ZERO, KIND_WOOD, 0.0, 0.9);
    let _ = mid;
    for s in 0..sides {
        let a = s as f32 / sides as f32 * TAU;
        let k = 1.0 + rng.range(-0.05, 0.03);
        let o = (u * a.cos() + v * a.sin()) * r * k;
        mesh.vert(c + o + n * rng.range(-0.02, 0.02) * r, n, Vec2::new(a.cos(), a.sin()) * k, KIND_WOOD, 0.0, 0.8);
    }
    for s in 0..sides as u32 {
        mesh.tri(start, start + 1 + s, start + 1 + (s + 1) % sides as u32);
    }
}

/// Clutter is lit with its ambient occlusion from how much it faces the sky (the underside of
/// a log against the ground gets almost none) and doesn't sway.
fn settle(m: &mut Mesh, kind: f32, ground_ao: f32) {
    for v in m.verts.iter_mut() {
        if v.attr[0] != KIND_WOOD {
            v.attr[0] = kind;
        }
        v.attr[1] = 0.0;
        let up = v.nrm[1] * 0.5 + 0.5;
        v.attr[2] = ground_ao + (1.0 - ground_ao) * up.powf(0.7);
    }
}

/// A fallen trunk lying along x, sunk a little into the ground, with broken-off branch stubs.
fn log(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let len = 4.2;
    let r = 0.27;
    let crown = plant_crown(1.0, 1.0);
    let mut m = Mesh::default();
    let n = 8;
    let pts: Vec<Vec3> = (0..=n).map(|i| Vec3::new(-len * 0.5 + len * i as f32 / n as f32, r * 0.62 + rng.range(-0.03, 0.03), rng.range(-0.06, 0.06))).collect();
    let rads: Vec<f32> = (0..=n).map(|i| r * (1.0 - 0.22 * i as f32 / n as f32)).collect();
    let trunk = Branch { pts, rads, level: 0 };
    emit_tube(&mut m, &trunk, 9, 1, &crown);
    for k in 0..3 {
        let t = 0.25 + 0.25 * k as f32 + rng.range(-0.05, 0.05);
        let (p, d, pr) = trunk.at(t);
        let a = rng.range(-0.6, 2.2);
        let out = Vec3::new(0.0, a.sin(), a.cos()).normalize();
        let dir = (out + d * rng.range(0.2, 0.6)).normalize();
        let stub = Branch { pts: vec![p, p + dir * rng.range(0.25, 0.6)], rads: vec![pr * 0.35, pr * 0.22], level: 1 };
        emit_tube(&mut m, &stub, 5, 1, &crown);
        emit_cap(&mut m, stub.pts[1], dir, pr * 0.22, 5, &mut rng);
    }
    let (p0, d0, r0) = trunk.at(0.0);
    let (p1, d1, r1) = trunk.at(1.0);
    emit_cap(&mut m, p0, -d0, r0, 9, &mut rng);
    emit_cap(&mut m, p1, d1, r1, 9, &mut rng);
    settle(&mut m, KIND_DRAPED, 0.15);
    m
}

/// A fallen branch: a crooked stick with a few side shoots, on the ground.
fn fallen_branch(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let crown = plant_crown(1.0, 1.0);
    let mut m = Mesh::default();
    let len = 2.6;
    let mut pts = vec![Vec3::new(-len * 0.5, 0.05, 0.0)];
    let mut d = Vec3::X;
    for _ in 0..6 {
        d = (d + Vec3::new(0.0, 0.0, rng.range(-0.35, 0.35))).normalize();
        let last = *pts.last().unwrap();
        pts.push(last + d * len / 6.0);
    }
    let rads: Vec<f32> = (0..pts.len()).map(|i| 0.055 * (1.0 - 0.7 * i as f32 / 6.0)).collect();
    let b = Branch { pts, rads, level: 1 };
    emit_tube(&mut m, &b, 5, 1, &crown);
    for k in 0..3 {
        let (p, d, pr) = b.at(0.3 + 0.2 * k as f32);
        let side = Vec3::new(0.0, rng.range(0.0, 0.25), if k % 2 == 0 { 1.0 } else { -1.0 });
        let dir = (d * 0.7 + side).normalize();
        let mut sp = vec![p];
        let mut sd = dir;
        for _ in 0..3 {
            sd = (sd + Vec3::new(0.0, -0.08, rng.range(-0.3, 0.3))).normalize();
            let last = *sp.last().unwrap();
            sp.push(last + sd * rng.range(0.2, 0.35));
        }
        let twig = Branch { rads: vec![pr * 0.6, pr * 0.45, pr * 0.3, pr * 0.15], pts: sp, level: 2 };
        emit_tube(&mut m, &twig, 3, 1, &crown);
    }
    settle(&mut m, KIND_DRAPED, 0.3);
    m
}

/// A stump: what is left of a felled or snapped tree, flared into its roots, its top cut.
fn stump(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let crown = plant_crown(1.0, 1.0);
    let mut m = Mesh::default();
    let h = 0.55;
    let r = 0.36;
    let pts: Vec<Vec3> = (0..=4).map(|i| Vec3::new(0.0, h * i as f32 / 4.0, 0.0)).collect();
    let b = Branch { pts, rads: vec![r; 5], level: 0 };
    let flare = Flare { amount: 0.9, lobes: 5, phase: rng.range(0.0, TAU) };
    emit_trunk(&mut m, &b, 15, 1, &crown, &flare);
    emit_cap(&mut m, Vec3::new(0.0, h, 0.0), (Vec3::Y + Vec3::new(rng.range(-0.15, 0.15), 0.0, rng.range(-0.15, 0.15))).normalize(), r * 1.02, 15, &mut rng);
    settle(&mut m, KIND_BARK, 0.35);
    m
}

/// Hash-based value noise in 3D, for the rocks.
fn vnoise(p: Vec3) -> f32 {
    let h = |x: i32, y: i32, z: i32| {
        let mut n = (x as u32).wrapping_mul(0x8DA6_B343) ^ (y as u32).wrapping_mul(0xD816_3841) ^ (z as u32).wrapping_mul(0xCB1A_B31F);
        n ^= n >> 13;
        n = n.wrapping_mul(0x5BD1_E995);
        (n ^ (n >> 15)) as f32 / u32::MAX as f32
    };
    let f = p.floor();
    let t = p - f;
    let s = t * t * (Vec3::splat(3.0) - 2.0 * t);
    let (x, y, z) = (f.x as i32, f.y as i32, f.z as i32);
    let mut acc = 0.0;
    for (dx, dy, dz) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] {
        let w = (if dx == 1 { s.x } else { 1.0 - s.x }) * (if dy == 1 { s.y } else { 1.0 - s.y }) * (if dz == 1 { s.z } else { 1.0 - s.z });
        acc += w * h(x + dx, y + dy, z + dz);
    }
    acc
}

/// Icosphere with `sub` subdivisions: unit directions and triangles.
fn icosphere(sub: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [(-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0), (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t), (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0)]
        .iter()
        .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
        .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..sub {
        let mut cache = std::collections::HashMap::new();
        let mut nf = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let mut mid = |p: u32, q: u32| -> u32 {
                *cache.entry((p.min(q), p.max(q))).or_insert_with(|| {
                    v.push(((v[p as usize] + v[q as usize]) * 0.5).normalize());
                    v.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            nf.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    (v, f)
}

/// A rock: a lumpy, flattened ball, a quarter of it in the ground. `flat` squashes it into a
/// slab. Textured from the rock scan by its world position (tree.wgsl).
fn rock(seed: u64, flat: f32) -> Mesh {
    let (dirs, faces) = icosphere(2);
    let o = Vec3::new((seed % 97) as f32, (seed % 89) as f32, (seed % 83) as f32) * 1.37;
    let pos: Vec<Vec3> = dirs
        .iter()
        .map(|&d| {
            let r = 0.75 + 0.45 * vnoise(d * 1.3 + o) + 0.12 * vnoise(d * 3.7 + o * 2.0) + 0.04 * vnoise(d * 9.0 - o);
            let p = d * r;
            // facets: flatten towards a few planes, as broken stone is
            Vec3::new(p.x, p.y * flat, p.z) - Vec3::Y * 0.25 * flat
        })
        .collect();
    let mut nrm = vec![Vec3::ZERO; pos.len()];
    for f in &faces {
        let n = (pos[f[1] as usize] - pos[f[0] as usize]).cross(pos[f[2] as usize] - pos[f[0] as usize]);
        for &i in f {
            nrm[i as usize] += n;
        }
    }
    let mut m = Mesh::default();
    for (i, p) in pos.iter().enumerate() {
        let n = nrm[i].normalize_or(dirs[i]);
        m.vert(*p, n, Vec2::ZERO, KIND_ROCK, 0.0, 1.0);
    }
    for f in &faces {
        m.tri(f[0], f[1], f[2]);
    }
    settle(&mut m, KIND_ROCK, 0.3);
    m
}

pub fn build_plants() -> Vec<Plant> {
    // (the clutter's species bits are picked where it lies, undergrowth.wgsl)
    let plants = vec![
        Plant { name: "fern", mesh: fern(0xFE12_0001), layer: 3, radius: 1.3 },
        Plant { name: "shrub", mesh: shrub(0x5821_0B02, 1.6, 0.55), layer: 0, radius: 1.6 },
        Plant { name: "ground cover", mesh: ground_cover(0x6C0F_E203), layer: 0, radius: 0.7 },
        Plant { name: "log", mesh: log(0x106_0004), layer: 0, radius: 3.5 },
        Plant { name: "fallen branch", mesh: fallen_branch(0xB4A_0005), layer: 0, radius: 2.2 },
        Plant { name: "stump", mesh: stump(0x57E_0006), layer: 0, radius: 0.9 },
        Plant { name: "rock", mesh: rock(0x40C_0007, 0.7), layer: 0, radius: 1.3 },
        Plant { name: "slab", mesh: rock(0x51A_0008, 0.42), layer: 0, radius: 1.3 },
    ];
    for p in &plants {
        println!("plant '{}': {} tris", p.name, p.mesh.tris());
    }
    plants
}
