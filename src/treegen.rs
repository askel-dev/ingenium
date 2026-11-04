//! Procedural tree meshes. Every species comes in several shapes, each grown from its own
//! branch skeleton (`build_shapes`):
//!   - forest-grown: they shed their lower branches in the shade, so the trunk stands bare up
//!     to a crown that starts about halfway up, and the crown is a shell of leaves around a
//!     mostly open inside, so sunflecks reach the floor;
//!   - open-grown, for forest edges, groups and lone trees in the meadows: limbs low on the
//!     trunk and a full crown;
//!   - the shrubs along a forest edge (broadleaf only).
//!
//! A tree's seed picks its shape (see scatter.wgsl), so neighbours rarely share one.
//!
//! Each shape has two mesh LODs built from its skeleton, so silhouettes stay consistent:
//!   LOD0  full branches + dense leaf cards
//!   LOD1  main limbs + fewer, larger cards (+ an inner cone for the spruce)
//! Further out a tree is an impostor baked from its LOD0 (impostor.rs).
//!
//! Leaves are lit as clumps: a leaf's normal leans out from the clump it grows in (the leaves
//! of one side branch) as well as from the crown, and leaves deep inside a clump are darker.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::TAU;

pub const KIND_BARK: f32 = 0.0;
pub const KIND_LEAF: f32 = 1.0;
pub const KIND_HULL: f32 = 2.0;

/// `TreeVertex::attr[3]` of meshes that are not trees (the undergrowth).
pub const NO_SLOT: f32 = -1.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TreeVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
    pub attr: [f32; 4], // kind, sway, ao, shape slot (NO_SLOT for plants)
}

pub struct Mesh {
    pub verts: Vec<TreeVertex>,
    pub idx: Vec<u32>,
    /// Written into every vertex, so the shader can look up the shape (see `Shape::slot`).
    slot: f32,
}

impl Mesh {
    pub fn new(slot: f32) -> Mesh {
        Mesh { verts: Vec::new(), idx: Vec::new(), slot }
    }
    fn vert(&mut self, p: Vec3, n: Vec3, uv: Vec2, kind: f32, sway: f32, ao: f32) -> u32 {
        self.verts.push(TreeVertex { pos: p.into(), nrm: n.into(), uv: uv.into(), attr: [kind, sway, ao, self.slot] });
        self.verts.len() as u32 - 1
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.idx.extend_from_slice(&[a, b, c]);
    }
    pub fn tris(&self) -> u32 {
        (self.idx.len() / 3) as u32
    }
}

pub const SPECIES: usize = 3;
pub const SPECIES_NAMES: [&str; SPECIES] = ["broadleaf", "birch", "spruce"];
/// Shape slots per species: the low 3 bits of a tree's seed pick one (see scatter.wgsl).
pub const VARIANTS: usize = 8;
pub const SLOTS: usize = SPECIES * VARIANTS;
/// The shapes that exist: per species 4 forest-grown and 2 open-grown, plus 2 edge shrubs.
pub const SHAPES: usize = 20;

/// Trunk radius at the ground per species (broadleaf, birch, spruce) of a forest-grown tree,
/// before the per-tree scale. The player collides with trunks of this size.
pub const TRUNK_R: [f32; SPECIES] = [0.42, 0.22, 0.34];
/// Open-grown trees put on more girth.
const OPEN_GIRTH: f32 = 1.15;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Form {
    Forest,
    Open,
    Shrub,
}

/// The form of a species' variant (`None` if the species has no such shape): 0-3 forest-grown,
/// 4-5 open-grown, 6-7 edge shrubs (broadleaf only).
pub fn form_of(species: usize, variant: usize) -> Option<Form> {
    match variant {
        0..=3 => Some(Form::Forest),
        4 | 5 => Some(Form::Open),
        6 | 7 if species == 0 => Some(Form::Shrub),
        _ => None,
    }
}

/// Radius of the trunk at the ground, before the tree's scale, that the player walks around:
/// 0 for shrubs, which are walked through.
pub fn trunk_radius(species: usize, variant: usize) -> f32 {
    match form_of(species.min(SPECIES - 1), variant) {
        Some(Form::Forest) => TRUNK_R[species.min(SPECIES - 1)],
        Some(Form::Open) => TRUNK_R[species.min(SPECIES - 1)] * OPEN_GIRTH,
        _ => 0.0,
    }
}

pub struct Shape {
    pub species: usize,
    pub variant: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub form: Form,
    pub lods: [Mesh; 2],
    /// Bounding sphere (for culling): centre height above the base and radius.
    pub center_y: f32,
    pub radius: f32,
    /// The crown as an ellipsoid, for its self-shadowing: centre and radii.
    pub crown_center: Vec3,
    pub crown_radii: Vec3,
    /// How fast sunlight dies out inside the crown, per metre.
    pub density: f32,
}

impl Shape {
    /// Index into the per-shape tables: species * VARIANTS + variant.
    pub fn slot(&self) -> usize {
        self.species * VARIANTS + self.variant
    }
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

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
    /// the main limb (level 1) it grows from
    limb: usize,
    /// the side branch (level 2) it grows from: its leaves form one clump
    clump: usize,
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
    /// Limbs forking off the top of a trunk that isn't excurrent (or a shrub's stems): how many
    /// and their angle from the vertical.
    fork: (u32, u32),
    fork_angle: (f32, f32),
    /// Lowest the branches off the trunk may point (y of their direction).
    min_rise: f32,
    /// How much of the crown's inside has died back: 1 leaves a shell of leaves (forest-grown),
    /// 0 a full crown.
    hollow: f32,
    /// A shrub: stems from the ground instead of a trunk.
    stems: bool,
    /// Sunlight extinction inside the crown, per metre (`Shape::density`).
    density: f32,
}

struct Leaf {
    pos: Vec3,
    dir: Vec3,
    clump: usize,
}

struct Tree {
    branches: Vec<Branch>,
    leaves: Vec<Leaf>,
}

#[allow(clippy::too_many_arguments)]
fn grow(rng: &mut Rng, p: &Params, tree: &mut Tree, start: Vec3, dir: Vec3, len: f32, radius: f32, level: u32, limb: usize, clump: usize) {
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
    let limb = if level == 1 { me } else { limb };
    let clump = if level == 2 { me } else { clump };
    tree.branches.push(Branch { pts, rads, level, limb, clump });
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
            cd.y = cd.y.max(p.min_rise);
            cd = cd.normalize();
        }
        let falloff = if level == 0 && p.excurrent { 1.0 - 0.75 * t } else { 1.0 - 0.35 * t };
        let clen = len * p.len_ratio[level as usize] * falloff * rng.range(0.8, 1.15);
        let crad = (pr * 0.62).max(0.012);
        grow(rng, p, tree, pos, cd, clen, crad, level + 1, limb, clump);
    }
}

fn build_skeleton(p: &Params, seed: u64) -> Tree {
    let mut rng = Rng(seed);
    let mut tree = Tree { branches: Vec::new(), leaves: Vec::new() };
    if p.stems {
        // a shrub: stems from the ground, leaning out
        let n = rng.count(p.fork.0, p.fork.1);
        for k in 0..n {
            let phi = k as f32 / n as f32 * TAU + rng.range(-0.4, 0.4);
            let ang = rng.range(p.fork_angle.0, p.fork_angle.1);
            let d = Vec3::new(phi.cos() * ang.sin(), ang.cos(), phi.sin() * ang.sin());
            let foot = Vec3::new(phi.cos(), 0.0, phi.sin()) * rng.range(0.03, 0.3);
            let len = p.height * rng.range(0.7, 1.0);
            let r = p.trunk_r * rng.range(0.7, 1.1);
            grow(&mut rng, p, &mut tree, foot, d, len, r, 1, 0, 0);
        }
    } else {
        let lean = Vec3::new(rng.range(-0.05, 0.05), 1.0, rng.range(-0.05, 0.05)).normalize();
        grow(&mut rng, p, &mut tree, Vec3::ZERO, lean, p.height * p.trunk_frac, p.trunk_r, 0, 0, 0);
        // non-excurrent trees: the trunk forks into a crown of limbs at its top
        if !p.excurrent {
            let top = tree.branches[0].at(1.0);
            let n = rng.count(p.fork.0, p.fork.1);
            for c in 0..n {
                let phi = c as f32 / n as f32 * TAU + rng.range(-0.3, 0.3);
                let ang = rng.range(p.fork_angle.0, p.fork_angle.1);
                let d = Vec3::new(phi.cos() * ang.sin(), ang.cos(), phi.sin() * ang.sin()).normalize();
                let len = p.height * (1.0 - p.trunk_frac) * rng.range(0.85, 1.05);
                grow(&mut rng, p, &mut tree, top.0, d, len, top.2 * 0.75, 1, 0, 0);
            }
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
            let (pos, dir, _) = b.at(t);
            leaves.push(Leaf { pos, dir, clump: b.clump });
        }
    }
    tree.leaves = leaves;
    tree
}

struct Crown {
    center: Vec3,
    radii: Vec3,
    height: f32,
    /// per clump (see `Branch::clump`): the centre and radius of its leaves
    clumps: HashMap<usize, (Vec3, f32)>,
}

impl Crown {
    fn clump(&self, id: usize) -> Option<(Vec3, f32)> {
        self.clumps.get(&id).copied()
    }
}

fn crown_of(tree: &Tree, height: f32, scale: Vec3) -> Crown {
    let n = tree.leaves.len().max(1) as f32;
    let center = tree.leaves.iter().fold(Vec3::ZERO, |a, l| a + l.pos) / n;
    let mut ext = Vec3::ZERO;
    for l in &tree.leaves {
        ext = ext.max((l.pos - center).abs());
    }
    let radii = (ext * scale).max(Vec3::splat(0.5));
    let mut groups: HashMap<usize, Vec<Vec3>> = HashMap::new();
    for l in &tree.leaves {
        groups.entry(l.clump).or_default().push(l.pos);
    }
    let clumps = groups
        .into_iter()
        .map(|(k, g)| {
            let c = g.iter().fold(Vec3::ZERO, |a, p| a + *p) / g.len() as f32;
            let mut d: Vec<f32> = g.iter().map(|p| p.distance(c)).collect();
            d.sort_by(f32::total_cmp);
            (k, (c, d[(d.len() * 85 / 100).min(d.len() - 1)] + 0.3))
        })
        .collect();
    Crown { center, radii, height, clumps }
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

/// A leaf card. Its normals lean out from the crown and from `clump` (centre, radius), the
/// leaves it grows among, so a crown is lit as a heap of clumps rather than one ball; and the
/// deeper a leaf sits in its clump, the darker.
#[allow(clippy::too_many_arguments)]
fn emit_card(mesh: &mut Mesh, attach: Vec3, axis: Vec3, side: Vec3, len: f32, width: f32, crown: &Crown, clump: Option<(Vec3, f32)>) {
    let b0 = attach - side * width * 0.5;
    let b1 = attach + side * width * 0.5;
    let corners = [b0, b1, b1 + axis * len, b0 + axis * len];
    let uvs = [Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)];
    let card_n = axis.cross(side).normalize();
    let start = mesh.verts.len() as u32;
    for k in 0..4 {
        let p = corners[k];
        let sph = ((p - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let (out, depth) = match clump {
            Some((c, r)) => ((p - c).normalize_or(sph), p.distance(c) / r),
            None => (sph, 1.0),
        };
        let cn = if card_n.dot(sph) < 0.0 { -card_n } else { card_n };
        let n = (sph * 0.4 + out * 0.45 + cn * 0.2 + Vec3::Y * 0.12).normalize();
        let ao = leaf_ao(p, crown) * (0.6 + 0.4 * smoothstep(0.1, 0.9, depth));
        mesh.vert(p, n, uvs[k], KIND_LEAF, sway_of(p, crown), ao);
    }
    mesh.tri(start, start + 1, start + 2);
    mesh.tri(start, start + 2, start + 3);
}

/// Leaves deep inside a crown are in the shade of the outer ones and die off, so a forest-grown
/// crown is a shell of leaves around a mostly open inside. Light gets through it to the forest
/// floor, and from below you see the limbs. A tree in the open keeps its inner leaves
/// (`hollow` near 0).
fn keep_leaf(pos: Vec3, crown: &Crown, hollow: f32, rng: &mut Rng) -> bool {
    let depth = ((pos - crown.center) / crown.radii).length();
    let shell = (0.1 + smoothstep(0.35, 0.85, depth)).min(1.0);
    rng.f() < 1.0 - hollow * (1.0 - shell)
}

#[allow(clippy::too_many_arguments)]
fn place_cards(mesh: &mut Mesh, tree: &Tree, crown: &Crown, rng: &mut Rng, levels: &[u32], per_branch: u32, size: f32, hollow: f32) {
    for b in tree.branches.iter().filter(|b| levels.contains(&b.level)) {
        for k in 0..per_branch {
            let t = 0.2 + 0.8 * (k as f32 + rng.f()) / per_branch as f32;
            let (pos, d, _) = b.at(t);
            if !keep_leaf(pos, crown, hollow, rng) {
                continue;
            }
            let out = ((pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
            let axis = (d * 0.45 + out * 0.55 + rng.unit() * 0.55).normalize();
            let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
            let s = size * rng.range(0.85, 1.15);
            emit_card(mesh, pos - axis * s * 0.15, axis, side, s, s, crown, crown.clump(b.clump));
        }
    }
}

/// A broadleaf tree, birch or shrub grown from `p`.
fn branched(species: usize, variant: usize, form: Form, p: Params, seed: u64) -> Shape {
    let slot = (species * VARIANTS + variant) as f32;
    let tree = build_skeleton(&p, seed);
    let crown = crown_of(&tree, p.height, p.crown_scale);
    let mut rng = Rng(seed ^ 0xABCDEF);
    let mut lod0 = Mesh::new(slot);
    let mut lod1 = Mesh::new(slot);

    for b in &tree.branches {
        let sides = [10, 6, 4, 3][b.level as usize];
        emit_tube(&mut lod0, b, sides, 1, &crown);
    }
    for leaf in &tree.leaves {
        if !keep_leaf(leaf.pos, &crown, p.hollow, &mut rng) {
            continue;
        }
        let out = ((leaf.pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let axis = (leaf.dir * 0.5 + out * 0.5 + rng.unit() * 0.6).normalize();
        let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
        let s = p.leaf_size * rng.range(0.8, 1.2);
        emit_card(&mut lod0, leaf.pos - axis * s * 0.1, axis, side, s, s, &crown, crown.clump(leaf.clump));
    }

    for b in tree.branches.iter().filter(|b| b.level <= 1) {
        emit_tube(&mut lod1, b, if b.level == 0 { 7 } else { 4 }, 2, &crown);
    }
    place_cards(&mut lod1, &tree, &crown, &mut rng, &[3], 3, p.leaf_size * 1.4, p.hollow);
    place_cards(&mut lod1, &tree, &crown, &mut rng, &[2], 2, p.leaf_size * 1.7, p.hollow);

    finish(species, variant, form, [lod0, lod1], &crown, p.density)
}

fn spruce(variant: usize, form: Form, seed: u64) -> Shape {
    let slot = (2 * VARIANTS + variant) as f32;
    let mut rng = Rng(seed);
    let open = form == Form::Open;
    // forest-grown spruce lose their lowest whorls in the shade; in the open they keep them
    let h = rng.range(21.0, 26.0) * if open { 0.85 } else { 1.0 };
    let base_y = if open { rng.range(0.25, 0.5) } else { rng.range(1.6, 3.2) };
    let max_r = if open { rng.range(4.9, 5.6) } else { rng.range(3.9, 4.7) };
    let trunk_r = TRUNK_R[2] * if open { OPEN_GIRTH } else { 1.0 };
    let trunk = Branch {
        pts: (0..=12).map(|i| Vec3::new(rng.range(-0.04, 0.04), h * i as f32 / 12.0, rng.range(-0.04, 0.04))).collect(),
        rads: (0..=12).map(|i| trunk_r * (1.0 - 0.92 * i as f32 / 12.0)).collect(),
        level: 0,
        limb: 0,
        clump: 0,
    };
    let crown = Crown {
        center: Vec3::new(0.0, (h + base_y) * 0.5 - 1.2, 0.0),
        radii: Vec3::new(max_r, (h - base_y) * 0.5 + 1.2, max_r),
        height: h,
        clumps: HashMap::new(),
    };
    // whorls of branches, longest at the bottom
    struct Limb {
        b: Branch,
        len: f32,
    }
    let mut limbs: Vec<Limb> = Vec::new();
    let mut y = base_y;
    let mut whorl = 0;
    let droop = rng.range(0.22, 0.34);
    while y < h - 0.5 {
        let frac = (h - y) / (h - base_y);
        let n = rng.count(5, 7);
        let off = rng.range(0.0, TAU);
        for k in 0..n {
            let phi = off + k as f32 / n as f32 * TAU + rng.range(-0.25, 0.25);
            let el = rng.range(-0.18, 0.06) - droop * frac;
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
            limbs.push(Limb { b: Branch { pts, rads: vec![r0, r0 * 0.7, r0 * 0.45, r0 * 0.2], level: 1, limb: whorl, clump: 0 }, len });
        }
        y += rng.range(0.38, 0.5);
        whorl += 1;
    }
    // A branch's needles are lit as one spray: their normals lean up and out from below its
    // middle, so the tiers catch the light on top.
    let spray = |l: &Limb| -> Option<(Vec3, f32)> { Some((l.b.at(0.4).0 - Vec3::Y * 0.6, l.len * 0.6 + 0.4)) };

    let mut lod0 = Mesh::new(slot);
    let mut lod1 = Mesh::new(slot);
    // LOD0: every branch, three needle cards each
    emit_tube(&mut lod0, &trunk, 8, 1, &crown);
    for l in &limbs {
        emit_tube(&mut lod0, &l.b, 3, 1, &crown);
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for (k, roll) in [(0.0f32, 0.0f32), (0.4, 0.25), (0.15, 1.2)] {
            let roll = roll + rng.range(-0.3, 0.3);
            let side = glam::Quat::from_axis_angle(d, roll) * flat;
            let card_len = l.len * 0.68;
            let w = (l.len * 0.55).clamp(0.45, 1.3);
            emit_card(&mut lod0, l.b.at(k).0, d, side, card_len, w, &crown, spray(l));
        }
    }
    // LOD1: every other whorl, two bigger cards per branch + inner cone
    emit_tube(&mut lod1, &trunk, 5, 2, &crown);
    for (i, l) in limbs.iter().enumerate() {
        if (l.b.limb % 2) != 0 || i % 6 == 5 {
            continue;
        }
        let d = (l.b.pts[3] - l.b.pts[0]).normalize();
        let flat = Vec3::Y.cross(d).normalize_or(Vec3::X);
        for roll in [0.0f32, 1.3] {
            let side = glam::Quat::from_axis_angle(d, roll + rng.range(-0.3, 0.3)) * flat;
            emit_card(&mut lod1, l.b.pts[0], d, side, l.len * 1.08, (l.len * 0.9).clamp(0.8, 2.4), &crown, spray(l));
        }
    }
    emit_cone(&mut lod1, h, base_y, max_r * 0.7, 10, &crown);
    let density = if open { 0.42 } else { 0.38 };
    finish(2, variant, form, [lod0, lod1], &crown, density)
}

fn emit_cone(mesh: &mut Mesh, h: f32, base_y: f32, r: f32, sides: usize, crown: &Crown) {
    let rings = [(base_y - 0.6, r * 0.55), (base_y + 0.6, r), (h * 0.45, r * 0.68), (h * 0.75, r * 0.32)];
    let mut starts = Vec::new();
    for &(y, rr) in &rings {
        let start = mesh.verts.len() as u32;
        for s in 0..sides {
            let a = s as f32 / sides as f32 * TAU;
            let p = Vec3::new(a.cos() * rr, y.max(0.0), a.sin() * rr);
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

fn finish(species: usize, variant: usize, form: Form, lods: [Mesh; 2], crown: &Crown, density: f32) -> Shape {
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
    Shape {
        species,
        variant,
        form,
        lods,
        center_y,
        radius,
        // a little inside the leaves' bounding ellipsoid, which their outermost tips poke out of
        crown_center: crown.center,
        crown_radii: crown.radii * 0.92,
        density,
    }
}

/// Oak-ish broadleaf.
fn oak(form: Form, rng: &mut Rng) -> Params {
    match form {
        Form::Forest => Params {
            height: 21.0 * rng.range(0.9, 1.1),
            trunk_frac: rng.range(0.42, 0.52),
            trunk_r: TRUNK_R[0],
            children: [(2, 3), (5, 7), (4, 6)],
            child_start: [0.72, 0.25, 0.2],
            angle: [(0.5, 0.9), (0.5, 0.95), (0.5, 1.0)],
            len_ratio: [0.6, 0.45 * rng.range(0.9, 1.1), 0.5],
            excurrent: false,
            tropism: [0.02, 0.06, 0.03, -0.02],
            wiggle: [0.05, 0.16, 0.25, 0.3],
            leaf_size: 1.3,
            leaves_per_twig: 6,
            crown_scale: Vec3::new(1.1, 1.0, 1.1),
            // forest-grown: the limbs reach up more than out, for a narrower crown
            fork: (3, 4),
            fork_angle: (0.36 * rng.range(0.85, 1.1), 0.62 * rng.range(0.9, 1.15)),
            min_rise: 0.15,
            hollow: 1.0,
            stems: false,
            density: 0.3,
        },
        // lower and broader, with long limbs low on a short trunk and a full crown
        Form::Open => Params {
            height: 16.0 * rng.range(0.9, 1.1),
            trunk_frac: rng.range(0.3, 0.38),
            trunk_r: TRUNK_R[0] * OPEN_GIRTH,
            children: [(4, 6), (5, 7), (4, 6)],
            child_start: [0.25, 0.2, 0.2],
            angle: [(0.95, 1.35), (0.5, 0.95), (0.5, 1.0)],
            len_ratio: [1.2, 0.5, 0.5],
            excurrent: false,
            tropism: [0.02, 0.02, 0.02, -0.03],
            wiggle: [0.05, 0.16, 0.25, 0.3],
            leaf_size: 1.35,
            leaves_per_twig: 7,
            crown_scale: Vec3::new(1.1, 1.0, 1.1),
            fork: (4, 5),
            fork_angle: (0.55, 0.95),
            min_rise: -0.08,
            hollow: 0.3,
            stems: false,
            density: 0.36,
        },
        // hazel or hawthorn along a forest edge: a thicket of stems, leafy to the ground
        Form::Shrub => Params {
            height: 4.6 * rng.range(0.9, 1.1),
            trunk_frac: 1.0,
            trunk_r: 0.05,
            children: [(0, 0), (4, 6), (3, 5)],
            child_start: [0.0, 0.15, 0.2],
            angle: [(0.0, 0.0), (0.55, 1.0), (0.5, 1.1)],
            len_ratio: [0.0, 0.55, 0.5],
            excurrent: false,
            tropism: [0.0, 0.04, 0.0, -0.04],
            wiggle: [0.0, 0.15, 0.25, 0.3],
            leaf_size: 0.85,
            leaves_per_twig: 5,
            crown_scale: Vec3::ONE,
            fork: (6, 8),
            fork_angle: (0.15, 0.65),
            min_rise: 0.0,
            hollow: 0.15,
            stems: true,
            density: 0.6,
        },
    }
}

fn birch(form: Form, rng: &mut Rng) -> Params {
    let open = form == Form::Open;
    Params {
        height: if open { 15.0 } else { 17.0 } * rng.range(0.9, 1.1),
        trunk_frac: 1.0,
        trunk_r: TRUNK_R[1] * if open { OPEN_GIRTH } else { 1.0 },
        children: [if open { (16, 20) } else { (10 + rng.count(0, 2), 13 + rng.count(0, 2)) }, (4, 6), (3, 5)],
        child_start: [if open { 0.12 } else { 0.45 * rng.range(0.85, 1.15) }, 0.2, 0.25],
        angle: [(0.55, 0.85), (0.5, 0.9), (0.6, 1.1)],
        len_ratio: [if open { 0.36 } else { 0.4 }, 0.6, 0.55],
        excurrent: true,
        tropism: if open { [0.0, 0.0, -0.14, -0.25] } else { [0.0, 0.03, -0.12, -0.22] },
        wiggle: [0.04, 0.14, 0.22, 0.3],
        leaf_size: 1.0,
        leaves_per_twig: 6,
        crown_scale: Vec3::new(1.05, 1.0, 1.05),
        fork: (0, 0),
        fork_angle: (0.0, 0.0),
        min_rise: if open { -0.2 } else { 0.15 },
        // a birch's crown is light and open anyway: hollowing it out like an oak's would leave
        // little more than white branches at a distance
        hollow: if open { 0.3 } else { 0.6 },
        stems: false,
        density: if open { 0.24 } else { 0.2 },
    }
}

/// Every tree shape: per species, variants 0-3 forest-grown, 4-5 open-grown, and for the
/// broadleaf 6-7 edge shrubs (see `form_of`).
pub fn build_shapes() -> Vec<Shape> {
    let mut shapes = Vec::with_capacity(SHAPES);
    for species in 0..SPECIES {
        for variant in 0..VARIANTS {
            let Some(form) = form_of(species, variant) else { continue };
            let seed = [0x1234_5678_9ABC_u64, 0xBEEF_F00D_1357, 0x5EED_5EED_2468][species] ^ (variant as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = Rng(seed | 1);
            let shape = match species {
                0 => branched(0, variant, form, oak(form, &mut rng), seed),
                1 => branched(1, variant, form, birch(form, &mut rng), seed),
                _ => spruce(variant, form, seed),
            };
            shapes.push(shape);
        }
    }
    assert_eq!(shapes.len(), SHAPES);
    for (s, name) in SPECIES_NAMES.iter().enumerate() {
        let of: Vec<&Shape> = shapes.iter().filter(|x| x.species == s).collect();
        let range = |l: usize| {
            let t: Vec<u32> = of.iter().map(|x| x.lods[l].tris()).collect();
            format!("{}-{}", t.iter().min().unwrap(), t.iter().max().unwrap())
        };
        let r = of.iter().map(|x| x.radius).fold(0.0, f32::max);
        println!("tree '{name}': {} shapes, LOD0 {} tris, LOD1 {} tris, bound r<={r:.1} m", of.len(), range(0), range(1));
    }
    shapes
}

/// Forest-floor plants, drawn with the tree shader (see undergrowth.rs). Each comes with the
/// leaf texture layer it uses and the radius of a sphere around it (for culling).
pub struct Plant {
    pub name: &'static str,
    pub mesh: Mesh,
    pub layer: u32,
    pub radius: f32,
}

fn plant_crown(height: f32, radius: f32) -> Crown {
    Crown { center: Vec3::new(0.0, height * 0.5, 0.0), radii: Vec3::new(radius, height * 0.5, radius).max(Vec3::splat(0.2)), height, clumps: HashMap::new() }
}

/// A fern: fronds rising from one crown and arching outwards, each a bent textured strip.
fn fern(seed: u64) -> Mesh {
    let mut rng = Rng(seed);
    let mut m = Mesh::new(NO_SLOT);
    let fronds = 13;
    for f in 0..fronds {
        let phi = f as f32 / fronds as f32 * TAU + rng.range(-0.25, 0.25);
        let out = Vec3::new(phi.cos(), 0.0, phi.sin());
        let len = rng.range(0.75, 1.1);
        let width = len * 0.62;
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
            let w = width * 0.5 * (0.35 + 0.65 * (t * 3.0).min(1.0));
            let sway = 0.5 * t * t;
            let ao = 0.45 + 0.55 * t;
            m.vert(p - side * w + Vec3::Y * w * 0.12, n, Vec2::new(0.0, 1.0 - t), KIND_LEAF, sway, ao);
            m.vert(p + side * w + Vec3::Y * w * 0.12, n, Vec2::new(1.0, 1.0 - t), KIND_LEAF, sway, ao);
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
        fork: (0, 0),
        fork_angle: (0.0, 0.0),
        min_rise: 0.15,
        hollow: 0.0,
        stems: true,
        density: 0.0,
    };
    let mut tree = Tree { branches: Vec::new(), leaves: Vec::new() };
    let stems = rng.count(3, 5);
    for k in 0..stems {
        let phi = k as f32 / stems as f32 * TAU + rng.range(-0.4, 0.4);
        let d = Vec3::new(phi.cos() * 0.5, 1.0, phi.sin() * 0.5).normalize();
        let len = height * rng.range(0.75, 1.0);
        grow(&mut rng, &p, &mut tree, Vec3::new(phi.cos(), 0.0, phi.sin()) * 0.05, d, len, 0.022, 1, k as usize, 0);
    }
    for b in tree.branches.iter() {
        let count = match b.level {
            3 => p.leaves_per_twig,
            2 => 1,
            _ => 0,
        };
        for k in 0..count {
            let t = 0.3 + 0.7 * (k as f32 + rng.f()) / count as f32;
            let (pos, dir, _) = b.at(t);
            tree.leaves.push(Leaf { pos, dir, clump: b.clump });
        }
    }
    let crown = crown_of(&tree, height, Vec3::ONE);
    let mut m = Mesh::new(NO_SLOT);
    for b in tree.branches.iter().filter(|b| b.level == 1) {
        emit_tube(&mut m, b, 3, 2, &crown);
    }
    for leaf in &tree.leaves {
        let out = ((leaf.pos - crown.center) / crown.radii).normalize_or(Vec3::Y);
        let axis = (leaf.dir * 0.5 + out * 0.5 + rng.unit() * 0.6).normalize();
        let side = axis.cross(rng.unit()).normalize_or(Vec3::X);
        let s = card * rng.range(0.8, 1.2);
        emit_card(&mut m, leaf.pos - axis * s * 0.1, axis, side, s, s, &crown, None);
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
    let mut m = Mesh::new(NO_SLOT);
    for k in 0..9 {
        let phi = k as f32 / 9.0 * TAU + rng.range(-0.3, 0.3);
        let out = Vec3::new(phi.cos(), 0.0, phi.sin());
        let el = rng.range(0.25, 0.7);
        let axis = (out * el.cos() + Vec3::Y * el.sin()).normalize();
        let side = Vec3::Y.cross(out).normalize();
        let s = rng.range(0.38, 0.55);
        let base = out * rng.range(0.0, 0.15) + Vec3::Y * rng.range(0.0, 0.08);
        let start = m.verts.len() as u32;
        emit_card(&mut m, base, axis, side, s, s, &crown, None);
        for v in &mut m.verts[start as usize..] {
            v.nrm = (Vec3::from(v.nrm) * 0.4 + Vec3::Y).normalize().into();
            v.attr[1] *= 0.25;
        }
    }
    m
}

pub fn build_plants() -> Vec<Plant> {
    let plants = vec![
        Plant { name: "fern", mesh: fern(0xFE12_0001), layer: 3, radius: 1.3 },
        Plant { name: "shrub", mesh: shrub(0x5821_0B02, 1.6, 0.55), layer: 0, radius: 1.6 },
        Plant { name: "ground cover", mesh: ground_cover(0x6C0F_E203), layer: 0, radius: 0.7 },
    ];
    for p in &plants {
        println!("plant '{}': {} tris", p.name, p.mesh.tris());
    }
    plants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shape_has_its_own_slot_and_tags_its_vertices() {
        let shapes = build_shapes();
        let mut seen = [false; SLOTS];
        for s in &shapes {
            assert_eq!(form_of(s.species, s.variant), Some(s.form));
            assert!(!seen[s.slot()], "slot {} twice", s.slot());
            seen[s.slot()] = true;
            for m in &s.lods {
                assert!(m.tris() > 50, "slot {}: {} tris", s.slot(), m.tris());
                assert!(m.verts.iter().all(|v| v.attr[3] == s.slot() as f32));
            }
            assert!(s.crown_radii.min_element() > 0.4 && s.density > 0.0);
        }
        // the shapes the slots name are exactly the ones built
        let named = (0..SPECIES).flat_map(|s| (0..VARIANTS).map(move |v| (s, v))).filter(|&(s, v)| form_of(s, v).is_some()).count();
        assert_eq!(named, SHAPES);
    }

    #[test]
    fn open_grown_crowns_start_lower_than_forest_grown_ones() {
        let shapes = build_shapes();
        // the height below which a tenth of the leaf vertices lie, relative to the tree's height
        let low_leaves = |s: &Shape| {
            let mut ys: Vec<f32> = s.lods[0].verts.iter().filter(|v| v.attr[0] == KIND_LEAF).map(|v| v.pos[1]).collect();
            ys.sort_by(f32::total_cmp);
            let top = *ys.last().unwrap();
            ys[ys.len() / 10] / top
        };
        for species in 0..SPECIES {
            let forest = shapes.iter().find(|s| s.species == species && s.form == Form::Forest).unwrap();
            let open = shapes.iter().find(|s| s.species == species && s.form == Form::Open).unwrap();
            println!("{}: lowest leaves at {:.2} of the height open-grown, {:.2} forest-grown", SPECIES_NAMES[species], low_leaves(open), low_leaves(forest));
            assert!(low_leaves(open) + 0.05 < low_leaves(forest));
        }
        assert_eq!(trunk_radius(0, 6), 0.0);
        assert!(trunk_radius(1, 4) > trunk_radius(1, 0));
    }
}
