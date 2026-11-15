//! Scanned foliage (CC0 plant models from Poly Haven, see assets/foliage/README.md). Their
//! texture atlases hold single leaves, fern fronds, conifer sprays, grass blades and flower
//! heads; at startup these are cut out and composed into the atlases the cards use:
//!
//!   t_leaf    Rgba8UnormSrgb  rgb albedo, a coverage
//!   t_leaf_n  Rgba8Unorm      rg normal xy (+x = +u, +y = +v), b roughness, a translucency
//!
//! Layer 0 broadleaf twigs, 1 birch twigs, 2 spruce branches, 3 fern fronds and ground-cover
//! sprigs, 4 the meadow (straightened grass blades, flower heads, seed heads, grass tufts).
//! A cluster is a twig with dozens of scanned leaves placed along it, each turned, foreshortened
//! and tilted; the tilt goes into the normal map, so every leaf of a card catches the light on
//! its own. The leaf layers have "near" cells for the full-detail cards and "far" cells with
//! more, smaller leaves for the bigger cards of the coarser LODs, so leaves keep their size
//! across LODs. Each cell's average albedo is set to what the painted foliage it replaces had
//! (the lighting, exposure and the hull and canopy-shell colours are tuned around that), and
//! every mip level keeps each cell's alpha-tested coverage, so canopies don't thin out with
//! distance.

use glam::Vec2;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wgpu::*;

/// Texels per side of every layer.
pub const SIZE: u32 = 2048;
pub const LAYERS: u32 = 5;

/// A rectangle of one atlas layer, in uv (u0, v0, u1, v1); a card's base is at v1.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub layer: u32,
    pub rect: [f32; 4],
}

const fn cell(layer: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Cell {
    Cell { layer, rect: [x0, y0, x1, y1] }
}

pub const BROADLEAF_NEAR: [Cell; 2] = [cell(0, 0.0, 0.0, 0.5, 0.5), cell(0, 0.5, 0.0, 1.0, 0.5)];
pub const BROADLEAF_FAR: [Cell; 2] = [cell(0, 0.0, 0.5, 0.5, 1.0), cell(0, 0.5, 0.5, 1.0, 1.0)];
pub const BIRCH_NEAR: [Cell; 2] = [cell(1, 0.0, 0.0, 0.5, 0.5), cell(1, 0.5, 0.0, 1.0, 0.5)];
pub const BIRCH_FAR: [Cell; 2] = [cell(1, 0.0, 0.5, 0.5, 1.0), cell(1, 0.5, 0.5, 1.0, 1.0)];
/// Spruce branches: the first two with their sprays spread out (LOD 0), the others denser.
pub const SPRUCE: [Cell; 4] = [cell(2, 0.0, 0.0, 0.5, 0.5), cell(2, 0.5, 0.0, 1.0, 0.5), cell(2, 0.0, 0.5, 0.5, 1.0), cell(2, 0.5, 0.5, 1.0, 1.0)];
pub const FERN: [Cell; 2] = [cell(3, 0.0, 0.0, 0.25, 1.0), cell(3, 0.25, 0.0, 0.5, 1.0)];
pub const SPRIG: [Cell; 2] = [cell(3, 0.5, 0.0, 1.0, 0.5), cell(3, 0.5, 0.5, 1.0, 1.0)];
// Layer 4 (the meadow) is laid out in grass.wgsl: 16 blade strips across u 0..0.5, flower
// heads in 0.125-wide cells in v 0..0.25 (buttercup x3, daisy x3, clover x2), seed heads in
// 0.0625-wide cells in v 0.25..0.5, grass tufts in 0.25 cells below.

// ---------------------------------------------------------------------------------------------
// Sources

struct SrcDef {
    id: &'static str,
    /// has a translucency map
    trans: bool,
}

const SOURCES: [SrcDef; 8] = [
    SrcDef { id: "shrub_01", trans: false },
    SrcDef { id: "shrub_03", trans: false },
    SrcDef { id: "fir_tree_01", trans: false },
    SrcDef { id: "fern_02", trans: false },
    SrcDef { id: "periwinkle_plant", trans: true },
    SrcDef { id: "celandine_01", trans: false },
    SrcDef { id: "grass_medium_02", trans: false },
    SrcDef { id: "grass_medium_01", trans: false },
];
const S1: usize = 0;
const S3: usize = 1;
const FIR: usize = 2;
const FRN: usize = 3;
const PERI: usize = 4;
const CEL: usize = 5;
const G2: usize = 6;
const G1: usize = 7;

/// One piece cut out of a source atlas: its bounding rect, and the points where it attaches
/// (petiole, stalk) and where it ends, all in 0..1 of the source.
#[derive(Clone, Copy)]
struct Sprite {
    src: usize,
    rect: [f32; 4],
    base: [f32; 2],
    tip: [f32; 2],
}

const fn sp(src: usize, rect: [f32; 4], base: [f32; 2], tip: [f32; 2]) -> Sprite {
    Sprite { src, rect, base, tip }
}

// Leaves (petiole at the bottom), found by labelling the alpha maps.
const SERRATE: [Sprite; 6] = [
    sp(S1, [0.439, 0.0, 0.686, 0.402], [0.594, 0.402], [0.57, 0.02]),
    sp(S1, [0.061, 0.08, 0.199, 0.299], [0.14, 0.299], [0.105, 0.082]),
    sp(S1, [0.742, 0.129, 0.881, 0.348], [0.82, 0.348], [0.79, 0.13]),
    sp(S1, [0.678, 0.395, 0.926, 0.768], [0.818, 0.768], [0.86, 0.395]),
    sp(S1, [0.201, 0.41, 0.473, 0.805], [0.365, 0.805], [0.335, 0.41]),
    sp(S1, [0.574, 0.709, 0.666, 0.879], [0.62, 0.879], [0.64, 0.709]),
];
/// The yellowing, spotted one: used now and then.
const SERRATE_OLD: Sprite = sp(S1, [0.018, 0.59, 0.25, 0.928], [0.155, 0.928], [0.13, 0.59]);
const HEART: [Sprite; 6] = [
    sp(S3, [0.055, 0.041, 0.264, 0.295], [0.165, 0.295], [0.145, 0.041]),
    sp(S3, [0.318, 0.121, 0.51, 0.348], [0.42, 0.348], [0.38, 0.121]),
    sp(S3, [0.031, 0.311, 0.311, 0.646], [0.185, 0.646], [0.165, 0.311]),
    sp(S3, [0.479, 0.604, 0.748, 0.932], [0.62, 0.932], [0.588, 0.604]),
    sp(S3, [0.299, 0.627, 0.4, 0.764], [0.355, 0.764], [0.345, 0.627]),
    sp(S3, [0.051, 0.676, 0.242, 0.9], [0.15, 0.9], [0.115, 0.676]),
];
const SPRAYS: [Sprite; 5] = [
    sp(FIR, [0.654, 0.045, 0.936, 0.369], [0.809, 0.367], [0.835, 0.05]),
    sp(FIR, [0.189, 0.049, 0.416, 0.311], [0.306, 0.311], [0.286, 0.05]),
    sp(FIR, [0.309, 0.406, 0.641, 0.775], [0.481, 0.775], [0.5, 0.41]),
    sp(FIR, [0.639, 0.455, 0.953, 0.83], [0.788, 0.83], [0.84, 0.46]),
    sp(FIR, [0.494, 0.264, 0.629, 0.4], [0.54, 0.4], [0.56, 0.265]),
];
const FRONDS: [Sprite; 2] = [
    sp(FRN, [0.088, 0.111, 0.234, 0.949], [0.168, 0.949], [0.18, 0.112]),
    sp(FRN, [0.715, 0.109, 0.836, 0.895], [0.797, 0.895], [0.78, 0.11]),
];
const PERI_LEAVES: [Sprite; 6] = [
    sp(PERI, [0.586, 0.066, 0.75, 0.143], [0.588, 0.085], [0.748, 0.115]),
    sp(PERI, [0.205, 0.119, 0.367, 0.199], [0.207, 0.15], [0.366, 0.17]),
    sp(PERI, [0.572, 0.266, 0.756, 0.361], [0.755, 0.29], [0.575, 0.32]),
    sp(PERI, [0.551, 0.66, 0.744, 0.758], [0.553, 0.71], [0.744, 0.69]),
    sp(PERI, [0.605, 0.799, 0.875, 0.969], [0.61, 0.965], [0.875, 0.8]),
    sp(PERI, [0.318, 0.801, 0.609, 0.969], [0.322, 0.965], [0.607, 0.81]),
];
const BUTTERCUPS: [Sprite; 3] = [
    sp(CEL, [0.283, 0.742, 0.408, 0.871], [0.345, 0.871], [0.345, 0.742]),
    sp(CEL, [0.068, 0.826, 0.188, 0.955], [0.128, 0.955], [0.128, 0.826]),
    sp(CEL, [0.457, 0.816, 0.561, 0.924], [0.509, 0.924], [0.509, 0.816]),
];
const PERI_FLOWERS: [Sprite; 2] = [
    sp(PERI, [0.289, 0.246, 0.525, 0.486], [0.407, 0.486], [0.407, 0.246]),
    sp(PERI, [0.797, 0.6, 0.965, 0.771], [0.881, 0.771], [0.881, 0.6]),
];
const PANICLES: [Sprite; 3] = [
    sp(G1, [0.852, 0.031, 0.879, 0.166], [0.866, 0.166], [0.866, 0.031]),
    sp(G1, [0.938, 0.287, 0.973, 0.416], [0.955, 0.416], [0.955, 0.287]),
    sp(G1, [0.521, 0.311, 0.609, 0.434], [0.565, 0.434], [0.565, 0.311]),
];
const TUFTS: [Sprite; 4] = [
    sp(G1, [0.209, 0.752, 0.471, 0.904], [0.34, 0.904], [0.34, 0.752]),
    sp(G1, [0.598, 0.758, 0.885, 0.861], [0.74, 0.861], [0.74, 0.758]),
    sp(G1, [0.598, 0.863, 0.787, 1.0], [0.69, 1.0], [0.69, 0.863]),
    sp(G1, [0.252, 0.891, 0.475, 1.0], [0.36, 1.0], [0.36, 0.891]),
];
/// Single grass blades, straightened into strips: the sprite and the stretch of it (top, base,
/// in 0..1 of the source) that is still upright. The last three are dry.
const BLADES: [(Sprite, f32, f32); 8] = [
    (sp(G2, [0.033, 0.037, 0.164, 0.963], [0.045, 0.963], [0.1, 0.15]), 0.15, 0.963),
    (sp(G2, [0.875, 0.037, 0.977, 0.998], [0.947, 0.998], [0.95, 0.2]), 0.2, 0.998),
    (sp(G2, [0.672, 0.172, 0.822, 0.893], [0.69, 0.893], [0.69, 0.25]), 0.25, 0.893),
    (sp(G2, [0.15, 0.176, 0.297, 0.971], [0.155, 0.82], [0.17, 0.27]), 0.27, 0.82),
    (sp(G2, [0.361, 0.236, 0.406, 0.725], [0.38, 0.725], [0.38, 0.26]), 0.26, 0.725),
    (sp(G2, [0.809, 0.732, 0.854, 0.998], [0.85, 0.998], [0.83, 0.74]), 0.74, 0.998),
    (sp(G2, [0.451, 0.059, 0.611, 0.762], [0.47, 0.762], [0.48, 0.35]), 0.35, 0.762),
    (sp(G2, [0.795, 0.262, 0.85, 0.619], [0.83, 0.619], [0.83, 0.27]), 0.27, 0.619),
];

/// A source atlas, kept as bytes: sRGB colour, alpha, OpenGL normal, roughness, translucency,
/// and its alpha components labelled at a quarter of the resolution.
struct Source {
    w: usize,
    col: Vec<u8>,
    a: Vec<u8>,
    nrm: Vec<u8>,
    rough: Vec<u8>,
    trans: Option<Vec<u8>>,
    lw: usize,
    labels: Vec<u32>,
    /// a stand-in (the files were missing): every sprite covers the whole image
    synthetic: bool,
}

fn asset_dir() -> Option<PathBuf> {
    let candidates = [PathBuf::from("assets/foliage"), Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/foliage")];
    candidates.into_iter().find(|p| p.is_dir())
}

fn decode(path: &Path, w: Option<u32>) -> Result<image::DynamicImage, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(match w {
        Some(w) if img.width() != w || img.height() != w => img.resize_exact(w, w, image::imageops::FilterType::Triangle),
        _ => img,
    })
}

fn load_source(dir: &Path, def: &SrcDef) -> Result<Source, String> {
    let d = dir.join(def.id);
    let files: Vec<&str> = if def.trans { vec!["diff.jpg", "alpha.jpg", "nor_gl.jpg", "arm.jpg", "translucency.jpg"] } else { vec!["diff.jpg", "alpha.jpg", "nor_gl.jpg", "arm.jpg"] };
    let mut imgs = std::thread::scope(|s| {
        let hs: Vec<_> = files.iter().map(|f| {
            let p = d.join(f);
            s.spawn(move || decode(&p, Some(SIZE)))
        }).collect();
        hs.into_iter().map(|h| h.join().expect("decode thread")).collect::<Result<Vec<_>, _>>()
    })?;
    let trans = if def.trans { Some(imgs.pop().unwrap().into_luma8().into_raw()) } else { None };
    let arm = imgs.pop().unwrap().into_rgb8().into_raw();
    let nrm = imgs.pop().unwrap().into_rgb8().into_raw();
    let a = imgs.pop().unwrap().into_luma8().into_raw();
    let col = imgs.pop().unwrap().into_rgb8().into_raw();
    let w = SIZE as usize;
    let rough = arm.chunks_exact(3).map(|p| p[1]).collect();
    let (lw, labels) = label(&a, w);
    Ok(Source { w, col, a, nrm, rough, trans, lw, labels, synthetic: false })
}

/// A plain leaf shape for a missing source.
fn synthetic_source() -> Source {
    let w = 256usize;
    let mut a = vec![0u8; w * w];
    for y in 0..w {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / w as f32;
            let half = 0.9 * (std::f32::consts::PI * v).sin();
            a[y * w + x] = if u.abs() < half { 255 } else { 0 };
        }
    }
    let lw = w / 4;
    Source {
        w,
        col: [70u8, 100, 40].repeat(w * w),
        a,
        nrm: [128u8, 128, 255].repeat(w * w),
        rough: vec![150; w * w],
        trans: None,
        lw,
        labels: vec![1; lw * lw],
        synthetic: true,
    }
}

/// Connected components of the alpha (8-connected, above half) at a quarter resolution.
fn label(a: &[u8], w: usize) -> (usize, Vec<u32>) {
    let lw = w / 4;
    let mut on = vec![false; lw * lw];
    for y in 0..lw {
        for x in 0..lw {
            let mut s = 0u32;
            for dy in 0..4 {
                for dx in 0..4 {
                    s += a[(y * 4 + dy) * w + x * 4 + dx] as u32;
                }
            }
            on[y * lw + x] = s > 128 * 16;
        }
    }
    let mut lab = vec![0u32; lw * lw];
    let mut next = 0;
    let mut stack = Vec::new();
    for i in 0..lw * lw {
        if !on[i] || lab[i] != 0 {
            continue;
        }
        next += 1;
        lab[i] = next;
        stack.push(i);
        while let Some(j) = stack.pop() {
            let (x, y) = ((j % lw) as i32, (j / lw) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (qx, qy) = (x + dx, y + dy);
                    if qx < 0 || qy < 0 || qx >= lw as i32 || qy >= lw as i32 {
                        continue;
                    }
                    let q = qy as usize * lw + qx as usize;
                    if on[q] && lab[q] == 0 {
                        lab[q] = next;
                        stack.push(q);
                    }
                }
            }
        }
    }
    (lw, lab)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// What a sprite samples to at one point.
#[derive(Clone, Copy, Default)]
struct Texel {
    c: [f32; 3],
    a: f32,
    /// normal in image space (x right, y down)
    n: [f32; 2],
    r: f32,
    t: f32,
}

/// A sprite made ready to stamp: pixel geometry and the mask of its own component.
struct Prepared {
    src: usize,
    rect: [f32; 4],
    base: Vec2,
    axis: Vec2,
    perp: Vec2,
    len: f32,
    /// extent along and across the axis, in units of `len`
    along: [f32; 2],
    across: [f32; 2],
    mask_origin: [usize; 2],
    mask_w: usize,
    mask_h: usize,
    mask: Vec<bool>,
}

fn prepare(s: &Sprite, srcs: &[Source]) -> Prepared {
    let src = &srcs[s.src];
    let wf = src.w as f32;
    let (rect, base, tip) = if src.synthetic { ([0.0, 0.0, 1.0, 1.0], [0.5, 1.0], [0.5, 0.0]) } else { (s.rect, s.base, s.tip) };
    let rect = rect.map(|v| v * wf);
    let base = Vec2::from(base) * wf;
    let tip = Vec2::from(tip) * wf;
    let axis = (tip - base).normalize();
    let perp = Vec2::new(-axis.y, axis.x);
    let len = (tip - base).length();
    let corners = [Vec2::new(rect[0], rect[1]), Vec2::new(rect[2], rect[1]), Vec2::new(rect[2], rect[3]), Vec2::new(rect[0], rect[3])];
    let mut along = [f32::MAX, f32::MIN];
    let mut across = [f32::MAX, f32::MIN];
    for c in corners {
        let d = c - base;
        along = [along[0].min(d.dot(axis) / len), along[1].max(d.dot(axis) / len)];
        across = [across[0].min(d.dot(perp) / len), across[1].max(d.dot(perp) / len)];
    }
    // the component this sprite is: the label covering most of its rect, grown by one cell
    let lw = src.lw;
    let s4 = src.w as f32 / lw as f32;
    let x0 = ((rect[0] / s4).floor() as usize).saturating_sub(1);
    let y0 = ((rect[1] / s4).floor() as usize).saturating_sub(1);
    let x1 = ((rect[2] / s4).ceil() as usize + 1).min(lw);
    let y1 = ((rect[3] / s4).ceil() as usize + 1).min(lw);
    let mut counts: std::collections::HashMap<u32, u32> = Default::default();
    for y in y0..y1 {
        for x in x0..x1 {
            let l = src.labels[y * lw + x];
            if l != 0 {
                *counts.entry(l).or_default() += 1;
            }
        }
    }
    let best = counts.into_iter().max_by_key(|&(l, n)| (n, l)).map(|(l, _)| l).unwrap_or(0);
    let (mw, mh) = (x1 - x0, y1 - y0);
    let mut mask = vec![false; mw * mh];
    for y in 0..mh {
        for x in 0..mw {
            let mut hit = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (qx, qy) = ((x0 + x) as i32 + dx, (y0 + y) as i32 + dy);
                    if qx >= 0 && qy >= 0 && (qx as usize) < lw && (qy as usize) < lw && src.labels[qy as usize * lw + qx as usize] == best {
                        hit = true;
                    }
                }
            }
            mask[y * mw + x] = hit;
        }
    }
    Prepared { src: s.src, rect, base, axis, perp, len, along, across, mask_origin: [x0, y0], mask_w: mw, mask_h: mh, mask }
}

struct Lut([f32; 256]);

impl Lut {
    fn new() -> Lut {
        Lut(std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
    }
}

/// Bilinear sample of a source at pixel position p (texel centres at +0.5).
fn sample(src: &Source, lut: &Lut, p: Vec2) -> Texel {
    let w = src.w;
    let x = (p.x - 0.5).clamp(0.0, (w - 1) as f32);
    let y = (p.y - 0.5).clamp(0.0, (w - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(w - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let taps = [(x0, y0, (1.0 - fx) * (1.0 - fy)), (x1, y0, fx * (1.0 - fy)), (x0, y1, (1.0 - fx) * fy), (x1, y1, fx * fy)];
    let mut t = Texel::default();
    // colour is weighted by alpha, so the background around a leaf doesn't bleed into it
    let mut ca = 0.0;
    for (tx, ty, wgt) in taps {
        let i = ty * w + tx;
        let a = src.a[i] as f32 / 255.0;
        t.a += a * wgt;
        let wa = wgt * a.max(1e-3);
        ca += wa;
        for k in 0..3 {
            t.c[k] += lut.0[src.col[i * 3 + k] as usize] * wa;
        }
        t.n[0] += (src.nrm[i * 3] as f32 / 127.5 - 1.0) * wa;
        // OpenGL normal maps point green up the image: flip to image space (y down)
        t.n[1] -= (src.nrm[i * 3 + 1] as f32 / 127.5 - 1.0) * wa;
        t.r += src.rough[i] as f32 / 255.0 * wa;
        if let Some(tr) = &src.trans {
            t.t += tr[i] as f32 / 255.0 * wa;
        }
    }
    let inv = 1.0 / ca;
    t.c = t.c.map(|v| v * inv);
    t.n = t.n.map(|v| v * inv);
    t.r *= inv;
    t.t *= inv;
    if src.trans.is_none() {
        // no map: leaf blades pass light, stalks and twigs (brown, dull) hardly any
        let s = t.c[0] + t.c[1] + t.c[2] + 1e-4;
        t.t = smoothstep(0.37, 0.46, t.c[1] / s) * 0.85 + 0.15;
    } else {
        t.t = (t.t * 1.6).min(1.0);
    }
    t
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------------------------
// Composition

/// A cell being composed, premultiplied by coverage.
struct Canvas {
    w: usize,
    h: usize,
    c: Vec<[f32; 4]>,
    n: Vec<[f32; 2]>,
    r: Vec<f32>,
    t: Vec<f32>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, h, c: vec![[0.0; 4]; w * h], n: vec![[0.0; 2]; w * h], r: vec![0.0; w * h], t: vec![0.0; w * h] }
    }

    fn over(&mut self, i: usize, t: &Texel, a: f32) {
        let k = 1.0 - a;
        let c = &mut self.c[i];
        for j in 0..3 {
            c[j] = t.c[j] * a + c[j] * k;
        }
        c[3] = a + c[3] * k;
        self.n[i] = [t.n[0] * a + self.n[i][0] * k, t.n[1] * a + self.n[i][1] * k];
        self.r[i] = t.r * a + self.r[i] * k;
        self.t[i] = t.t * a + self.t[i] * k;
    }

    /// A twig: a tapered capsule, rounded across (in cell pixels).
    fn twig(&mut self, a: Vec2, b: Vec2, r0: f32, r1: f32, col: [f32; 3]) {
        let rmax = r0.max(r1) + 1.5;
        let lo = a.min(b) - Vec2::splat(rmax);
        let hi = a.max(b) + Vec2::splat(rmax);
        let ab = b - a;
        let l2 = ab.length_squared().max(1e-6);
        let side = Vec2::new(-ab.y, ab.x).normalize_or_zero();
        for y in (lo.y.floor().max(0.0) as usize)..(hi.y.ceil().min(self.h as f32) as usize) {
            for x in (lo.x.floor().max(0.0) as usize)..(hi.x.ceil().min(self.w as f32) as usize) {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
                let q = p - (a + ab * t);
                let r = r0 + (r1 - r0) * t;
                let cov = (r - q.length() + 0.5).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                let across = (q.dot(side) / r.max(0.5)).clamp(-1.0, 1.0);
                let shade = 0.75 + 0.25 * (1.0 - across * across).sqrt();
                let tex = Texel { c: col.map(|v| v * shade), a: 1.0, n: (side * across * 0.8).into(), r: 0.75, t: 0.1 };
                self.over(y * self.w + x, &tex, cov);
            }
        }
    }
}

/// One leaf (frond, spray, flower...) put down on a cell.
#[derive(Clone, Copy)]
struct Leaf {
    sprite: usize,
    base: Vec2,
    /// unit, from the base towards the tip
    dir: Vec2,
    len: f32,
    /// width compared to the scan: a leaf turned about its own axis looks narrower
    squash: f32,
    mirror: bool,
    shade: f32,
    /// the leaf's tilt, added to its normals
    tilt: Vec2,
}

enum Item {
    Leaf(Leaf),
    Twig { a: Vec2, b: Vec2, r0: f32, r1: f32 },
}

fn stamp(cv: &mut Canvas, srcs: &[Source], prep: &[Prepared], lut: &Lut, l: &Leaf) {
    let p = &prep[l.sprite];
    let src = &srcs[p.src];
    let ad = l.dir;
    let pd = Vec2::new(-ad.y, ad.x);
    let m = if l.mirror { -1.0 } else { 1.0 };
    let scale = l.len / p.len; // cell px per source px
    let to_cell = |al: f32, ac: f32| l.base + ad * al * l.len + pd * ac * l.len * l.squash * m;
    let corners = [to_cell(p.along[0], p.across[0]), to_cell(p.along[1], p.across[0]), to_cell(p.along[1], p.across[1]), to_cell(p.along[0], p.across[1])];
    let lo = corners.iter().fold(Vec2::splat(f32::MAX), |a, c| a.min(*c));
    let hi = corners.iter().fold(Vec2::splat(f32::MIN), |a, c| a.max(*c));
    let s4 = src.w as f32 / src.lw as f32;
    for y in (lo.y.floor().max(0.0) as usize)..(hi.y.ceil().min(cv.h as f32) as usize) {
        for x in (lo.x.floor().max(0.0) as usize)..(hi.x.ceil().min(cv.w as f32) as usize) {
            let d = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - l.base;
            let along = d.dot(ad) / scale;
            let across = d.dot(pd) / (scale * l.squash) * m;
            let s = p.base + p.axis * along + p.perp * across;
            if s.x < p.rect[0] || s.y < p.rect[1] || s.x >= p.rect[2] || s.y >= p.rect[3] {
                continue;
            }
            let mx = (s.x / s4) as usize;
            let my = (s.y / s4) as usize;
            if mx < p.mask_origin[0] || my < p.mask_origin[1] || mx - p.mask_origin[0] >= p.mask_w || my - p.mask_origin[1] >= p.mask_h {
                continue;
            }
            if !p.mask[(my - p.mask_origin[1]) * p.mask_w + mx - p.mask_origin[0]] {
                continue;
            }
            // quick reject on the nearest texel's alpha
            let ni = (s.y as usize).min(src.w - 1) * src.w + (s.x as usize).min(src.w - 1);
            if src.a[ni] == 0 {
                continue;
            }
            let mut t = sample(src, lut, s);
            if t.a <= 0.002 {
                continue;
            }
            // the normal into the cell's frame: along/across the scan's axis, then along/across
            // the leaf as placed
            let n = Vec2::from(t.n);
            let (na, np) = (n.dot(p.axis), n.dot(p.perp) * m);
            let mut nd = ad * na + pd * np + l.tilt;
            let len = nd.length();
            if len > 0.92 {
                nd *= 0.92 / len;
            }
            t.n = nd.into();
            t.c = t.c.map(|v| v * l.shade);
            cv.over(y * cv.w + x, &t, t.a);
        }
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
    fn pick(&mut self, n: usize) -> usize {
        ((self.f() * n as f32) as usize).min(n - 1)
    }
}

fn rot(v: Vec2, a: f32) -> Vec2 {
    Vec2::new(v.x * a.cos() - v.y * a.sin(), v.x * a.sin() + v.y * a.cos())
}

/// Corners of a leaf's footprint, for fitting a layout into its cell.
fn leaf_bounds(l: &Leaf, p: &Prepared) -> [Vec2; 4] {
    let pd = Vec2::new(-l.dir.y, l.dir.x);
    let m = if l.mirror { -1.0 } else { 1.0 };
    let f = |al: f32, ac: f32| l.base + l.dir * al * l.len + pd * ac * l.len * l.squash * m;
    [f(p.along[0], p.across[0]), f(p.along[1], p.across[0]), f(p.along[1], p.across[1]), f(p.along[0], p.across[1])]
}

/// Scale a layout (base of the twig at the origin, growing up = -y) to fill its cell: centred
/// across, standing on the bottom edge, `margin` (a fraction of the cell) clear all round.
fn fit(items: &mut [Item], prep: &[Prepared], w: usize, h: usize, margin: f32) {
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for it in items.iter() {
        match it {
            Item::Leaf(l) => {
                for c in leaf_bounds(l, &prep[l.sprite]) {
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            }
            Item::Twig { a, b, r0, .. } => {
                lo = lo.min(a.min(*b) - Vec2::splat(*r0));
                hi = hi.max(a.max(*b) + Vec2::splat(*r0));
            }
        }
    }
    let (mw, mh) = (w as f32 * (1.0 - 2.0 * margin), h as f32 * (1.0 - 2.0 * margin));
    let size = hi - lo;
    let s = (mw / size.x).min(mh / size.y);
    let off = Vec2::new(w as f32 * 0.5 - (lo.x + hi.x) * 0.5 * s, h as f32 * (1.0 - margin) - hi.y * s);
    for it in items.iter_mut() {
        match it {
            Item::Leaf(l) => {
                l.base = l.base * s + off;
                l.len *= s;
            }
            Item::Twig { a, b, r0, r1 } => {
                *a = *a * s + off;
                *b = *b * s + off;
                *r0 *= s;
                *r1 *= s;
            }
        }
    }
}

fn render(cv: &mut Canvas, items: &[Item], srcs: &[Source], prep: &[Prepared], lut: &Lut, twig_col: [f32; 3]) {
    for it in items {
        match it {
            Item::Twig { a, b, r0, r1 } => cv.twig(*a, *b, *r0, *r1, twig_col),
            Item::Leaf(l) => stamp(cv, srcs, prep, lut, l),
        }
    }
}

/// A leaf at `base`, angled off its twig.
fn leaf_on(rng: &mut Rng, sprite: usize, base: Vec2, dir: Vec2, len: f32) -> Leaf {
    let squash = rng.range(0.5, 1.0);
    // a leaf seen narrower is turned about its axis: its normal leans across, one way or the other
    let turn = (1.0 - squash * squash).sqrt() * if rng.f() < 0.5 { -1.0 } else { 1.0 };
    let pd = Vec2::new(-dir.y, dir.x);
    let tilt = pd * turn * 0.7 + dir * rng.range(-0.35, 0.35) + Vec2::new(rng.range(-0.15, 0.15), rng.range(-0.15, 0.15));
    Leaf { sprite, base, dir, len, squash, mirror: rng.f() < 0.5, shade: 1.0, tilt }
}

struct ClusterStyle {
    /// prepared sprite indices of the leaves, and of the odd old leaf
    leaves: Vec<usize>,
    old: Option<usize>,
    side_twigs: (u32, u32),
    /// leaf length, as a fraction of the main twig
    leaf_len: (f32, f32),
    /// leaves per leaf length along a twig
    density: f32,
    /// how much the side twigs and leaves hang down (birch)
    droop: f32,
}

/// A twig: a main shoot with side shoots, leaves along all of them.
fn cluster(rng: &mut Rng, st: &ClusterStyle) -> Vec<Item> {
    let mut twigs: Vec<(Vec2, Vec2, f32)> = Vec::new();
    let top = Vec2::new(rng.range(-0.12, 0.12), -1.0);
    twigs.push((Vec2::ZERO, top, 0.012));
    let n = st.side_twigs.0 + (rng.f() * (st.side_twigs.1 - st.side_twigs.0 + 1) as f32) as u32;
    let main_dir = top.normalize();
    for i in 0..n {
        let t = 0.08 + 0.8 * (i as f32 + rng.range(0.2, 0.8)) / n as f32;
        let p = top * t;
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let d = rot(main_dir, side * rng.range(0.7, 1.3));
        let len = rng.range(0.45, 0.7) * (1.0 - 0.45 * t);
        let mut end = p + d * len;
        end.y += st.droop * len * 0.9;
        twigs.push((p, end, 0.007));
    }
    let mut leaves = Vec::new();
    for &(a, b, _) in &twigs {
        let dir = (b - a).normalize();
        let tl = (b - a).length();
        let mean_len = (st.leaf_len.0 + st.leaf_len.1) * 0.5;
        let count = ((tl / mean_len) * st.density).ceil() as usize;
        for k in 0..count {
            let t = 0.12 + 0.88 * (k as f32 + rng.range(0.0, 0.6)) / count as f32;
            let p = a + (b - a) * t;
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            let mut d = rot(dir, side * rng.range(0.45, 1.0));
            d = (d + Vec2::new(0.0, st.droop * 0.9)).normalize();
            let sprite = match st.old {
                Some(o) if rng.f() < 0.04 => o,
                _ => st.leaves[rng.pick(st.leaves.len())],
            };
            let len = rng.range(st.leaf_len.0, st.leaf_len.1);
            leaves.push(leaf_on(rng, sprite, p, d, len));
        }
        // a leaf at the tip
        let d = (dir + Vec2::new(0.0, st.droop * 0.5)).normalize();
        let sprite = st.leaves[rng.pick(st.leaves.len())];
        let len = rng.range(st.leaf_len.0, st.leaf_len.1) * 1.1;
        leaves.push(leaf_on(rng, sprite, b, d, len));
    }
    // back to front in random order; the leaves behind are in the shade of those in front
    let mut keyed: Vec<(f32, Leaf)> = leaves.into_iter().map(|l| (rng.f(), l)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut items: Vec<Item> = twigs.iter().map(|&(a, b, r)| Item::Twig { a, b, r0: r, r1: r * 0.5 }).collect();
    for (depth, mut l) in keyed {
        l.shade = (0.68 + 0.32 * depth) * rng.range(0.9, 1.1);
        items.push(Item::Leaf(l));
    }
    items
}

/// A spruce branch: an axis with sprays off both sides, shorter towards the tip.
fn spruce_branch(rng: &mut Rng, sprays: &[usize], dense: bool) -> Vec<Item> {
    let top = Vec2::new(rng.range(-0.05, 0.05), -1.0);
    let dir = top.normalize();
    let mut items = vec![Item::Twig { a: Vec2::ZERO, b: top, r0: 0.014, r1: 0.006 }];
    let mut leaves = Vec::new();
    let step = if dense { 0.04 } else { 0.055 };
    let mut t = 0.04;
    let mut side = 1.0;
    while t < 0.9 {
        let p = top * t;
        let reps = if dense { 2 } else { 1 };
        for r in 0..reps {
            let s = if r == 0 { side } else { -side };
            let d = rot(dir, s * rng.range(0.5, 0.95));
            let len = rng.range(0.5, 0.68) * (1.0 - 0.55 * t) * if r == 1 { 0.75 } else { 1.0 };
            let sprite = sprays[rng.pick(sprays.len())];
            let mut l = leaf_on(rng, sprite, p, d, len);
            l.squash = rng.range(0.85, 1.0);
            leaves.push(l);
        }
        side = -side;
        t += step * rng.range(0.8, 1.2);
    }
    let sprite = sprays[rng.pick(sprays.len())];
    let mut l = leaf_on(rng, sprite, top * 0.82, dir, 0.3);
    l.squash = 1.0;
    leaves.push(l);
    let mut keyed: Vec<(f32, Leaf)> = leaves.into_iter().map(|l| (rng.f(), l)).collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (depth, mut l) in keyed {
        l.shade = (0.72 + 0.28 * depth) * rng.range(0.92, 1.08);
        items.push(Item::Leaf(l));
    }
    items
}

/// A ground-cover sprig: a stem with pairs of opposite leaves.
fn sprig(rng: &mut Rng, leaves: &[usize]) -> Vec<Item> {
    let top = Vec2::new(rng.range(-0.1, 0.1), -1.0);
    let dir = top.normalize();
    let mut items = vec![Item::Twig { a: Vec2::ZERO, b: top, r0: 0.012, r1: 0.007 }];
    let mut ls = Vec::new();
    let pairs = [0.22, 0.42, 0.62, 0.8, 1.0];
    for (i, &t) in pairs.iter().enumerate() {
        let p = top * t;
        let turn = if i % 2 == 0 { 0.0 } else { 0.25 };
        for s in [-1.0f32, 1.0] {
            let a = s * (rng.range(0.75, 1.15) - turn * 0.5) * if i == pairs.len() - 1 { 0.45 } else { 1.0 };
            let d = rot(dir, a);
            let len = rng.range(0.32, 0.42) * (1.0 - 0.3 * t);
            let sprite = leaves[rng.pick(leaves.len())];
            ls.push(leaf_on(rng, sprite, p, d, len));
        }
    }
    for (i, mut l) in ls.into_iter().enumerate() {
        l.shade = rng.range(0.82, 1.08) * if i % 4 < 2 { 0.92 } else { 1.0 };
        items.push(Item::Leaf(l));
    }
    items
}

/// One piece filling its cell, upright.
fn single(sprite: usize) -> Vec<Item> {
    vec![Item::Leaf(Leaf { sprite, base: Vec2::ZERO, dir: Vec2::new(0.0, -1.0), len: 1.0, squash: 1.0, mirror: false, shade: 1.0, tilt: Vec2::ZERO })]
}

/// Grass blades straightened into strips side by side: for each row of a strip, the blade's
/// run of pixels in that row of the scan is stretched across the strip, so the strip is the
/// blade's surface (colour, streaks, dry tip) with its outline taken away. The blade's shape
/// comes from the geometry (grass.wgsl).
fn blade_strips(cv: &mut Canvas, srcs: &[Source], prep: &[Prepared], lut: &Lut, blades: &[(usize, f32, f32)], strips: usize) {
    let sw = cv.w / strips;
    for s in 0..strips {
        let (pi, top, bottom) = blades[s % blades.len()];
        let mirror = s >= blades.len();
        let p = &prep[pi];
        let src = &srcs[p.src];
        let wf = src.w as f32;
        let (x0, x1) = (p.rect[0] as usize, (p.rect[2] as usize).min(src.w - 1));
        let mut centre = p.base.x;
        let mut prev = (centre - 2.0, centre + 2.0);
        let s4 = src.w / src.lw;
        let in_mask = |x: usize, y: usize| {
            let (mx, my) = (x / s4, y / s4);
            mx >= p.mask_origin[0] && my >= p.mask_origin[1] && mx - p.mask_origin[0] < p.mask_w && my - p.mask_origin[1] < p.mask_h && p.mask[(my - p.mask_origin[1]) * p.mask_w + mx - p.mask_origin[0]]
        };
        // from the base up, following the blade
        for row in (0..cv.h).rev() {
            let v = (row as f32 + 0.5) / cv.h as f32;
            let sy = ((top + v * (bottom - top)) * wf).min(wf - 1.0);
            let yi = sy as usize;
            // the run of blade pixels nearest the last row's centre
            let mut best: Option<(f32, f32)> = None;
            let mut x = x0;
            while x <= x1 {
                if src.a[yi * src.w + x] > 127 && in_mask(x, yi) {
                    let start = x;
                    while x <= x1 && src.a[yi * src.w + x] > 127 {
                        x += 1;
                    }
                    let run = (start as f32, x as f32);
                    let c = (run.0 + run.1) * 0.5;
                    if best.is_none_or(|b| (c - centre).abs() < ((b.0 + b.1) * 0.5 - centre).abs()) {
                        best = Some(run);
                    }
                }
                x += 1;
            }
            let first = row + 1 == cv.h;
            let run = match best {
                Some(r) if first || ((r.0 + r.1) * 0.5 - centre).abs() < 40.0 => r,
                _ => prev,
            };
            prev = run;
            centre = (run.0 + run.1) * 0.5;
            for c in 0..sw {
                let mut u = (c as f32 + 0.5) / sw as f32;
                if mirror {
                    u = 1.0 - u;
                }
                // keep clear of the antialiased edge
                let sx = run.0 + 0.5 + (run.1 - run.0 - 1.0).max(0.0) * (0.1 + 0.8 * u);
                let mut t = sample(src, lut, Vec2::new(sx, sy + 0.5));
                if mirror {
                    t.n[0] = -t.n[0];
                }
                t.t = 1.0;
                cv.over(row * cv.w + s * sw + c, &t, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cells, colour, mips

/// What a cell holds and the average albedo to give it.
enum Content {
    Cluster(ClusterStyle, u64),
    Spruce(bool, u64),
    Sprig(u64),
    Single(usize),
    Flower(usize, Recolour),
    Strips,
    Seeds,
}

#[derive(Clone, Copy)]
enum Recolour {
    None,
    /// white petals around the yellow centre
    Daisy,
    /// violet petals
    Clover,
}

struct Job {
    layer: usize,
    /// in texels of the layer
    rect: [usize; 4],
    content: Content,
    /// average linear albedo the cell is scaled to, and how much of the scan's own hue it keeps
    mean: [f32; 3],
    keep_hue: f32,
    margin: f32,
}

// The painted foliage's average albedo, alpha-weighted (linear), which the shading and the far
// LODs were tuned around.
const MEAN_BROADLEAF: [f32; 3] = [0.031, 0.058, 0.0058];
const MEAN_BIRCH: [f32; 3] = [0.055, 0.087, 0.0096];
const MEAN_SPRUCE: [f32; 3] = [0.0216, 0.0565, 0.0132];
const MEAN_FERN: [f32; 3] = [0.039, 0.089, 0.0073];
/// Ground cover: the painted broadleaf it used to share, lighter (fresh, thin leaves).
const MEAN_COVER: [f32; 3] = [0.045, 0.08, 0.012];
/// The blade strips are scaled to this; grass.wgsl recolours them by the meadow's tint over it.
pub const MEAN_BLADE: [f32; 3] = [0.06, 0.1, 0.025];
// flower_color() in globals.wgsl: buttercup, daisy, clover
const MEAN_FLOWERS: [[f32; 3]; 3] = [[0.80, 0.58, 0.05], [0.78, 0.78, 0.72], [0.34, 0.15, 0.48]];
const MEAN_SEEDS: [f32; 3] = [0.16, 0.13, 0.07];
const MEAN_TUFTS: [f32; 3] = [0.06, 0.1, 0.025];

fn jobs(prep_index: &PrepIndex) -> Vec<Job> {
    let s = SIZE as usize;
    let h = s / 2;
    let q = s / 4;
    let mut v = Vec::new();
    let broad = |n: (f32, f32), dens: f32| ClusterStyle {
        leaves: prep_index.serrate.iter().chain(prep_index.heart.iter()).copied().collect(),
        old: Some(prep_index.serrate_old),
        side_twigs: (6, 9),
        leaf_len: n,
        density: dens,
        droop: 0.0,
    };
    let birch = |n: (f32, f32), dens: f32| ClusterStyle {
        leaves: prep_index.serrate.clone(),
        old: Some(prep_index.serrate_old),
        side_twigs: (4, 6),
        leaf_len: n,
        density: dens,
        droop: 0.5,
    };
    let quads = [[0, 0, h, h], [h, 0, s, h], [0, h, h, s], [h, h, s, s]];
    for (i, r) in quads.iter().enumerate() {
        let far = i >= 2;
        let st = if far { broad((0.11, 0.15), 2.6) } else { broad((0.16, 0.23), 2.2) };
        v.push(Job { layer: 0, rect: *r, content: Content::Cluster(st, 11 + i as u64), mean: MEAN_BROADLEAF, keep_hue: 0.25, margin: 0.02 });
        let st = if far { birch((0.085, 0.11), 3.8) } else { birch((0.12, 0.16), 3.3) };
        v.push(Job { layer: 1, rect: *r, content: Content::Cluster(st, 21 + i as u64), mean: MEAN_BIRCH, keep_hue: 0.25, margin: 0.02 });
        v.push(Job { layer: 2, rect: *r, content: Content::Spruce(far, 31 + i as u64), mean: MEAN_SPRUCE, keep_hue: 0.25, margin: 0.02 });
    }
    for (i, &f) in prep_index.fronds.iter().enumerate() {
        v.push(Job { layer: 3, rect: [i * q, 0, (i + 1) * q, s], content: Content::Single(f), mean: MEAN_FERN, keep_hue: 0.5, margin: 0.01 });
    }
    for i in 0..2 {
        v.push(Job { layer: 3, rect: [h, i * h, s, (i + 1) * h], content: Content::Sprig(41 + i as u64), mean: MEAN_COVER, keep_hue: 0.6, margin: 0.02 });
    }
    // the meadow
    v.push(Job { layer: 4, rect: [0, 0, h, s], content: Content::Strips, mean: MEAN_BLADE, keep_hue: 0.5, margin: 0.0 });
    let fc = s / 8;
    let flowers: [(usize, Recolour, usize); 8] = [
        (prep_index.buttercups[0], Recolour::None, 0),
        (prep_index.buttercups[1], Recolour::None, 0),
        (prep_index.buttercups[2], Recolour::None, 0),
        (prep_index.buttercups[0], Recolour::Daisy, 1),
        (prep_index.buttercups[1], Recolour::Daisy, 1),
        (prep_index.buttercups[2], Recolour::Daisy, 1),
        (prep_index.peri_flowers[0], Recolour::Clover, 2),
        (prep_index.peri_flowers[1], Recolour::Clover, 2),
    ];
    for (i, &(f, rc, kind)) in flowers.iter().enumerate() {
        let (x, y) = (h + (i % 4) * fc, (i / 4) * fc);
        v.push(Job { layer: 4, rect: [x, y, x + fc, y + fc], content: Content::Flower(f, rc), mean: MEAN_FLOWERS[kind], keep_hue: 0.0, margin: 0.04 });
    }
    v.push(Job { layer: 4, rect: [h, q, s, h], content: Content::Seeds, mean: MEAN_SEEDS, keep_hue: 0.6, margin: 0.0 });
    for (i, &t) in prep_index.tufts.iter().enumerate() {
        let (x, y) = (h + (i % 2) * q, h + (i / 2) * q);
        v.push(Job { layer: 4, rect: [x, y, x + q, y + q], content: Content::Single(t), mean: MEAN_TUFTS, keep_hue: 0.5, margin: 0.02 });
    }
    v
}

/// Where each sprite list landed among the prepared sprites.
struct PrepIndex {
    serrate: Vec<usize>,
    serrate_old: usize,
    heart: Vec<usize>,
    sprays: Vec<usize>,
    fronds: Vec<usize>,
    peri_leaves: Vec<usize>,
    buttercups: Vec<usize>,
    peri_flowers: Vec<usize>,
    panicles: Vec<usize>,
    tufts: Vec<usize>,
    blades: Vec<(usize, f32, f32)>,
}

fn prepare_all(srcs: &[Source]) -> (Vec<Prepared>, PrepIndex) {
    let mut prep = Vec::new();
    let mut add = |list: &[Sprite]| -> Vec<usize> {
        list.iter()
            .map(|s| {
                prep.push(prepare(s, srcs));
                prep.len() - 1
            })
            .collect()
    };
    let serrate = add(&SERRATE);
    let serrate_old = add(&[SERRATE_OLD])[0];
    let heart = add(&HEART);
    let sprays = add(&SPRAYS);
    let fronds = add(&FRONDS);
    let peri_leaves = add(&PERI_LEAVES);
    let buttercups = add(&BUTTERCUPS);
    let peri_flowers = add(&PERI_FLOWERS);
    let panicles = add(&PANICLES);
    let tufts = add(&TUFTS);
    let blade_sprites: Vec<Sprite> = BLADES.iter().map(|b| b.0).collect();
    let bi = add(&blade_sprites);
    let blades = bi.iter().zip(BLADES.iter()).map(|(&i, b)| (i, b.1, b.2)).collect();
    (prep, PrepIndex { serrate, serrate_old, heart, sprays, fronds, peri_leaves, buttercups, peri_flowers, panicles, tufts, blades })
}

fn compose(job: &Job, srcs: &[Source], prep: &[Prepared], idx: &PrepIndex, lut: &Lut) -> Canvas {
    let (w, h) = (job.rect[2] - job.rect[0], job.rect[3] - job.rect[1]);
    let mut cv = Canvas::new(w, h);
    let twig_col = [0.1, 0.075, 0.05];
    match &job.content {
        Content::Cluster(st, seed) => {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut items = cluster(&mut rng, st);
            fit(&mut items, prep, w, h, job.margin);
            render(&mut cv, &items, srcs, prep, lut, twig_col);
        }
        Content::Spruce(dense, seed) => {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut items = spruce_branch(&mut rng, &idx.sprays, *dense);
            fit(&mut items, prep, w, h, job.margin);
            render(&mut cv, &items, srcs, prep, lut, [0.08, 0.055, 0.035]);
        }
        Content::Sprig(seed) => {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut items = sprig(&mut rng, &idx.peri_leaves);
            fit(&mut items, prep, w, h, job.margin);
            render(&mut cv, &items, srcs, prep, lut, [0.06, 0.07, 0.03]);
        }
        Content::Single(i) | Content::Flower(i, _) => {
            let mut items = single(*i);
            fit(&mut items, prep, w, h, job.margin);
            render(&mut cv, &items, srcs, prep, lut, twig_col);
            if let Content::Flower(_, rc) = &job.content {
                recolour(&mut cv, *rc);
            }
        }
        Content::Strips => blade_strips(&mut cv, srcs, prep, lut, &idx.blades, 16),
        Content::Seeds => {
            // eight seed heads side by side: the three scans, mirrored and in other ripeness
            let cw = w / 8;
            for k in 0..8 {
                let mut sub = Canvas::new(cw, h);
                let mut items = single(idx.panicles[k % idx.panicles.len()]);
                if let Item::Leaf(l) = &mut items[0] {
                    l.mirror = k >= 4;
                }
                fit(&mut items, prep, cw, h, 0.02);
                render(&mut sub, &items, srcs, prep, lut, twig_col);
                let tint = [[1.0, 1.0, 1.0], [1.15, 1.0, 0.75], [0.9, 0.8, 0.95], [1.1, 1.05, 0.9]][k % 4];
                for y in 0..h {
                    for x in 0..cw {
                        let i = y * cw + x;
                        let o = y * w + k * cw + x;
                        cv.c[o] = [sub.c[i][0] * tint[0], sub.c[i][1] * tint[1], sub.c[i][2] * tint[2], sub.c[i][3]];
                        cv.n[o] = sub.n[i];
                        cv.r[o] = sub.r[i];
                        cv.t[o] = sub.t[i];
                    }
                }
            }
        }
    }
    cv
}

/// Daisies and clover from the buttercup and periwinkle scans: the petals keep their shading
/// and take the other colour; the centre of a daisy stays yellow.
fn recolour(cv: &mut Canvas, rc: Recolour) {
    let (petal, centre_r) = match rc {
        Recolour::None => return,
        Recolour::Daisy => ([0.8f32, 0.8, 0.76], 0.15),
        Recolour::Clover => ([0.38, 0.16, 0.55], 0.05),
    };
    let (w, h) = (cv.w as f32, cv.h as f32);
    for y in 0..cv.h {
        for x in 0..cv.w {
            let i = y * cv.w + x;
            let a = cv.c[i][3];
            if a <= 0.0 {
                continue;
            }
            let c = [cv.c[i][0] / a, cv.c[i][1] / a, cv.c[i][2] / a];
            let r = Vec2::new((x as f32 + 0.5) / w - 0.5, (y as f32 + 0.5) / h - 0.5).length();
            let keep = smoothstep(centre_r + 0.02, centre_r - 0.02, r);
            let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            // a buttercup's petals are about 0.45 in luminance
            let k = (lum / 0.45).min(1.4);
            for j in 0..3 {
                let p = petal[j] * k;
                cv.c[i][j] = (c[j] * keep + p * (1.0 - keep)) * a;
            }
        }
    }
}

/// Unpremultiply, set the average albedo, and give the empty texels the cell's mean colour (so
/// filtering doesn't darken the edges) and a flat normal.
fn finish_colour(cv: &mut Canvas, job: &Job) -> f32 {
    let mut sum = [0.0f64; 3];
    let mut wsum = 0.0f64;
    let (mut rs, mut ts) = (0.0f64, 0.0f64);
    for i in 0..cv.c.len() {
        let a = cv.c[i][3];
        if a > 1e-4 {
            for k in 0..3 {
                cv.c[i][k] /= a;
            }
            cv.n[i] = [cv.n[i][0] / a, cv.n[i][1] / a];
            cv.r[i] /= a;
            cv.t[i] /= a;
        }
        if a >= 0.5 {
            for k in 0..3 {
                sum[k] += cv.c[i][k] as f64;
            }
            rs += cv.r[i] as f64;
            ts += cv.t[i] as f64;
            wsum += 1.0;
        }
    }
    let wsum = wsum.max(1.0);
    let src_mean: [f32; 3] = std::array::from_fn(|k| (sum[k] / wsum).max(1e-5) as f32);
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let (lt, ls) = (lum(job.mean), lum(src_mean));
    let target: [f32; 3] = std::array::from_fn(|k| job.mean[k] * (1.0 - job.keep_hue) + src_mean[k] * lt / ls * job.keep_hue);
    let scale: [f32; 3] = std::array::from_fn(|k| target[k] / src_mean[k]);
    let mean_r = (rs / wsum) as f32;
    let mean_t = (ts / wsum) as f32;
    let mut covered = 0usize;
    for i in 0..cv.c.len() {
        let a = cv.c[i][3];
        for k in 0..3 {
            cv.c[i][k] *= scale[k];
        }
        if a < 0.5 {
            let f = a / 0.5;
            for k in 0..3 {
                cv.c[i][k] = cv.c[i][k] * f + target[k] * (1.0 - f);
            }
            cv.n[i] = [cv.n[i][0] * f, cv.n[i][1] * f];
            cv.r[i] = cv.r[i] * f + mean_r * (1.0 - f);
            cv.t[i] = cv.t[i] * f + mean_t * (1.0 - f);
        } else {
            covered += 1;
        }
    }
    covered as f32 / cv.c.len() as f32
}

fn coverage(a: &[f32], scale: f32) -> f32 {
    a.iter().filter(|&&v| v * scale >= 0.5).count() as f32 / a.len() as f32
}

/// The cell's mip chain, packed: (albedo + alpha, normal + roughness + translucency) per level,
/// down to where the cell is a single texel thick.
fn cell_mips(mut cv: Canvas, target: f32, srgb: &[u8]) -> Vec<(usize, usize, Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut nlen: Vec<f32> = vec![1.0; cv.w * cv.h];
    loop {
        let mut a8 = Vec::with_capacity(cv.w * cv.h * 4);
        let mut b8 = Vec::with_capacity(cv.w * cv.h * 4);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let s = |v: f32| srgb[(v.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize];
        for i in 0..cv.c.len() {
            let c = cv.c[i];
            a8.extend_from_slice(&[s(c[0]), s(c[1]), s(c[2]), q(c[3])]);
            // Toksvig: the shorter the averaged normal, the rougher
            let l = nlen[i].clamp(1e-3, 1.0);
            let alpha = cv.r[i] * cv.r[i];
            let rough = (alpha * alpha + (1.0 - l) / l).min(1.0).sqrt().sqrt();
            b8.extend_from_slice(&[q(cv.n[i][0] * 0.5 + 0.5), q(cv.n[i][1] * 0.5 + 0.5), q(rough), q(cv.t[i])]);
        }
        out.push((cv.w, cv.h, a8, b8));
        if cv.w < 2 || cv.h < 2 {
            break;
        }
        let (w, h) = (cv.w / 2, cv.h / 2);
        let mut next = Canvas::new(w, h);
        let mut next_len = vec![1.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                let mut n3 = [0.0f32; 3];
                let (mut r, mut t, mut ws, mut nw) = (0.0, 0.0, 0.0, 0.0);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = (y * 2 + dy) * cv.w + x * 2 + dx;
                    let p = cv.c[i];
                    let wgt = p[3].max(1e-3);
                    for k in 0..3 {
                        acc[k] += p[k] * wgt;
                    }
                    acc[3] += p[3];
                    let n = cv.n[i];
                    let nz = (1.0 - n[0] * n[0] - n[1] * n[1]).max(0.0).sqrt();
                    let l = nlen[i];
                    n3[0] += n[0] * l * wgt;
                    n3[1] += n[1] * l * wgt;
                    n3[2] += nz * l * wgt;
                    r += cv.r[i] * wgt;
                    t += cv.t[i] * wgt;
                    ws += wgt;
                    nw += wgt;
                }
                let o = y * w + x;
                next.c[o] = [acc[0] / ws, acc[1] / ws, acc[2] / ws, acc[3] / 4.0];
                let ln = (n3[0] * n3[0] + n3[1] * n3[1] + n3[2] * n3[2]).sqrt() / nw;
                let ln = ln.max(1e-4);
                next.n[o] = [n3[0] / nw / ln, n3[1] / nw / ln];
                next_len[o] = ln;
                next.r[o] = r / ws;
                next.t[o] = t / ws;
            }
        }
        // rescale alpha so the alpha-tested coverage matches the top level
        if target > 0.0 && target < 1.0 {
            let a: Vec<f32> = next.c.iter().map(|c| c[3]).collect();
            let (mut lo, mut hi) = (0.5f32, 8.0f32);
            for _ in 0..14 {
                let mid = 0.5 * (lo + hi);
                if coverage(&a, mid) < target { lo = mid } else { hi = mid }
            }
            let sc = 0.5 * (lo + hi);
            for c in next.c.iter_mut() {
                c[3] = (c[3] * sc).min(1.0);
            }
        }
        cv = next;
        nlen = next_len;
    }
    out
}

/// The composed atlas, ready to upload: per layer, per mip level, the two textures' bytes.
pub struct Composed {
    levels: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    pub secs: f32,
}

/// Decode the scans and compose the atlases (CPU only, so it can run beside other startup work).
pub fn compose_all() -> Composed {
    let t0 = Instant::now();
    let dir = asset_dir();
    if dir.is_none() {
        eprintln!("foliage: assets/foliage not found, using plain leaf shapes");
    }
    let srcs: Vec<Source> = std::thread::scope(|s| {
        let hs: Vec<_> = SOURCES
            .iter()
            .map(|def| {
                let dir = dir.clone();
                s.spawn(move || match dir.as_deref().map(|d| load_source(d, def)) {
                    Some(Ok(src)) => src,
                    Some(Err(e)) => {
                        eprintln!("foliage: {e}; using a plain leaf shape for {}", def.id);
                        synthetic_source()
                    }
                    None => synthetic_source(),
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("foliage decode")).collect()
    });
    let (prep, idx) = prepare_all(&srcs);
    let lut = Lut::new();
    let srgb: Vec<u8> = (0..4096).map(|i| (linear_to_srgb(i as f32 / 4095.0) * 255.0 + 0.5) as u8).collect();
    let jobs = jobs(&idx);
    // a small pool: each worker takes the next cell
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(jobs.len());
    let results: Vec<(usize, Vec<(usize, usize, Vec<u8>, Vec<u8>)>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= jobs.len() {
                            break;
                        }
                        let mut cv = compose(&jobs[j], &srcs, &prep, &idx, &lut);
                        let cov = finish_colour(&mut cv, &jobs[j]);
                        done.push((j, cell_mips(cv, cov, &srgb)));
                    }
                    done
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("foliage worker")).collect()
    });
    drop(srcs);

    // assemble the layers level by level; once a cell is under a texel thick, the rest of the
    // chain is a plain box filter of the whole layer
    let mut levels: Vec<Vec<(Vec<u8>, Vec<u8>)>> = Vec::new();
    let mut cells: Vec<Option<&Vec<(usize, usize, Vec<u8>, Vec<u8>)>>> = vec![None; jobs.len()];
    for (j, m) in &results {
        cells[*j] = Some(m);
    }
    let mips = SIZE.ilog2() as usize + 1;
    for layer in 0..LAYERS as usize {
        let mut chain: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for m in 0..mips {
            let size = (SIZE as usize >> m).max(1);
            let in_layer: Vec<usize> = (0..jobs.len()).filter(|&j| jobs[j].layer == layer).collect();
            let all_there = in_layer.iter().all(|&j| cells[j].unwrap().len() > m && (jobs[j].rect[2] - jobs[j].rect[0]) >> m >= 1 && (jobs[j].rect[3] - jobs[j].rect[1]) >> m >= 1);
            if all_there {
                let mut a = vec![0u8; size * size * 4];
                let mut b = vec![0u8; size * size * 4];
                for &j in &in_layer {
                    let (cw, ch, ca, cb) = &cells[j].unwrap()[m];
                    let (x0, y0) = (jobs[j].rect[0] >> m, jobs[j].rect[1] >> m);
                    for y in 0..*ch {
                        let o = ((y0 + y) * size + x0) * 4;
                        a[o..o + cw * 4].copy_from_slice(&ca[y * cw * 4..(y + 1) * cw * 4]);
                        b[o..o + cw * 4].copy_from_slice(&cb[y * cw * 4..(y + 1) * cw * 4]);
                    }
                }
                chain.push((a, b));
            } else {
                let (pa, pb) = chain.last().unwrap();
                let ps = size * 2;
                let mut a = vec![0u8; size * size * 4];
                let mut b = vec![0u8; size * size * 4];
                for y in 0..size {
                    for x in 0..size {
                        let mut sa = [0.0f32; 4];
                        let mut sb = [0u32; 4];
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let i = ((y * 2 + dy) * ps + x * 2 + dx) * 4;
                            for k in 0..3 {
                                sa[k] += lut.0[pa[i + k] as usize] * 0.25;
                            }
                            sa[3] += pa[i + 3] as f32 * 0.25;
                            for k in 0..4 {
                                sb[k] += pb[i + k] as u32;
                            }
                        }
                        let o = (y * size + x) * 4;
                        for k in 0..3 {
                            a[o + k] = srgb[(sa[k].clamp(0.0, 1.0) * 4095.0 + 0.5) as usize];
                        }
                        a[o + 3] = (sa[3] + 0.5) as u8;
                        for k in 0..4 {
                            b[o + k] = ((sb[k] + 2) / 4) as u8;
                        }
                    }
                }
                chain.push((a, b));
            }
        }
        levels.push(chain);
    }
    Composed { levels, secs: t0.elapsed().as_secs_f32() }
}

pub struct Foliage {
    _a: Texture,
    _b: Texture,
    pub view_a: TextureView,
    pub view_b: TextureView,
}

impl Foliage {
    pub fn upload(device: &Device, queue: &Queue, c: &Composed) -> Foliage {
        let mips = SIZE.ilog2() + 1;
        let make = |label: &str, format: TextureFormat, second: bool| {
            let tex = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: LAYERS },
                mip_level_count: mips,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (layer, chain) in c.levels.iter().enumerate() {
                for (mip, (a, b)) in chain.iter().enumerate() {
                    let w = (SIZE >> mip).max(1);
                    queue.write_texture(
                        TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: Origin3d { x: 0, y: 0, z: layer as u32 }, aspect: TextureAspect::All },
                        if second { b } else { a },
                        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(w) },
                        Extent3d { width: w, height: w, depth_or_array_layers: 1 },
                    );
                }
            }
            let view = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
            (tex, view)
        };
        let (a, view_a) = make("foliage albedo", TextureFormat::Rgba8UnormSrgb, false);
        let (b, view_b) = make("foliage normal", TextureFormat::Rgba8Unorm, true);
        println!("foliage: {LAYERS} atlas layers at {SIZE}² composed from scans in {:.2}s", c.secs);
        Foliage { _a: a, _b: b, view_a, view_b }
    }
}

/// Write each layer (over grey) and its normal map as PNGs at half size (debugging aid):
/// `<path>_<layer>.png` and `<path>_<layer>_n.png`.
pub fn debug_png(path: &str) {
    let c = compose_all();
    let path = path.trim_end_matches(".png");
    let lut = Lut::new();
    for (layer, chain) in c.levels.iter().enumerate() {
        let (a, b) = &chain[1];
        let w = (SIZE / 2) as usize;
        let mut out = vec![0u8; w * w * 4];
        let mut outn = vec![0u8; w * w * 4];
        for i in 0..w * w {
            let al = a[i * 4 + 3] as f32 / 255.0;
            for k in 0..3 {
                let bg = [0.55f32, 0.6, 0.65][k].powf(2.2);
                out[i * 4 + k] = (linear_to_srgb(lut.0[a[i * 4 + k] as usize] * al + bg * (1.0 - al)) * 255.0 + 0.5) as u8;
                outn[i * 4 + k] = b[i * 4 + k];
            }
            out[i * 4 + 3] = 255;
            outn[i * 4 + 3] = 255;
        }
        crate::write_png(&format!("{path}_{layer}.png"), w as u32, w as u32, &out);
        crate::write_png(&format!("{path}_{layer}_n.png"), w as u32, w as u32, &outn);
    }
    println!("foliage: composed in {:.2}s", c.secs);
}
