//! Procedurally painted foliage textures (one array layer per species):
//! 0 = broadleaf cluster, 1 = birch twig, 2 = spruce sprig, 3 = fern frond (undergrowth).
//! Mip levels keep their alpha-tested coverage so distant canopies don't thin out.

use glam::Vec2;
use wgpu::*;

pub const SIZE: u32 = 512;
pub const LAYERS: u32 = 4;

struct Img {
    w: usize,
    px: Vec<[f32; 4]>,
}

impl Img {
    fn new(w: usize) -> Img {
        Img { w, px: vec![[0.0; 4]; w * w] }
    }

    fn blend(&mut self, x: usize, y: usize, c: [f32; 3], a: f32) {
        if a <= 0.0 {
            return;
        }
        let p = &mut self.px[y * self.w + x];
        for k in 0..3 {
            p[k] = c[k] * a + p[k] * (1.0 - a);
        }
        p[3] = a + p[3] * (1.0 - a);
    }

    fn bbox(&self, lo: Vec2, hi: Vec2) -> (usize, usize, usize, usize) {
        let max = self.w as f32 - 1.0;
        (
            lo.x.floor().clamp(0.0, max) as usize,
            lo.y.floor().clamp(0.0, max) as usize,
            hi.x.ceil().clamp(0.0, max) as usize,
            hi.y.ceil().clamp(0.0, max) as usize,
        )
    }

    fn capsule(&mut self, a: Vec2, b: Vec2, r: f32, c: [f32; 3]) {
        let (x0, y0, x1, y1) = self.bbox(a.min(b) - Vec2::splat(r + 1.0), a.max(b) + Vec2::splat(r + 1.0));
        let ab = b - a;
        let l2 = ab.length_squared().max(1e-6);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
                let d = (p - (a + ab * t)).length();
                let cov = (r - d + 0.5).clamp(0.0, 1.0);
                self.blend(x, y, c, cov);
            }
        }
    }

    /// A pointed leaf blade from `base` along `dir` (unit), with a lighter midrib.
    fn leaf(&mut self, base: Vec2, dir: Vec2, len: f32, width: f32, c: [f32; 3]) {
        let perp = Vec2::new(-dir.y, dir.x);
        let tip = base + dir * len;
        let pad = Vec2::splat(width * len + 2.0);
        let (x0, y0, x1, y1) = self.bbox(base.min(tip) - pad, base.max(tip) + pad);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - base;
                let u = p.dot(dir) / len;
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let v = p.dot(perp);
                let half = width * len * (std::f32::consts::PI * u).sin().powf(0.75) * (1.0 - 0.25 * u);
                let cov = (half - v.abs() + 0.5).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                let rel = (v.abs() / half.max(1e-3)).clamp(0.0, 1.0);
                let vein = (-(v / 1.2).powi(2)).exp() * (1.0 - u);
                let shade = 0.82 + 0.25 * (1.0 - rel) - 0.12 * rel.powi(4) + 0.12 * (u - 0.5) * (v.signum());
                let col = [
                    c[0] * shade + vein * 0.05,
                    c[1] * shade + vein * 0.07,
                    c[2] * shade + vein * 0.02,
                ];
                self.blend(x, y, col, cov);
            }
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
}

fn lin(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| v.powf(2.2))
}

fn rot(v: Vec2, a: f32) -> Vec2 {
    Vec2::new(v.x * a.cos() - v.y * a.sin(), v.x * a.sin() + v.y * a.cos())
}

fn paint_broadleaf(img: &mut Img, rng: &mut Rng, small: bool) {
    let s = img.w as f32;
    let stem_col = lin([0.26, 0.22, 0.12]);
    let (base_col, n_twigs, leaf_len, per_twig, droop) = if small {
        ([0.30, 0.38, 0.13], 7, 0.085, 6, 0.6)
    } else {
        ([0.22, 0.30, 0.10], 6, 0.17, 4, 0.0)
    };
    let stem_a = Vec2::new(0.5 * s, 0.99 * s);
    let stem_b = Vec2::new(0.5 * s + rng.range(-0.06, 0.06) * s, 0.12 * s);
    // leaves drawn back-to-front: earlier ones darker to fake depth
    let mut leaves: Vec<(Vec2, Vec2, f32, f32)> = Vec::new();
    let mut twigs: Vec<(Vec2, Vec2)> = vec![(stem_a, stem_b)];
    for i in 0..n_twigs {
        let t = 0.2 + 0.75 * (i as f32 + rng.f() * 0.5) / n_twigs as f32;
        let p = stem_a + (stem_b - stem_a) * t;
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let up = (stem_b - stem_a).normalize();
        let ang = side * rng.range(0.5, 1.0) + side * droop * 0.5;
        let d = rot(up, ang);
        let len = rng.range(0.22, 0.34) * s * (1.0 - 0.4 * t);
        let mut end = p + d * len;
        end.y += droop * len * 0.5;
        twigs.push((p, end));
    }
    for &(a, b) in &twigs {
        let dir = (b - a).normalize();
        for k in 0..per_twig {
            let t = 0.3 + 0.7 * (k as f32 + 0.5) / per_twig as f32;
            let p = a + (b - a) * t;
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            let d = rot(dir, side * rng.range(0.35, 0.9));
            let d = (d + Vec2::new(0.0, droop * 0.6)).normalize();
            leaves.push((p, d, leaf_len * s * rng.range(0.8, 1.15), rng.f()));
        }
        // tip leaf
        leaves.push((b, dir, leaf_len * s * rng.range(0.9, 1.2), rng.f()));
    }
    for &(a, b) in &twigs {
        img.capsule(a, b, if a == stem_a { 3.5 } else { 2.0 }, stem_col);
    }
    let n = leaves.len();
    for (i, (p, d, len, r)) in leaves.into_iter().enumerate() {
        let depth = 0.65 + 0.35 * (i as f32 / n as f32);
        let var = 0.85 + 0.3 * r;
        let c = [base_col[0] * var * (0.95 + 0.1 * r), base_col[1] * var, base_col[2] * var];
        let c = lin(c).map(|v| v * depth);
        img.leaf(p, d, len, if small { 0.32 } else { 0.38 }, c);
    }
}

fn paint_spruce(img: &mut Img, rng: &mut Rng) {
    let s = img.w as f32;
    let stem_a = Vec2::new(0.5 * s, 0.99 * s);
    let stem_b = Vec2::new(0.5 * s, 0.06 * s);
    let twig_col = lin([0.30, 0.20, 0.12]);
    let mut shoots: Vec<(Vec2, Vec2)> = vec![(stem_a, stem_b)];
    for i in 0..9 {
        let t = 0.12 + 0.8 * i as f32 / 9.0;
        let p = stem_a + (stem_b - stem_a) * t;
        for side in [-1.0f32, 1.0] {
            let ang = side * rng.range(0.75, 1.0);
            let d = rot(Vec2::new(0.0, -1.0), ang);
            let len = (0.36 * (1.0 - t * 0.75) + rng.range(-0.03, 0.03)) * s;
            shoots.push((p, p + d * len));
        }
    }
    for &(a, b) in &shoots {
        img.capsule(a, b, 1.6, twig_col);
    }
    for (si, &(a, b)) in shoots.iter().enumerate() {
        let dir = (b - a).normalize();
        let len = (b - a).length();
        let steps = (len / 3.2) as i32;
        for k in 0..steps {
            let t = k as f32 / steps as f32;
            let p = a + (b - a) * t;
            for side in [-1.0f32, 1.0] {
                let nd = rot(dir, side * rng.range(0.7, 1.1));
                let nl = rng.range(12.0, 19.0) * if si == 0 { 1.2 } else { 1.0 } * (1.0 - 0.3 * t);
                let r = rng.f();
                let tipness = t;
                let c = lin([
                    0.10 + 0.08 * r + 0.10 * tipness,
                    0.20 + 0.08 * r + 0.12 * tipness,
                    0.11 + 0.05 * r + 0.04 * tipness,
                ])
                .map(|v| v * (0.7 + 0.3 * (k as f32 / steps as f32)));
                img.capsule(p, p + nd * nl, 1.3, c);
            }
        }
    }
}

/// A fern frond, base at the bottom edge and tip at the top: a rachis with pairs of pinnae,
/// longest a third of the way up, each cut into rounded pinnules.
fn paint_fern(img: &mut Img, rng: &mut Rng) {
    let s = img.w as f32;
    let base = Vec2::new(0.5 * s, 0.995 * s);
    let tip = Vec2::new(0.5 * s, 0.02 * s);
    img.capsule(base, tip, 2.0, lin([0.28, 0.32, 0.12]));
    let pairs = 21;
    for i in 0..pairs {
        let t = 0.05 + 0.93 * (i as f32 + rng.f() * 0.25) / pairs as f32;
        let p = base + (tip - base) * t;
        // pinna length: grows to a third of the way up, then tapers to the tip
        let shape = if t < 0.33 { 0.6 + 0.4 * (t / 0.33) } else { 1.0 - (t - 0.33) / 0.67 * 0.92 };
        let len = 0.43 * s * shape * rng.range(0.92, 1.05);
        let light = 0.8 + 0.35 * t;
        for side in [-1.0f32, 1.0] {
            // pinnae sweep towards the tip
            let d = rot(Vec2::new(0.0, -1.0), side * rng.range(1.15, 1.3));
            let end = p + d * len;
            img.capsule(p, end, 1.0, lin([0.22, 0.30, 0.10]));
            // pinnules: small rounded lobes in pairs along the pinna, pointing forwards
            let lobes = ((len / s) * 30.0).max(2.0) as i32;
            for k in 0..lobes {
                let u = (k as f32 + 0.5) / lobes as f32;
                let q = p + (end - p) * u;
                let lobe_len = (0.03 * s * (1.0 - 0.55 * u) * shape.sqrt()).max(3.0);
                let r = rng.f();
                let c = lin([0.20 + 0.05 * r, 0.33 + 0.05 * r, 0.08 + 0.02 * r]).map(|v| v * light);
                for lside in [-1.0f32, 1.0] {
                    let ld = rot(d, lside * 0.9);
                    img.leaf(q, ld, lobe_len, 0.45, c);
                }
            }
        }
    }
}

fn to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

fn coverage(px: &[[f32; 4]], scale: f32) -> f32 {
    px.iter().filter(|p| p[3] * scale >= 0.5).count() as f32 / px.len() as f32
}

/// Mip chain with alpha-coverage preservation; returns RGBA8 (sRGB colour) per level.
fn build_mips(mut img: Img) -> Vec<Vec<u8>> {
    // fill transparent texels with the mean leaf colour so filtering doesn't darken edges
    let mut sum = [0.0f32; 3];
    let mut wsum = 0.0;
    for p in &img.px {
        for k in 0..3 {
            sum[k] += p[k] * p[3];
        }
        wsum += p[3];
    }
    let mean = sum.map(|v| v / wsum.max(1e-6));
    for p in img.px.iter_mut() {
        if p[3] < 0.5 {
            let a = p[3] / 0.5;
            for k in 0..3 {
                p[k] = p[k] * a + mean[k] * (1.0 - a);
            }
        }
    }
    let target = coverage(&img.px, 1.0);
    let mut levels = Vec::new();
    let mut cur = img;
    loop {
        levels.push(cur.px.iter().flat_map(|p| [to_srgb(p[0]), to_srgb(p[1]), to_srgb(p[2]), (p[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]).collect());
        if cur.w == 1 {
            break;
        }
        let w = cur.w / 2;
        let mut next = Img::new(w);
        for y in 0..w {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = cur.px[(y * 2 + dy) * cur.w + x * 2 + dx];
                    for k in 0..3 {
                        acc[k] += p[k] * p[3];
                    }
                    acc[3] += p[3];
                }
                let a = acc[3] / 4.0;
                let c = if acc[3] > 1e-5 { [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]] } else { mean };
                next.px[y * w + x] = [c[0], c[1], c[2], a];
            }
        }
        // rescale alpha so the alpha-tested coverage matches the top level
        let (mut lo, mut hi) = (0.5f32, 8.0f32);
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            if coverage(&next.px, mid) < target { lo = mid } else { hi = mid }
        }
        let s = 0.5 * (lo + hi);
        for p in next.px.iter_mut() {
            p[3] = (p[3] * s).min(1.0);
        }
        cur = next;
    }
    levels
}

pub fn build(device: &Device, queue: &Queue) -> (Texture, TextureView) {
    let mip_count = SIZE.ilog2() + 1;
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("leaf atlas"),
        size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: LAYERS },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for layer in 0..LAYERS {
        let mut img = Img::new(SIZE as usize);
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ (layer as u64 + 1) * 0x51ED27);
        match layer {
            0 => paint_broadleaf(&mut img, &mut rng, false),
            1 => paint_broadleaf(&mut img, &mut rng, true),
            2 => paint_spruce(&mut img, &mut rng),
            _ => paint_fern(&mut img, &mut rng),
        }
        for (mip, data) in build_mips(img).into_iter().enumerate() {
            let w = (SIZE >> mip).max(1);
            queue.write_texture(
                TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: Origin3d { x: 0, y: 0, z: layer }, aspect: TextureAspect::All },
                &data,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(w) },
                Extent3d { width: w, height: w, depth_or_array_layers: 1 },
            );
        }
    }
    let view = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
    (tex, view)
}

/// Write the foliage layers side by side as a PNG (debugging aid).
pub fn debug_png(path: &str) {
    let w = SIZE as usize;
    let n = LAYERS as usize;
    let mut out = vec![0u8; w * n * w * 4];
    for layer in 0..n {
        let mut img = Img::new(w);
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ (layer as u64 + 1) * 0x51ED27);
        match layer {
            0 => paint_broadleaf(&mut img, &mut rng, false),
            1 => paint_broadleaf(&mut img, &mut rng, true),
            2 => paint_spruce(&mut img, &mut rng),
            _ => paint_fern(&mut img, &mut rng),
        }
        for y in 0..w {
            for x in 0..w {
                let p = img.px[y * w + x];
                let bg = [0.55f32, 0.6, 0.65];
                let o = (y * w * n + layer * w + x) * 4;
                for k in 0..3 {
                    out[o + k] = to_srgb(p[k] * p[3] + bg[k].powf(2.2) * (1.0 - p[3]));
                }
                out[o + 3] = 255;
            }
        }
    }
    crate::write_png(path, (w * n) as u32, w as u32, &out);
}
