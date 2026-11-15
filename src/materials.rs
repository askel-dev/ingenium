//! Scanned PBR materials (CC0 sets from Poly Haven, see assets/materials/README.md), decoded at
//! startup into two texture arrays with full mip chains:
//!
//!   t_mat_a  Rgba8UnormSrgb  rgb albedo, a height (0..1 over the set)
//!   t_mat_b  Rgba8Unorm      rg tangent-space normal xy (+x = +u, +y = +v), b roughness, a AO
//!
//! Each set's albedo is rescaled so its average is the colour the surface had before the scans,
//! which the lighting, exposure and atmosphere are tuned around (`Def::mean`). The normal maps
//! are OpenGL convention (+y towards the top of the image, i.e. towards -v), so y is flipped.
//! Coarser mips average the normals and fold the lost length into the roughness (Toksvig), so a
//! distant bumpy surface turns rough instead of sparkling.

use std::path::{Path, PathBuf};
use std::time::Instant;
use wgpu::*;

/// Texels per side of every layer. 2048 keeps scans sharp at arm's length; 1024 would use a
/// quarter of the memory (2 x 8 layers x 22 MB at 2048).
pub const SIZE: u32 = 2048;

struct Def {
    id: &'static str,
    /// average linear albedo after rescaling
    mean: [f32; 3],
    /// 1 keeps the scan's contrast around its mean, less flattens it
    contrast: f32,
    /// average AO to scale the map to: the albedo already has the old surface's average, so
    /// the crevices may darken it only a little more on the whole (some scans' AO maps are
    /// very dark: 0.48 on average for the meadow)
    ao: f32,
}

/// Layer order is fixed: tree.wgsl and terrain.wgsl index it with the MAT_* constants.
const DEFS: [Def; 8] = [
    // 0 broadleaf bark: grey-brown, deeply furrowed like an oak
    Def { id: "bark_willow_02", mean: [0.13, 0.11, 0.09], contrast: 1.0, ao: 0.8 },
    // 1 birch: a pale, smooth bark; the dark birch marks are painted over it in tree.wgsl
    Def { id: "japanese_sycamore", mean: [0.50, 0.49, 0.46], contrast: 0.45, ao: 0.95 },
    // 2 spruce: scaly conifer bark (its average is already what the spruce had)
    Def { id: "pine_bark", mean: [0.14, 0.088, 0.056], contrast: 1.0, ao: 0.85 },
    // 3 forest floor: dry leaf litter broken up with needles and twigs
    Def { id: "forest_leaves_04", mean: [0.105, 0.078, 0.052], contrast: 1.0, ao: 0.9 },
    // 4 forest floor: moss over bare soil
    Def { id: "brown_mud_leaves_01", mean: [0.07, 0.075, 0.03], contrast: 1.0, ao: 0.9 },
    // 5 meadow: grass and twigs over soil (recoloured by meadow_tint in terrain.wgsl)
    Def { id: "forrest_ground_01", mean: [0.045, 0.083, 0.016], contrast: 1.0, ao: 0.92 },
    // 6 bare soil with small stones: slopes, worn patches
    Def { id: "forest_ground_04", mean: [0.11, 0.085, 0.058], contrast: 1.0, ao: 0.92 },
    // 7 rock face
    Def { id: "rock_face_03", mean: [0.155, 0.142, 0.125], contrast: 1.0, ao: 0.92 },
];

pub struct Materials {
    _a: Texture,
    _b: Texture,
    pub view_a: TextureView,
    pub view_b: TextureView,
}

fn asset_dir() -> Option<PathBuf> {
    let candidates = [PathBuf::from("assets/materials"), Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/materials")];
    candidates.into_iter().find(|p| p.is_dir())
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb_u8_exact(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Linear to 8-bit sRGB through a table (the exact curve per texel was most of the startup).
struct SrgbLut(Vec<u8>);

impl SrgbLut {
    const N: usize = 1 << 14;
    fn new() -> SrgbLut {
        SrgbLut((0..Self::N).map(|i| linear_to_srgb_u8_exact(i as f32 / (Self::N - 1) as f32)).collect())
    }
    fn get(&self, c: f32) -> u8 {
        self.0[(c.clamp(0.0, 1.0) * (Self::N - 1) as f32 + 0.5) as usize]
    }
}

/// Decode a JPEG and resample it to SIZE x SIZE (sets come at 1K/2K, some not square).
fn load(path: &Path) -> Result<image::RgbImage, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.into_rgb8();
    let (w, h) = img.dimensions();
    Ok(if (w, h) == (SIZE, SIZE) {
        img
    } else if w == SIZE && h == 2 * SIZE {
        // a scan twice as tall as wide: average row pairs
        let src = img.as_raw();
        let row = (w * 3) as usize;
        let mut out = vec![0u8; row * SIZE as usize];
        for y in 0..SIZE as usize {
            let (a, b) = (&src[2 * y * row..(2 * y + 1) * row], &src[(2 * y + 1) * row..(2 * y + 2) * row]);
            for (o, (x, y)) in out[y * row..(y + 1) * row].iter_mut().zip(a.iter().zip(b)) {
                *o = ((*x as u16 + *y as u16 + 1) / 2) as u8;
            }
        }
        image::RgbImage::from_raw(SIZE, SIZE, out).unwrap()
    } else if w * 2 == SIZE && h * 2 == SIZE {
        upsample2(&img)
    } else {
        image::imageops::resize(&img, SIZE, SIZE, image::imageops::FilterType::Triangle)
    })
}

/// Bilinear 2x upsample that wraps at the edges (the scans tile).
fn upsample2(img: &image::RgbImage) -> image::RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let src = img.as_raw();
    let mut out = vec![0u8; w * h * 12];
    let at = |x: usize, y: usize, c: usize| src[((y % h) * w + (x % w)) * 3 + c] as u32;
    for y in 0..2 * h {
        // output texel centres sit at 1/4 and 3/4 between source texels
        let (y0, fy) = if y % 2 == 0 { ((y / 2 + h - 1) % h, 3) } else { (y / 2, 1) };
        for x in 0..2 * w {
            let (x0, fx) = if x % 2 == 0 { ((x / 2 + w - 1) % w, 3) } else { (x / 2, 1) };
            for c in 0..3 {
                // weights in quarters: the nearer source texel gets 3
                let v = at(x0, y0, c) * (4 - fx) * (4 - fy)
                    + at(x0 + 1, y0, c) * fx * (4 - fy)
                    + at(x0, y0 + 1, c) * (4 - fx) * fy
                    + at(x0 + 1, y0 + 1, c) * fx * fy;
                out[(y * 2 * w + x) * 3 + c] = ((v + 8) / 16) as u8;
            }
        }
    }
    image::RgbImage::from_raw((2 * w) as u32, (2 * h) as u32, out).unwrap()
}

/// One layer at full resolution, in float: linear albedo + height, and unit normal + roughness + AO.
struct Layer {
    a: Vec<[f32; 4]>,
    n: Vec<[f32; 3]>,
    rough: Vec<f32>,
    ao: Vec<f32>,
}

fn build_layer(dir: &Path, def: &Def) -> Result<Layer, String> {
    let d = dir.join(def.id);
    // the four maps decode in parallel too
    let [diff, nor, arm, disp] = std::thread::scope(|s| {
        ["diff.jpg", "nor_gl.jpg", "arm.jpg", "disp.jpg"].map(|f| {
            let path = d.join(f);
            s.spawn(move || load(&path))
        })
        .map(|h| h.join().expect("decode thread"))
    });
    let (diff, nor, arm, disp) = (diff?, nor?, arm?, disp?);
    let count = (SIZE * SIZE) as usize;
    let lut: Vec<f32> = (0..256).map(|i| srgb_to_linear(i as f32 / 255.0)).collect();

    // albedo, rescaled to the target mean
    let mut lin: Vec<[f32; 3]> = diff.pixels().map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]]).collect();
    let mut sum = [0.0f64; 3];
    for p in &lin {
        for c in 0..3 {
            sum[c] += p[c] as f64;
        }
    }
    let src_mean: [f32; 3] = std::array::from_fn(|c| (sum[c] / count as f64).max(1e-4) as f32);
    for p in lin.iter_mut() {
        for c in 0..3 {
            p[c] = def.mean[c] * (p[c] / src_mean[c]).powf(def.contrast);
        }
    }
    // the contrast curve moves the mean a little: put it back
    if def.contrast != 1.0 {
        let mut s = [0.0f64; 3];
        for p in &lin {
            for c in 0..3 {
                s[c] += p[c] as f64;
            }
        }
        for p in lin.iter_mut() {
            for c in 0..3 {
                p[c] *= def.mean[c] / (s[c] / count as f64) as f32;
            }
        }
    }

    // height: stretch the 1st..99th percentile to 0..1 so every set blends alike
    let mut hist = [0u32; 256];
    for p in disp.pixels() {
        hist[p[0] as usize] += 1;
    }
    let pct = |f: f32| {
        let target = (count as f32 * f) as u32;
        let mut acc = 0;
        for (i, &h) in hist.iter().enumerate() {
            acc += h;
            if acc >= target {
                return i as f32 / 255.0;
            }
        }
        1.0
    };
    let (h0, h1) = (pct(0.01), pct(0.99).max(pct(0.01) + 1.0 / 255.0));

    // AO: scale its darkening so the average comes out at `def.ao` (never deepen it)
    let raw_mean = arm.pixels().map(|p| p[0] as f64).sum::<f64>() / count as f64 / 255.0;
    let keep = ((1.0 - def.ao) / (1.0 - raw_mean as f32).max(1e-3)).min(1.0);
    let ao: Vec<f32> = arm.pixels().map(|p| 1.0 - (1.0 - p[0] as f32 / 255.0) * keep).collect();
    let rough: Vec<f32> = arm.pixels().map(|p| p[1] as f32 / 255.0).collect();
    let a: Vec<[f32; 4]> = lin.iter().zip(disp.pixels()).map(|(c, h)| [c[0], c[1], c[2], ((h[0] as f32 / 255.0 - h0) / (h1 - h0)).clamp(0.0, 1.0)]).collect();
    let n: Vec<[f32; 3]> = nor
        .pixels()
        .map(|p| {
            let x = p[0] as f32 / 127.5 - 1.0;
            let y = -(p[1] as f32 / 127.5 - 1.0); // OpenGL green points up the image: flip to +v
            let z = (p[2] as f32 / 127.5 - 1.0).max(0.0);
            let l = (x * x + y * y + z * z).sqrt().max(1e-4);
            [x / l, y / l, z / l]
        })
        .collect();
    Ok(Layer { a, n, rough, ao })
}

/// A layer for a missing set: the target colour, flat.
fn flat_layer(def: &Def) -> Layer {
    let count = (SIZE * SIZE) as usize;
    Layer {
        a: vec![[def.mean[0], def.mean[1], def.mean[2], 0.5]; count],
        n: vec![[0.0, 0.0, 1.0]; count],
        rough: vec![0.85; count],
        ao: vec![1.0; count],
    }
}

/// Full mip chains of one layer, packed to bytes: (albedo + height, normal + roughness + AO).
fn pack_mips(mut l: Layer) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let srgb = SrgbLut::new();
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut size = SIZE as usize;
    // `len` is the length of the averaged normals (1 at the top level)
    let mut len: Vec<f32> = vec![1.0; size * size];
    loop {
        let mut a8 = Vec::with_capacity(size * size * 4);
        for p in &l.a {
            a8.extend_from_slice(&[srgb.get(p[0]), srgb.get(p[1]), srgb.get(p[2]), (p[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]);
        }
        let mut b8 = Vec::with_capacity(size * size * 4);
        for i in 0..size * size {
            let n = l.n[i];
            // Toksvig: the shorter the averaged normal, the more the bumps under this texel
            // spread the reflection
            let ln = len[i].clamp(1e-3, 1.0);
            let alpha = l.rough[i] * l.rough[i];
            let a2 = (alpha * alpha + (1.0 - ln) / ln).min(1.0);
            let r = a2.sqrt().sqrt();
            let q = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            b8.extend_from_slice(&[q(n[0]), q(n[1]), (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8, (l.ao[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]);
        }
        out_a.push(a8);
        out_b.push(b8);
        if size == 1 {
            break;
        }
        // 2x2 box down: albedo in linear light, normals as vectors (renormalised for storage,
        // with their length carried in `len`)
        let h = size / 2;
        let mut na = Layer { a: Vec::with_capacity(h * h), n: Vec::with_capacity(h * h), rough: Vec::with_capacity(h * h), ao: Vec::with_capacity(h * h) };
        let mut nlen = Vec::with_capacity(h * h);
        for y in 0..h {
            for x in 0..h {
                let idx = [(2 * y) * size + 2 * x, (2 * y) * size + 2 * x + 1, (2 * y + 1) * size + 2 * x, (2 * y + 1) * size + 2 * x + 1];
                let mut a = [0.0f32; 4];
                let mut n = [0.0f32; 3];
                let (mut r, mut o) = (0.0, 0.0);
                for &i in &idx {
                    for c in 0..4 {
                        a[c] += l.a[i][c] * 0.25;
                    }
                    for c in 0..3 {
                        n[c] += l.n[i][c] * len[i] * 0.25;
                    }
                    r += l.rough[i] * 0.25;
                    o += l.ao[i] * 0.25;
                }
                let ln = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-4);
                na.a.push(a);
                na.n.push([n[0] / ln, n[1] / ln, n[2] / ln]);
                na.rough.push(r);
                na.ao.push(o);
                nlen.push(ln);
            }
        }
        l = na;
        len = nlen;
        size = h;
    }
    (out_a, out_b)
}

impl Materials {
    pub fn new(device: &Device, queue: &Queue) -> Materials {
        let t0 = Instant::now();
        let dir = asset_dir();
        if dir.is_none() {
            eprintln!("materials: assets/materials not found, using flat colours");
        }
        // one thread per layer: decoding and mip building are independent
        let packed: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)> = std::thread::scope(|s| {
            let handles: Vec<_> = DEFS
                .iter()
                .map(|def| {
                    let dir = dir.clone();
                    s.spawn(move || {
                        let layer = match dir.as_deref().map(|d| build_layer(d, def)) {
                            Some(Ok(l)) => l,
                            Some(Err(e)) => {
                                eprintln!("materials: {e}; using a flat colour for {}", def.id);
                                flat_layer(def)
                            }
                            None => flat_layer(def),
                        };
                        pack_mips(layer)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("material thread")).collect()
        });

        let mips = SIZE.ilog2() + 1;
        let make = |label: &str, format: TextureFormat, pick: &dyn Fn(&(Vec<Vec<u8>>, Vec<Vec<u8>>)) -> &Vec<Vec<u8>>| {
            let tex = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: DEFS.len() as u32 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (layer, p) in packed.iter().enumerate() {
                for (mip, data) in pick(p).iter().enumerate() {
                    let w = (SIZE >> mip).max(1);
                    queue.write_texture(
                        TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: Origin3d { x: 0, y: 0, z: layer as u32 }, aspect: TextureAspect::All },
                        data,
                        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(w) },
                        Extent3d { width: w, height: w, depth_or_array_layers: 1 },
                    );
                }
            }
            let view = tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
            (tex, view)
        };
        let (a, view_a) = make("material albedo", TextureFormat::Rgba8UnormSrgb, &|p| &p.0);
        let (b, view_b) = make("material normal", TextureFormat::Rgba8Unorm, &|p| &p.1);
        println!("materials: {} scanned sets at {SIZE}² in {:.2}s", DEFS.len(), t0.elapsed().as_secs_f32());
        Materials { _a: a, _b: b, view_a, view_b }
    }
}
