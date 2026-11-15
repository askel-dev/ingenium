//! Impostors: what stands in for a tree beyond its mesh LODs. Each tree variant's LOD0 is
//! rendered at startup from GRID x GRID directions spread over the upper hemisphere
//! (hemi-octahedral: tree_common.wgsl, `frame_dir`) into one layer of an atlas: albedo and
//! coverage, and the normal (tree space), ambient occlusion and whether the surface is leaf or
//! bark. Unlit, so the impostor is lit at runtime like the mesh (tree.wgsl, `fs_imp`), and the
//! per-tree colour applies to both alike.
//!
//! The bake renders at twice the final size and is reduced on the CPU: coverage becomes the
//! fraction of covered samples, and the mips keep each frame's alpha-tested coverage (as the
//! foliage atlas does), so a distant tree doesn't thin out. Both maps hold their values
//! multiplied by the coverage, so filtering never pulls in the empty texels around the leaves.

use crate::treegen::TreeVertex;
use crate::util::*;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use std::time::Instant;
use wgpu::*;

pub const GRID: u32 = 8;
pub const FRAME: u32 = 128;
pub const ATLAS: u32 = GRID * FRAME;
/// supersampling of the bake (each way)
const SS: u32 = 2;
/// down to two texels per frame
const MIPS: u32 = 7;

pub struct Impostors {
    _a: Texture,
    _b: Texture,
    pub view_a: TextureView,
    pub view_b: TextureView,
    /// per variant: x frame centre height, y frame half size (tree.wgsl, `ImpKinds`)
    pub kinds_buf: Buffer,
}

/// What to bake for one tree variant: its LOD0 in the shared tree buffers.
pub struct BakeKind {
    pub center_y: f32,
    pub half: f32,
    pub species: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BakeUniform {
    center_y: f32,
    half: f32,
    species: u32,
    pad: u32,
}

fn srgb_lut() -> Vec<f32> {
    (0..256)
        .map(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
        .collect()
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Full-sphere octahedral encoding, y up (tree_common.wgsl, `oct_decode`), in 0..1.
fn oct_encode(n: Vec3) -> [f32; 2] {
    let s = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut x, mut z) = (n.x / s, n.z / s);
    if n.y < 0.0 {
        let (ox, oz) = (x, z);
        x = (1.0 - oz.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        z = (1.0 - ox.abs()) * if oz >= 0.0 { 1.0 } else { -1.0 };
    }
    [x * 0.5 + 0.5, z * 0.5 + 0.5]
}

/// One texel before packing: linear albedo, coverage, normal, ao, leafness.
#[derive(Clone, Copy, Default)]
struct Px {
    c: [f32; 3],
    a: f32,
    n: Vec3,
    ao: f32,
    leaf: f32,
}

/// Reduce one layer of the bake (`a`, `b`: SS * ATLAS square, rgba8) to its packed mip chain.
fn reduce(a: &[u8], b: &[u8], lut: &[f32]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let w0 = (ATLAS * SS) as usize;
    let w = ATLAS as usize;
    let mut lvl = vec![Px::default(); w * w];
    for y in 0..w {
        for x in 0..w {
            let mut p = Px::default();
            let mut cnt = 0.0;
            for sy in 0..SS as usize {
                for sx in 0..SS as usize {
                    let i = ((y * SS as usize + sy) * w0 + x * SS as usize + sx) * 4;
                    if a[i + 3] < 64 {
                        continue;
                    }
                    cnt += 1.0;
                    for k in 0..3 {
                        p.c[k] += lut[a[i + k] as usize];
                    }
                    p.n += Vec3::new(b[i] as f32, b[i + 1] as f32, b[i + 2] as f32) / 127.5 - 1.0;
                    p.ao += b[i + 3] as f32 / 255.0;
                    p.leaf += if a[i + 3] > 192 { 1.0 } else { 0.0 };
                }
            }
            if cnt > 0.0 {
                for k in 0..3 {
                    p.c[k] /= cnt;
                }
                p.n = p.n.normalize_or(Vec3::Y);
                p.ao /= cnt;
                p.leaf /= cnt;
                p.a = cnt / (SS * SS) as f32;
            }
            lvl[y * w + x] = p;
        }
    }
    // each frame's alpha-tested coverage, for the mips to keep
    let fr = FRAME as usize;
    let grid = GRID as usize;
    let mut target = vec![0.0f32; grid * grid];
    for (f, t) in target.iter_mut().enumerate() {
        let (fx, fy) = (f % grid, f / grid);
        let mut n = 0;
        for y in 0..fr {
            for x in 0..fr {
                if lvl[(fy * fr + y) * w + fx * fr + x].a >= 0.5 {
                    n += 1;
                }
            }
        }
        *t = n as f32 / (fr * fr) as f32;
    }
    let mut out = Vec::with_capacity(MIPS as usize);
    let mut lw = w;
    loop {
        out.push(pack(&lvl, lut));
        if out.len() == MIPS as usize {
            break;
        }
        let nw = lw / 2;
        let mut next = vec![Px::default(); nw * nw];
        for y in 0..nw {
            for x in 0..nw {
                let mut p = Px::default();
                let mut ws = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let q = lvl[(y * 2 + dy) * lw + x * 2 + dx];
                    let wq = q.a.max(1e-3);
                    for k in 0..3 {
                        p.c[k] += q.c[k] * wq;
                    }
                    p.n += q.n * wq;
                    p.ao += q.ao * wq;
                    p.leaf += q.leaf * wq;
                    p.a += q.a;
                    ws += wq;
                }
                for k in 0..3 {
                    p.c[k] /= ws;
                }
                p.n = p.n.normalize_or(Vec3::Y);
                p.ao /= ws;
                p.leaf /= ws;
                p.a /= 4.0;
                next[y * nw + x] = p;
            }
        }
        // rescale each frame's alpha so the alpha-tested coverage matches the top level
        let nf = nw / grid;
        for f in 0..grid * grid {
            let t = target[f];
            if t <= 0.0 || t >= 1.0 {
                continue;
            }
            let (fx, fy) = (f % grid, f / grid);
            let texels: Vec<usize> = (0..nf).flat_map(|y| (0..nf).map(move |x| (fy * nf + y) * nw + fx * nf + x)).collect();
            let cov = |s: f32| texels.iter().filter(|&&i| next[i].a * s >= 0.5).count() as f32 / texels.len() as f32;
            let (mut lo, mut hi) = (0.5f32, 8.0f32);
            for _ in 0..14 {
                let mid = 0.5 * (lo + hi);
                if cov(mid) < t { lo = mid } else { hi = mid }
            }
            let s = 0.5 * (lo + hi);
            for &i in &texels {
                next[i].a = (next[i].a * s).min(1.0);
            }
        }
        lvl = next;
        lw = nw;
    }
    out
}

/// Pack a level: (albedo * coverage in sRGB, coverage), (octahedral normal, ao, leafness), all
/// but the coverage multiplied by it.
fn pack(lvl: &[Px], _lut: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut a8 = Vec::with_capacity(lvl.len() * 4);
    let mut b8 = Vec::with_capacity(lvl.len() * 4);
    for p in lvl {
        let a = p.a;
        a8.extend_from_slice(&[q(linear_to_srgb(p.c[0] * a)), q(linear_to_srgb(p.c[1] * a)), q(linear_to_srgb(p.c[2] * a)), q(a)]);
        let o = oct_encode(p.n);
        b8.extend_from_slice(&[q(o[0] * a), q(o[1] * a), q(p.ao * a), q(p.leaf * a)]);
    }
    (a8, b8)
}

#[allow(clippy::too_many_arguments)]
pub fn bake(device: &Device, queue: &Queue, global_bgl: &BindGroupLayout, global_bg: &BindGroup, vbuf: &Buffer, ibuf: &Buffer, kinds: &[BakeKind]) -> Impostors {
    let t0 = Instant::now();
    let size = ATLAS * SS;
    let target = |label: &str, format: TextureFormat| {
        device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let ta = target("impostor bake a", TextureFormat::Rgba8UnormSrgb);
    let tb = target("impostor bake b", TextureFormat::Rgba8Unorm);
    let td = device.create_texture(&TextureDescriptor {
        label: Some("impostor bake depth"),
        size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let (va, vb, vd) = (ta.create_view(&Default::default()), tb.create_view(&Default::default()), td.create_view(&Default::default()));

    let bake_bgl = bgl(device, "impostor bake", &[uniform_entry(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)]);
    let module = shader(device, "impostor bake", &[NOISE_WGSL, GLOBALS_WGSL, TREE_COMMON_WGSL, include_str!("shaders/impostor_bake.wgsl")]);
    let layout = pipeline_layout(device, &[global_bgl, &bake_bgl]);
    let vattrs = [
        VertexAttribute { format: VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
        VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
        VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 3 },
    ];
    let pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("impostor bake"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_bake"),
            compilation_options: Default::default(),
            buffers: &[Some(VertexBufferLayout { array_stride: std::mem::size_of::<TreeVertex>() as u64, step_mode: VertexStepMode::Vertex, attributes: &vattrs })],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::Greater)),
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("fs_bake"),
            compilation_options: Default::default(),
            targets: &[
                Some(ColorTargetState { format: TextureFormat::Rgba8UnormSrgb, blend: None, write_mask: ColorWrites::ALL }),
                Some(ColorTargetState { format: TextureFormat::Rgba8Unorm, blend: None, write_mask: ColorWrites::ALL }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    });

    let bytes = (size * size * 4) as u64;
    let staging = buffer(device, "impostor readback", bytes * 2, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
    let mut jobs = Vec::new();
    for k in kinds {
        let u = BakeUniform { center_y: k.center_y, half: k.half, species: k.species, pad: 0 };
        let ubuf = buffer_init(device, "impostor bake", bytemuck::bytes_of(&u), BufferUsages::UNIFORM);
        let bg = bind_group(device, "impostor bake", &bake_bgl, &[ubuf.as_entire_binding()]);
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let clear = |view| RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: StoreOp::Store },
            };
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("impostor bake"),
                color_attachments: &[Some(clear(&va)), Some(clear(&vb))],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &vd,
                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&pipe);
            rp.set_bind_group(0, global_bg, &[]);
            rp.set_bind_group(1, &bg, &[]);
            rp.set_vertex_buffer(0, vbuf.slice(..));
            rp.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
            rp.draw_indexed(k.first_index..k.first_index + k.index_count, k.base_vertex, 0..GRID * GRID);
        }
        for (i, t) in [&ta, &tb].into_iter().enumerate() {
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: t, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo { buffer: &staging, layout: TexelCopyBufferLayout { offset: i as u64 * bytes, bytes_per_row: Some(size * 4), rows_per_image: Some(size) } },
                Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        let data = read_buffer_blocking(device, &staging, bytes * 2);
        jobs.push(std::thread::spawn(move || {
            let lut = srgb_lut();
            let (a, b) = data.split_at(bytes as usize);
            reduce(a, b, &lut)
        }));
    }
    let layers: Vec<Vec<(Vec<u8>, Vec<u8>)>> = jobs.into_iter().map(|j| j.join().expect("impostor reduce")).collect();
    if let Ok(p) = std::env::var("INGENIUM_IMPOSTOR_PNG") {
        debug_png(&p, &layers);
    }

    let make = |label: &str, format: TextureFormat, second: bool| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: kinds.len() as u32 },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, chain) in layers.iter().enumerate() {
            for (mip, (a, b)) in chain.iter().enumerate() {
                let w = ATLAS >> mip;
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
    let (a, view_a) = make("impostor albedo", TextureFormat::Rgba8UnormSrgb, false);
    let (b, view_b) = make("impostor normal", TextureFormat::Rgba8Unorm, true);
    let mut kv = [Vec4::ZERO; 12];
    for (i, k) in kinds.iter().enumerate() {
        kv[i] = Vec4::new(k.center_y, k.half, 0.0, 0.0);
    }
    let kinds_buf = buffer_init(device, "impostor kinds", bytemuck::cast_slice(&kv), BufferUsages::UNIFORM);
    println!("impostors: {} variants x {}² frames at {FRAME}² baked in {:.2}s", kinds.len(), GRID, t0.elapsed().as_secs_f32());
    Impostors { _a: a, _b: b, view_a, view_b, kinds_buf }
}

/// Write each layer's top level (over grey) as a PNG (debugging aid, with the environment
/// variable INGENIUM_IMPOSTOR_PNG=path): `<path>_<layer>.png`.
fn debug_png(path: &str, layers: &[Vec<(Vec<u8>, Vec<u8>)>]) {
    let lut = srgb_lut();
    for (l, chain) in layers.iter().enumerate() {
        let a = &chain[0].0;
        let w = ATLAS as usize;
        let mut out = vec![0u8; w * w * 4];
        for i in 0..w * w {
            let al = a[i * 4 + 3] as f32 / 255.0;
            for k in 0..3 {
                let bg = 0.35f32;
                out[i * 4 + k] = (linear_to_srgb(lut[a[i * 4 + k] as usize] + bg * (1.0 - al)) * 255.0 + 0.5) as u8;
            }
            out[i * 4 + 3] = 255;
        }
        crate::write_png(&format!("{path}_{l}.png"), w as u32, w as u32, &out);
    }
}
