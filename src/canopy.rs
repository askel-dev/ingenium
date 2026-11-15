//! The canopy light volume: a clipmap of foliage density around the camera, built on the GPU
//! from the tree list, and what it gives the shading: how much of the sky each point sees
//! through the leaves (by direction), how much sun gets down to it, and how bright the leaves
//! that block its sky are. A forest's cousin of voxel GI: nothing is voxelised from meshes,
//! since where every tree stands and what its crown is made of is already known.
//!
//! Four levels of 128 x 128 voxels (1, 2, 4 and 8 m) around the camera. Vertically each follows
//! the terrain: from 4 m under the ground to 44 m over it, which holds the tallest spruce, in
//! rows no taller than 2 m: a canopy is a layer only some 20 m thick, and cubes 8 m tall
//! smeared its top up into the open air, so even the crowns' tops saw little sky. The levels
//! sit one above the other in each 3D texture and wrap around in x and z (a texel is a world
//! voxel modulo 128), so when the camera moves only the slabs it uncovers are rebuilt.
//! See canopy.wgsl for the passes.

use crate::foliage::LeafCover;
use crate::forest::{KINDS, SCATTER_CELL};
use crate::treegen::{TreeKind, KIND_BARK, KIND_LEAF};
use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{IVec2, Vec2, Vec3, Vec3Swizzles, Vec4};
use std::num::NonZeroU64;
use std::time::Instant;
use wgpu::*;

pub const LEVELS: usize = 4;
/// Voxels across a level.
const N: i32 = 128;
/// The volume's vertical range, in metres above the terrain.
const Y0: f32 = -4.0;
const Y1: f32 = 44.0;
/// The tallest a voxel gets (m).
const MAX_DY: f32 = 2.0;
/// Each level's first row in the stacked textures: 48, 24, 24 and 24 rows.
const SLAB: [u32; LEVELS] = [0, 48, 72, 96];
const ROWS: u32 = 120;

/// Level l's rows: the voxels are 2^l m across and at most MAX_DY tall.
fn rows(level: u32) -> u32 {
    ((Y1 - Y0) / ((1u32 << level) as f32).min(MAX_DY)) as u32
}
/// Once the sun has moved, every frame each level refreshes one of this many strips of itself
/// (sun and sky), so the change reaches all of it in as many frames. Nothing else in the volume
/// changes (the trees stand still; clouds and wind are left to the shading), so with the sun
/// still there is nothing to refresh.
const SWEEP: i32 = 32;
/// A frame's boxes: per level at most two scrolled slabs, the two that moved INSET inside, and a
/// sweep strip.
const MAX_BOXES: usize = 24;
/// A voxel's light is worked out once it is this many voxels inside its level, where its cones
/// find the level around it on every side; at the edge they would have to start out in the
/// next, coarser level, and it would keep that. The shading doesn't reach this far out.
const INSET: i32 = 3;
// what a box runs (`BoxParams::passes`)
const DENSITY: u32 = 1;
const SUN: u32 = 2;
const LIGHT: u32 = 4;
const BOX_STRIDE: u64 = 256;

/// Per-tree density: voxels per side of a kind's box.
const KIND_RES: usize = 32;
const KIND_MIPS: u32 = 6;
/// Leaves bunch up (cards on a twig, twigs on a branch) and shade each other more than leaves
/// spread evenly through a voxel would: they block this much of what they would spread out.
/// (Measured: the volume's light straight up under a closed canopy against the sky seen
/// between the drawn leaves.)
const CLUMPING: f32 = 0.8;
/// The largest a tree is scaled and stretched (tree_common.wgsl, `inst`).
const MAX_SIZE: f32 = 1.30 * 1.12;

const DENSITY_FMT: TextureFormat = TextureFormat::Rgba16Float;
const SUN_FMT: TextureFormat = TextureFormat::Rgba8Unorm;
const LIGHT_FMT: TextureFormat = TextureFormat::Rgba16Float;

/// Extinction per tree kind, in the kind's own space (see `bake_kinds`).
pub struct KindDensity {
    _tex: Texture,
    pub view: TextureView,
    /// per kind: x centre height, y half side of its box (tree space, m), z voxel size (m)
    pub boxes: [Vec4; KINDS],
    /// how far any tree's wood or leaves reach from its trunk (m), at the largest size
    pub reach: f32,
}

/// The extinction (1/m) of each kind's LOD0, in a 32^3 box around (0, center_y, 0) that holds
/// the whole tree, with mips. Leaves block `0.5 * area` of light (a flat leaf at a random angle
/// shows half its area), of the area of the card that is leaf (`cover`, from the atlas); bark
/// `area / 4` (any convex body shows a quarter of its surface on average). The kinds are
/// stacked along z.
pub fn bake_kinds(device: &Device, queue: &Queue, kinds: &[TreeKind], cover: &LeafCover) -> KindDensity {
    let t0 = Instant::now();
    let baked: Vec<Baked> = std::thread::scope(|s| {
        let jobs: Vec<_> = kinds.iter().map(|k| s.spawn(move || bake_kind(k, cover))).collect();
        jobs.into_iter().map(|j| j.join().expect("canopy density bake")).collect()
    });
    let n = KIND_RES as u32;
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("tree density"),
        size: Extent3d { width: n, height: n, depth_or_array_layers: n * KINDS as u32 },
        mip_level_count: KIND_MIPS,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::R16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut levels: Vec<Vec<Vec<f32>>> = baked.iter().map(|b| vec![b.vol.clone()]).collect();
    for kind in levels.iter_mut() {
        for m in 1..KIND_MIPS as usize {
            let next = downsample(&kind[m - 1], KIND_RES >> (m - 1));
            kind.push(next);
        }
    }
    for m in 0..KIND_MIPS as usize {
        let r = (KIND_RES >> m) as u32;
        let data: Vec<u16> = levels.iter().flat_map(|k| k[m].iter().map(|&x| f16_bits(x))).collect();
        queue.write_texture(
            TexelCopyTextureInfo { texture: &tex, mip_level: m as u32, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            cast_slice(&data),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(r * 2), rows_per_image: Some(r) },
            Extent3d { width: r, height: r, depth_or_array_layers: r * KINDS as u32 },
        );
    }
    let mut boxes = [Vec4::ZERO; KINDS];
    for (k, kind) in kinds.iter().enumerate() {
        boxes[k] = Vec4::new(kind.center_y, kind.half, 2.0 * kind.half / KIND_RES as f32, 0.0);
    }
    let reach = baked.iter().map(|b| b.reach).fold(0.0, f32::max) * MAX_SIZE;
    let range = |f: fn(&Baked) -> f32| {
        let v: Vec<f32> = baked.iter().map(f).collect();
        (v.iter().copied().fold(f32::INFINITY, f32::min), v.iter().copied().fold(0.0, f32::max))
    };
    let (lai, gap) = (range(|b| b.lai), range(|b| b.gap));
    println!(
        "canopy: tree densities baked in {:.0} ms · under a crown: leaf area index {:.1}-{:.1}, light straight through {:.0}-{:.0}% · crowns reach {reach:.1} m",
        t0.elapsed().as_secs_f32() * 1000.0,
        lai.0,
        lai.1,
        gap.0 * 100.0,
        gap.1 * 100.0,
    );
    let view = tex.create_view(&Default::default());
    KindDensity { _tex: tex, view, boxes, reach }
}

struct Baked {
    vol: Vec<f32>,
    /// furthest any leaf or bark is from the trunk axis (m)
    reach: f32,
    /// leaf area over the ground the crown covers
    lai: f32,
    /// light that gets straight through the crown, on average over the ground it covers
    gap: f32,
}

fn bake_kind(kind: &TreeKind, cover: &LeafCover) -> Baked {
    let n = KIND_RES;
    let kv = 2.0 * kind.half / n as f32;
    let lo = Vec3::new(-kind.half, kind.center_y - kind.half, -kind.half);
    let mut vol = vec![0.0f32; n * n * n];
    let mut leaf = vec![0.0f32; n * n]; // leaf area per column, for the leaf area index
    let mesh = &kind.lods[0];
    let mut reach = 0.0f32;
    for tri in mesh.idx.as_chunks::<3>().0 {
        let v = [tri[0], tri[1], tri[2]].map(|i| &mesh.verts[i as usize]);
        let k = v[0].attr[0];
        let is_leaf = k == KIND_LEAF;
        // the leaf cards' share of leaf, over the atlas cell each shows
        let leaf_part = if is_leaf {
            let uv = v.map(|v| Vec2::from(v.uv));
            cover.over(v[0].attr[3] as usize, uv[0].min(uv[1]).min(uv[2]), uv[0].max(uv[1]).max(uv[2]))
        } else {
            0.0
        };
        let cross = if is_leaf {
            0.5 * leaf_part * CLUMPING
        } else if k == KIND_BARK {
            0.25
        } else {
            continue;
        };
        let p = v.map(|v| Vec3::from(v.pos));
        for q in p {
            reach = reach.max(q.xz().length());
        }
        let area = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
        let edge = (p[1] - p[0]).length().max((p[2] - p[0]).length()).max((p[2] - p[1]).length());
        // the triangle cut into m * m equal ones, each splatted at its centroid
        let m = ((edge / (kv * 0.5)).ceil() as usize).max(1);
        let w = area * cross / (m * m) as f32;
        let (e1, e2) = ((p[1] - p[0]) / m as f32, (p[2] - p[0]) / m as f32);
        let mut put = |a: f32, b: f32| {
            let q = p[0] + e1 * a + e2 * b;
            let g = (q - lo) / kv - 0.5;
            splat(&mut vol, n, g, w);
            if is_leaf {
                let (x, z) = ((g.x + 0.5).floor() as i32, (g.z + 0.5).floor() as i32);
                if (0..n as i32).contains(&x) && (0..n as i32).contains(&z) {
                    leaf[z as usize * n + x as usize] += w / (0.5 * CLUMPING);
                }
            }
        };
        for i in 0..m {
            for j in 0..m - i {
                put(i as f32 + 1.0 / 3.0, j as f32 + 1.0 / 3.0);
                if i + j + 1 < m {
                    put(i as f32 + 2.0 / 3.0, j as f32 + 2.0 / 3.0);
                }
            }
        }
    }
    let inv = 1.0 / (kv * kv * kv);
    vol.iter_mut().for_each(|x| *x *= inv);
    // leaf area index over the columns that hold a fair share of leaves
    let total: f32 = leaf.iter().sum();
    let mean = total / leaf.iter().filter(|&&a| a > 0.0).count().max(1) as f32;
    let under: Vec<usize> = (0..n * n).filter(|&c| leaf[c] > mean * 0.2).collect();
    let covered = under.len().max(1) as f32 * kv * kv;
    let gap = under.iter().map(|&c| (-(0..n).map(|y| vol[((c / n) * n + y) * n + c % n]).sum::<f32>() * kv).exp()).sum::<f32>() / under.len().max(1) as f32;
    Baked { vol, reach, lai: total / covered, gap }
}

fn splat(vol: &mut [f32], n: usize, g: Vec3, w: f32) {
    let f = g.floor();
    let t = g - f;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let (x, y, z) = (f.x as i32 + dx, f.y as i32 + dy, f.z as i32 + dz);
                let r = 0..n as i32;
                if !(r.contains(&x) && r.contains(&y) && r.contains(&z)) {
                    continue;
                }
                let wx = if dx == 1 { t.x } else { 1.0 - t.x };
                let wy = if dy == 1 { t.y } else { 1.0 - t.y };
                let wz = if dz == 1 { t.z } else { 1.0 - t.z };
                vol[(z as usize * n + y as usize) * n + x as usize] += w * wx * wy * wz;
            }
        }
    }
}

/// Half the resolution: each voxel the mean of the 8 it covers.
fn downsample(src: &[f32], n: usize) -> Vec<f32> {
    let h = n / 2;
    let mut out = vec![0.0; h * h * h];
    for z in 0..h {
        for y in 0..h {
            for x in 0..h {
                let mut s = 0.0;
                for (dx, dy, dz) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)] {
                    s += src[((2 * z + dz) * n + 2 * y + dy) * n + 2 * x + dx];
                }
                out[(z * h + y) * h + x] = s / 8.0;
            }
        }
    }
    out
}

/// f32 to IEEE half (round to nearest; the densities are positive and well inside its range).
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        return sign | ((m + (1 << (shift - 1))) >> shift) as u16;
    }
    let h = ((exp as u32) << 10) | (man >> 13);
    // rounding may carry into the exponent, which is still the right answer
    sign | (h + ((man >> 12) & 1)) as u16
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoxParams {
    /// the box's first voxel (world voxel coordinates of its level) and its size in voxels
    min: [i32; 2],
    size: [u32; 2],
    level: u32,
    /// DENSITY | SUN | LIGHT
    passes: u32,
    _pad: [u32; 58],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StaticParams {
    kinds: [Vec4; KINDS],
    /// x scatter cells per side, y tree count
    dims: [u32; 4],
    /// scatter cells to look through around a column, per level
    reach: [u32; 4],
    /// x scatter cell size (m), y highest terrain (m)
    misc: Vec4,
}

struct Pipes {
    density: ComputePipeline,
    sun: ComputePipeline,
    light: ComputePipeline,
    density_bg: BindGroup,
    sun_bg: BindGroup,
    light_bg: BindGroup,
}

pub struct Canopy {
    _textures: [Texture; 3],
    /// What the shading samples (group 0): the sky's SH over the leaves' light, see canopy.wgsl.
    pub light_view: TextureView,
    /// Repeats in x and z (the volume wraps around), clamps in y.
    pub sampler: Sampler,
    /// Unfiltered, for gathering the heightmap (globals.wgsl, `height_fast`).
    pub point_sampler: Sampler,
    density_view: TextureView,
    sun_view: TextureView,
    box_buf: Buffer,
    pipes: Option<Pipes>,
    /// each level's first voxel (in its own voxel units), None until it is filled
    origins: [Option<IVec2>; LEVELS],
    sweep: i32,
    /// strips still to refresh since the sun last moved
    sweep_left: i32,
    sun: Vec3,
    atmos: Option<usize>,
    boxes: Vec<BoxParams>,
    /// this frame refills every level
    pub filling: bool,
}

fn volume(device: &Device, label: &str, rows: u32, format: TextureFormat) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: N as u32, height: rows, depth_or_array_layers: N as u32 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry { binding, visibility: ShaderStages::COMPUTE, ty, count: None }
}

fn tex3d() -> BindingType {
    BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D3, multisampled: false }
}

fn store3d(format: TextureFormat) -> BindingType {
    BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format, view_dimension: TextureViewDimension::D3 }
}

fn uniform(dynamic: bool) -> BindingType {
    BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: None }
}

fn storage(read_only: bool) -> BindingType {
    BindingType::Buffer { ty: BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }
}

fn group(device: &Device, label: &str, layout: &BindGroupLayout, entries: &[(u32, BindingResource)]) -> BindGroup {
    let e: Vec<BindGroupEntry> = entries.iter().map(|(b, r)| BindGroupEntry { binding: *b, resource: r.clone() }).collect();
    device.create_bind_group(&BindGroupDescriptor { label: Some(label), layout, entries: &e })
}

impl Canopy {
    /// The volume's textures; `init` builds the passes once the forest exists.
    pub fn new(device: &Device) -> Canopy {
        let density = volume(device, "canopy density", ROWS, DENSITY_FMT);
        let sun = volume(device, "canopy sun", ROWS, SUN_FMT);
        let light = volume(device, "canopy light", 2 * ROWS, LIGHT_FMT);
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("canopy"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        Canopy {
            point_sampler: device.create_sampler(&SamplerDescriptor { label: Some("point"), ..Default::default() }),
            light_view: light.create_view(&Default::default()),
            density_view: density.create_view(&Default::default()),
            sun_view: sun.create_view(&Default::default()),
            _textures: [density, sun, light],
            sampler,
            box_buf: buffer(device, "canopy boxes", BOX_STRIDE * MAX_BOXES as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST),
            pipes: None,
            origins: [None; LEVELS],
            sweep: 0,
            sweep_left: 0,
            sun: Vec3::ZERO,
            atmos: None,
            boxes: Vec::new(),
            filling: false,
        }
    }

    /// Builds the passes, and once the index from scatter cell to tree. `trees`: the tree
    /// records (scatter.wgsl), `max_height`: the highest the terrain gets.
    #[allow(clippy::too_many_arguments)]
    pub fn init(&mut self, device: &Device, queue: &Queue, global_bgl: &BindGroupLayout, trees: &Buffer, tree_count: u32, kinds: &KindDensity, world: f32, max_height: f32) {
        let dim = (world / SCATTER_CELL) as u32;
        let cells = buffer(device, "canopy tree cells", dim as u64 * dim as u64 * 4, BufferUsages::STORAGE);
        let mut reach = [0u32; 4];
        for (l, r) in reach.iter_mut().enumerate() {
            let vs = (1u32 << l) as f32;
            *r = ((kinds.reach + vs * 0.71) / SCATTER_CELL).ceil() as u32;
        }
        let params = StaticParams { kinds: kinds.boxes, dims: [dim, tree_count, 0, 0], reach, misc: Vec4::new(SCATTER_CELL, max_height + Y1, 0.0, 0.0) };
        let static_buf = buffer_init(device, "canopy params", bytemuck::bytes_of(&params), BufferUsages::UNIFORM);

        // the index: one pass over the trees, each writing itself (+ 1) into its cell
        let index_bgl = bgl(device, "canopy index", &[uniform_entry(0, ShaderStages::COMPUTE), storage_entry(1, ShaderStages::COMPUTE, true), storage_entry(2, ShaderStages::COMPUTE, false)]);
        let index_bg = bind_group(device, "canopy index", &index_bgl, &[static_buf.as_entire_binding(), trees.as_entire_binding(), cells.as_entire_binding()]);
        let module = shader(device, "canopy index", &[include_str!("shaders/canopy_index.wgsl")]);
        let index_pipe = compute_pipeline(device, "canopy index", &pipeline_layout(device, &[&index_bgl]), &module, "main");
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&index_pipe);
            cp.set_bind_group(0, &index_bg, &[]);
            let groups = tree_count.div_ceil(256);
            let (x, y) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups.max(1), 1) };
            cp.dispatch_workgroups(x, y, 1);
        }
        queue.submit([enc.finish()]);

        let kind_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("tree density"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let boxes = BindingResource::Buffer(BufferBinding { buffer: &self.box_buf, offset: 0, size: NonZeroU64::new(BOX_STRIDE) });
        let module = shader(device, "canopy", &[NOISE_WGSL, GLOBALS_WGSL, TREE_COMMON_WGSL, include_str!("shaders/canopy.wgsl")]);
        let make = |label: &str, entries: &[BindGroupLayoutEntry], resources: &[(u32, BindingResource)], entry_point: &str| {
            let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some(label), entries });
            let bg = group(device, label, &layout, resources);
            let pipe = compute_pipeline(device, label, &pipeline_layout(device, &[global_bgl, &layout]), &module, entry_point);
            (pipe, bg)
        };
        let (density, density_bg) = make(
            "canopy density",
            &[
                entry(0, uniform(true)),
                entry(1, uniform(false)),
                entry(2, storage(true)),
                entry(3, storage(true)),
                entry(4, tex3d()),
                entry(5, BindingType::Sampler(SamplerBindingType::Filtering)),
                entry(6, store3d(DENSITY_FMT)),
            ],
            &[
                (0, boxes.clone()),
                (1, static_buf.as_entire_binding()),
                (2, trees.as_entire_binding()),
                (3, cells.as_entire_binding()),
                (4, BindingResource::TextureView(&kinds.view)),
                (5, BindingResource::Sampler(&kind_sampler)),
                (6, BindingResource::TextureView(&self.density_view)),
            ],
            "density",
        );
        let (sun, sun_bg) = make(
            "canopy sun",
            &[entry(0, uniform(true)), entry(1, uniform(false)), entry(7, tex3d()), entry(9, store3d(SUN_FMT))],
            &[(0, boxes.clone()), (1, static_buf.as_entire_binding()), (7, BindingResource::TextureView(&self.density_view)), (9, BindingResource::TextureView(&self.sun_view))],
            "sun",
        );
        let (light, light_bg) = make(
            "canopy light",
            &[entry(0, uniform(true)), entry(1, uniform(false)), entry(7, tex3d()), entry(10, tex3d()), entry(11, store3d(LIGHT_FMT))],
            &[
                (0, boxes),
                (1, static_buf.as_entire_binding()),
                (7, BindingResource::TextureView(&self.density_view)),
                (10, BindingResource::TextureView(&self.sun_view)),
                (11, BindingResource::TextureView(&self.light_view)),
            ],
            "light",
        );
        self.pipes = Some(Pipes { density, sun, light, density_bg, sun_bg, light_bg });
    }

    /// Moves the levels with the camera and picks this frame's work: the slabs each level
    /// scrolled over (density, sun and sky) and, while the sun has moved, a strip of it to
    /// refresh (sun and sky). Every level is refilled when it jumped further than its own
    /// width, when the time of day changed and when the volume is switched back on. Returns the
    /// frame uniform's `canopy`.
    pub fn update(&mut self, queue: &Queue, cam: Vec3, sun: Vec3, atmos: usize, enabled: bool) -> [Vec4; LEVELS + 1] {
        self.boxes.clear();
        self.filling = false;
        let mut out = [Vec4::ZERO; LEVELS + 1];
        out[LEVELS] = Vec4::new(Y0, Y1, 0.0, 0.0);
        if !enabled || self.pipes.is_none() {
            self.origins = [None; LEVELS];
            return out;
        }
        let refill = self.atmos != Some(atmos);
        self.atmos = Some(atmos);
        if sun != self.sun {
            self.sun = sun;
            self.sweep_left = SWEEP;
        }
        let sweep = (self.sweep_left > 0).then_some(self.sweep);
        if sweep.is_some() {
            self.sweep = (self.sweep + 1) % SWEEP;
            self.sweep_left -= 1;
        }
        let mut full = 0;
        let mut add = |min: IVec2, size: IVec2, level: usize, passes: u32| {
            self.boxes.push(BoxParams { min: min.into(), size: [size.x as u32, size.y as u32], level: level as u32, passes, _pad: [0; 58] });
        };
        for l in 0..LEVELS {
            let vs = (1 << l) as f32;
            let o = IVec2::new((cam.x / vs).floor() as i32, (cam.z / vs).floor() as i32) - N / 2;
            match self.origins[l] {
                Some(prev) if !refill && (o - prev).abs().max_element() < N => {
                    // The slabs the level moved onto get their density and a first sun (for the
                    // cones that pass through them), and the ones that just came INSET voxels
                    // inside the sun again and their light.
                    let d = o - prev;
                    if d.x != 0 {
                        let x0 = if d.x > 0 { prev.x + N } else { o.x };
                        add(IVec2::new(x0, o.y), IVec2::new(d.x.abs(), N), l, DENSITY | SUN);
                        let x1 = if d.x > 0 { x0 - INSET } else { x0 + INSET };
                        add(IVec2::new(x1, o.y), IVec2::new(d.x.abs(), N), l, SUN | LIGHT);
                    }
                    if d.y != 0 {
                        let z0 = if d.y > 0 { prev.y + N } else { o.y };
                        add(IVec2::new(o.x, z0), IVec2::new(N, d.y.abs()), l, DENSITY | SUN);
                        let z1 = if d.y > 0 { z0 - INSET } else { z0 + INSET };
                        add(IVec2::new(o.x, z1), IVec2::new(N, d.y.abs()), l, SUN | LIGHT);
                    }
                    if let Some(s) = sweep {
                        let w = N / SWEEP;
                        add(IVec2::new(o.x + s * w, o.y), IVec2::new(w, N), l, SUN | LIGHT);
                    }
                }
                _ => {
                    add(o, IVec2::splat(N), l, DENSITY | SUN | LIGHT);
                    full += 1;
                }
            }
            self.origins[l] = Some(o);
            out[l] = Vec4::new(o.x as f32 * vs, o.y as f32 * vs, vs, SLAB[l] as f32);
        }
        self.filling = full == LEVELS;
        if self.filling {
            self.sweep_left = 0; // everything is new
        }
        out[LEVELS].w = LEVELS as f32;
        let mut bytes = vec![0u8; self.boxes.len() * BOX_STRIDE as usize];
        for (i, b) in self.boxes.iter().enumerate() {
            bytes[i * BOX_STRIDE as usize..(i + 1) * BOX_STRIDE as usize].copy_from_slice(bytemuck::bytes_of(b));
        }
        queue.write_buffer(&self.box_buf, 0, &bytes);
        out
    }

    /// Whether `record` has anything to do this frame.
    pub fn busy(&self) -> bool {
        !self.boxes.is_empty()
    }

    /// The frame's work, in a compute pass. `global_bg` must not hold the light volume itself.
    pub fn record(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup) {
        let Some(p) = &self.pipes else { return };
        let off = |i: usize| [(i as u64 * BOX_STRIDE) as u32];
        cp.set_bind_group(0, global_bg, &[]);
        // one thread per voxel (canopy.wgsl, `box_voxel`): at most 128 * 128 * 48 / 64 groups
        for (pass, pipe, bg) in [(DENSITY, &p.density, &p.density_bg), (SUN, &p.sun, &p.sun_bg), (LIGHT, &p.light, &p.light_bg)] {
            cp.set_pipeline(pipe);
            for (i, b) in self.boxes.iter().enumerate().filter(|(_, b)| b.passes & pass != 0) {
                cp.set_bind_group(1, bg, &off(i));
                cp.dispatch_workgroups((b.size[0] * b.size[1] * rows(b.level)).div_ceil(64), 1, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        for (x, bits) in [(0.0f32, 0u16), (1.0, 0x3c00), (0.5, 0x3800), (2.0, 0x4000), (65504.0, 0x7bff), (1e-7, 0x0002), (0.333_251_95, 0x3555)] {
            assert_eq!(f16_bits(x), bits, "{x}");
        }
    }

    #[test]
    fn slabs_stack_every_level() {
        let mut row = 0;
        for (l, &s) in SLAB.iter().enumerate() {
            assert_eq!(s, row);
            row += rows(l as u32);
        }
        assert_eq!(row, ROWS);
    }

    #[test]
    fn mips_keep_the_mean() {
        let n = 4;
        let v: Vec<f32> = (0..n * n * n).map(|i| i as f32).collect();
        let d = downsample(&v, n);
        let mean = |x: &[f32]| x.iter().sum::<f32>() / x.len() as f32;
        assert!((mean(&v) - mean(&d)).abs() < 1e-4);
    }
}
