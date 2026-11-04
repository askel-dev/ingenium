//! Procedural terrain: generated once on the GPU, rendered with CDLOD
//! (quadtree patches whose vertices morph smoothly between detail levels).
//! The patches are picked by distance once per frame, then each view (the camera and the
//! shadow cascades) draws only those whose height bounds touch its frustum.

use crate::util::*;
use bytemuck::cast_slice;
use glam::{Vec2, Vec3, Vec4};
use wgpu::*;

pub const WORLD: f32 = 8192.0;
pub const HM_RES: u32 = 2048;
const LEAF: f32 = 32.0;
const PATCH_N: u32 = 32;
pub const LEVELS: usize = 9;
/// Most patches in one list.
const MAX_PATCHES: usize = 2048;
pub const PATCH_TRIS: u32 = PATCH_N * PATCH_N * 2;
/// Views the patches are culled for: the camera, then the two shadow cascades.
pub const VIEWS: usize = 3;
/// Patch lists in the instance buffer: one per view, then the camera's far canopy shell.
const LISTS: usize = VIEWS + 1;
const SHELL: usize = VIEWS;
/// Furthest the detail noise (`detail_h`, at most ~0.47 m) moves a vertex off the
/// heightmap, with margin.
const DETAIL_PAD: f32 = 1.0;
/// Highest the canopy shell rises above the ground (`shell_height` in terrain.wgsl,
/// at most 16.1 m), with margin.
const SHELL_PAD: f32 = 19.5;

pub struct Terrain {
    pub world: f32,
    pub res: u32,
    pub height_view: TextureView,
    pub normal_view: TextureView,
    pub data_view: TextureView,
    _textures: Vec<Texture>,
    heights: Vec<f32>,
    data: Vec<u8>,
    /// Lowest and highest ground under each quadtree node, per level (level 0 first).
    bounds: Vec<Vec<[f32; 2]>>,
    index_buf: Buffer,
    index_count: u32,
    patch_buf: Buffer,
    /// Patches the distance-based LOD selection picked this frame, before culling.
    pub selected: u32,
    /// (first instance, count) of each list in `patch_buf`.
    lists: [(u32, u32); LISTS],
    ranges: [f32; LEVELS],
    pipe: Option<RenderPipeline>,
    pipe_depth: Option<RenderPipeline>,
    pipe_shadow: Option<RenderPipeline>,
    pipe_canopy: Option<RenderPipeline>,
}

impl Terrain {
    pub fn new(device: &Device, queue: &Queue) -> Terrain {
        let res = HM_RES;
        let mk = |label: &str, format: TextureFormat| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: res, height: res, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let height_tex = mk("height", TextureFormat::R32Float);
        let normal_tex = mk("normal", TextureFormat::Rgba16Float);
        let data_tex = mk("world data", TextureFormat::Rgba8Unorm);
        let height_view = height_tex.create_view(&Default::default());
        let normal_view = normal_tex.create_view(&Default::default());
        let data_view = data_tex.create_view(&Default::default());

        // --- generate on the GPU
        let params: [f32; 4] = [WORLD, res as f32, 0.0, 0.0];
        let params_buf = buffer_init(device, "terrain gen params", cast_slice(&params), BufferUsages::UNIFORM);
        let layout = bgl(
            device,
            "terrain gen",
            &[
                uniform_entry(0, ShaderStages::COMPUTE),
                storage_texture_entry(1, TextureFormat::R32Float),
                storage_texture_entry(2, TextureFormat::Rgba8Unorm),
            ],
        );
        let bg = bind_group(
            device,
            "terrain gen",
            &layout,
            &[
                params_buf.as_entire_binding(),
                BindingResource::TextureView(&height_view),
                BindingResource::TextureView(&data_view),
            ],
        );
        let module = shader(device, "terrain gen", &[NOISE_WGSL, include_str!("shaders/terrain_gen.wgsl")]);
        let pl = pipeline_layout(device, &[&layout]);
        let pipe = compute_pipeline(device, "terrain gen", &pl, &module, "main");

        // second pass: normal map + sky visibility, read back from the finished heightmap
        let vis_layout = bgl(
            device,
            "sky visibility",
            &[
                uniform_entry(0, ShaderStages::COMPUTE),
                texture_entry(1, ShaderStages::COMPUTE, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
                storage_texture_entry(2, TextureFormat::Rgba16Float),
                texture_entry(3, ShaderStages::COMPUTE, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
            ],
        );
        let vis_bg = bind_group(
            device,
            "sky visibility",
            &vis_layout,
            &[params_buf.as_entire_binding(), BindingResource::TextureView(&height_view), BindingResource::TextureView(&normal_view), BindingResource::TextureView(&data_view)],
        );
        let vis_module = shader(device, "sky visibility", &[NOISE_WGSL, include_str!("shaders/sky_vis.wgsl")]);
        let vis_pipe = compute_pipeline(device, "sky visibility", &pipeline_layout(device, &[&vis_layout]), &vis_module, "main");

        let row = res * 4;
        let rb_height = buffer(device, "height readback", (row * res) as u64, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        let rb_data = buffer(device, "data readback", (row * res) as u64, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&vis_pipe);
            cp.set_bind_group(0, &vis_bg, &[]);
            cp.dispatch_workgroups(res.div_ceil(8), res.div_ceil(8), 1);
        }
        for (tex, buf) in [(&height_tex, &rb_height), (&data_tex, &rb_data)] {
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo {
                    buffer: buf,
                    layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(res) },
                },
                Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        let hbytes = read_buffer_blocking(device, &rb_height, (row * res) as u64);
        let heights: Vec<f32> = cast_slice(&hbytes).to_vec();
        let data = read_buffer_blocking(device, &rb_data, (row * res) as u64);

        // --- patch mesh: (N+1)^2 virtual vertices, positions come from vertex_index
        let n = PATCH_N;
        let mut idx: Vec<u32> = Vec::with_capacity((n * n * 6) as usize);
        for z in 0..n {
            for x in 0..n {
                let i0 = z * (n + 1) + x;
                let i1 = i0 + 1;
                let i2 = i0 + n + 1;
                let i3 = i2 + 1;
                idx.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }
        let index_buf = buffer_init(device, "terrain idx", cast_slice(&idx), BufferUsages::INDEX);
        let patch_buf = buffer(device, "terrain patches", (LISTS * MAX_PATCHES * 16) as u64, BufferUsages::VERTEX | BufferUsages::COPY_DST);
        let bounds = height_bounds(&heights, res);

        let mut ranges = [0.0; LEVELS];
        for (l, r) in ranges.iter_mut().enumerate() {
            *r = 3.0 * LEAF * (1u32 << l) as f32;
        }

        let (hmin, hmax) = heights.iter().fold((f32::MAX, f32::MIN), |(a, b), &h| (a.min(h), b.max(h)));
        println!("terrain: {res}x{res} heightmap over {:.1} km, heights {hmin:.0}..{hmax:.0} m", WORLD / 1000.0);

        Terrain {
            world: WORLD,
            res,
            height_view,
            normal_view,
            data_view,
            _textures: vec![height_tex, normal_tex, data_tex],
            heights,
            data,
            bounds,
            index_buf,
            index_count: idx.len() as u32,
            patch_buf,
            selected: 0,
            lists: [(0, 0); LISTS],
            ranges,
            pipe: None,
            pipe_depth: None,
            pipe_shadow: None,
            pipe_canopy: None,
        }
    }

    /// Morph start/end per level for the frame uniform.
    pub fn morph_ranges(&self) -> [Vec4; 12] {
        let mut out = [Vec4::ZERO; 12];
        for l in 0..LEVELS {
            let end = self.ranges[l];
            let prev = if l == 0 { 0.0 } else { self.ranges[l - 1] };
            let start = prev + (end - prev) * 0.6;
            out[l] = Vec4::new(start, end, 0.0, 0.0);
        }
        out
    }

    pub fn create_pipelines(&mut self, device: &Device, global: &BindGroupLayout, pass: &BindGroupLayout, hdr: TextureFormat, msaa: u32) {
        let module = shader(device, "terrain", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/terrain.wgsl")]);
        let layout = pipeline_layout(device, &[global, pass]);
        let attrs = [VertexAttribute { format: VertexFormat::Float32x4, offset: 0, shader_location: 0 }];
        let vbl = VertexBufferLayout { array_stride: 16, step_mode: VertexStepMode::Instance, attributes: &attrs };
        // the camera's colour pass, its depth-only occluder pass, or a shadow cascade
        let make = |shadow: bool, depth_only: bool| {
            let targets = scene_targets(hdr, !depth_only);
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(if shadow { "terrain shadow" } else if depth_only { "terrain depth" } else { "terrain" }),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vbl.clone())],
                },
                primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
                depth_stencil: Some(if shadow {
                    shadow_depth_state()
                } else {
                    depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual)
                }),
                multisample: MultisampleState { count: if shadow { 1 } else { msaa }, ..Default::default() },
                fragment: if shadow {
                    None
                } else {
                    Some(FragmentState {
                        module: &module,
                        entry_point: Some(if depth_only { "fs_depth" } else { "fs" }),
                        compilation_options: Default::default(),
                        targets: &targets,
                    })
                },
                multiview_mask: None,
                cache: None,
            })
        };
        self.pipe = Some(make(false, false));
        self.pipe_depth = Some(make(false, true));
        self.pipe_shadow = Some(make(true, false));

        let targets = scene_targets(hdr, true);
        self.pipe_canopy = Some(device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("canopy shell"),
            layout: Some(&layout),
            vertex: VertexState { module: &module, entry_point: Some("vs_canopy"), compilation_options: Default::default(), buffers: &[Some(vbl.clone())] },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual)),
            multisample: MultisampleState { count: msaa, mask: !0, alpha_to_coverage_enabled: msaa > 1 },
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs_canopy"), compilation_options: Default::default(), targets: &targets }),
            multiview_mask: None,
            cache: None,
        }));
    }

    /// Pick quadtree patches around `cam` by distance, then list for each view only the
    /// patches whose bounds touch its frustum (`views`: the camera, then the shadow
    /// cascades). The level of detail does not depend on the view, so every pass draws the
    /// same surface and the morphing stays crack-free. Patches reaching beyond `shell_from`
    /// metres are also listed for the camera's far canopy shell.
    pub fn select(&mut self, queue: &Queue, cam: Vec2, shell_from: f32, views: &[[Vec4; 6]; VIEWS]) {
        let mut picked: Vec<[f32; 4]> = Vec::with_capacity(1024);
        self.select_node(0.0, 0.0, LEVELS - 1, cam, &mut picked);
        picked.truncate(MAX_PATCHES);
        self.selected = picked.len() as u32;
        let mut lists: [Vec<[f32; 4]>; LISTS] = Default::default();
        for q in &picked {
            let (lo, hi) = self.patch_box(q);
            for (v, planes) in views.iter().enumerate() {
                if !outside(planes, lo, hi) {
                    lists[v].push(*q);
                }
            }
            let far = Vec2::new((cam.x - q[0]).abs().max((cam.x - q[0] - q[2]).abs()), (cam.y - q[1]).abs().max((cam.y - q[1] - q[2]).abs()));
            if far.length() > shell_from && !outside(&views[0], lo, hi + Vec3::Y * SHELL_PAD) {
                lists[SHELL].push(*q);
            }
        }
        let mut all: Vec<[f32; 4]> = Vec::with_capacity(lists.iter().map(Vec::len).sum());
        for (range, list) in self.lists.iter_mut().zip(&lists) {
            *range = (all.len() as u32, list.len() as u32);
            all.extend_from_slice(list);
        }
        if !all.is_empty() {
            queue.write_buffer(&self.patch_buf, 0, cast_slice(&all));
        }
    }

    /// Patches drawn this frame by view `view` (0 camera, 1 and 2 the shadow cascades).
    pub fn drawn(&self, view: usize) -> u32 {
        self.lists[view].1
    }

    /// World-space box around everything a patch can draw: the heightmap texels its
    /// vertices interpolate (wherever morphing moves them inside the patch) plus the detail
    /// noise.
    fn patch_box(&self, q: &[f32; 4]) -> (Vec3, Vec3) {
        let level = q[3] as usize;
        let n = (WORLD / LEAF) as usize >> level;
        let (i, j) = ((q[0] / q[2]) as usize, (q[1] / q[2]) as usize);
        let [lo, hi] = self.bounds[level][j.min(n - 1) * n + i.min(n - 1)];
        // a little slack for rounding in the vertex positions
        let m = 0.25;
        (Vec3::new(q[0] - m, lo - DETAIL_PAD, q[1] - m), Vec3::new(q[0] + q[2] + m, hi + DETAIL_PAD, q[1] + q[2] + m))
    }

    pub fn draw_canopy(&self, pass: &mut RenderPass<'_>) {
        let (first, count) = self.lists[SHELL];
        if count == 0 {
            return;
        }
        pass.set_pipeline(self.pipe_canopy.as_ref().unwrap());
        pass.set_vertex_buffer(0, self.patch_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, first..first + count);
    }

    fn select_node(&self, x: f32, z: f32, level: usize, cam: Vec2, out: &mut Vec<[f32; 4]>) {
        let size = LEAF * (1u32 << level) as f32;
        let closest = Vec2::new(cam.x.clamp(x, x + size), cam.y.clamp(z, z + size));
        let dmin = closest.distance(cam);
        if level == 0 || dmin > self.ranges[level - 1] {
            out.push([x, z, size, level as f32]);
            return;
        }
        let h = size * 0.5;
        for (ox, oz) in [(0.0, 0.0), (h, 0.0), (0.0, h), (h, h)] {
            self.select_node(x + ox, z + oz, level - 1, cam, out);
        }
    }

    /// Draw the patches listed for `view`: 0 is the camera, 1 and 2 the shadow cascades.
    pub fn draw(&self, pass: &mut RenderPass<'_>, view: usize) {
        let (first, count) = self.lists[view];
        if count == 0 {
            return;
        }
        let pipe = if view == 0 { &self.pipe } else { &self.pipe_shadow };
        pass.set_pipeline(pipe.as_ref().unwrap());
        pass.set_vertex_buffer(0, self.patch_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, first..first + count);
    }

    /// The camera's patches, depth only (the occluder pass before the main pass).
    pub fn draw_depth(&self, pass: &mut RenderPass<'_>) {
        let (first, count) = self.lists[0];
        if count == 0 {
            return;
        }
        pass.set_pipeline(self.pipe_depth.as_ref().unwrap());
        pass.set_vertex_buffer(0, self.patch_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, first..first + count);
    }

    fn sample<F: Fn(usize) -> f32>(&self, x: f32, z: f32, f: F) -> f32 {
        let t = self.world / self.res as f32;
        let u = x / t - 0.5;
        let v = z / t - 0.5;
        let max = self.res as i32 - 1;
        let (i, j) = (u.floor() as i32, v.floor() as i32);
        let (fx, fz) = (u - u.floor(), v - v.floor());
        let at = |a: i32, b: i32| f((b.clamp(0, max) * self.res as i32 + a.clamp(0, max)) as usize);
        let top = at(i, j) * (1.0 - fx) + at(i + 1, j) * fx;
        let bot = at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx;
        top * (1.0 - fz) + bot * fz
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z, |i| self.heights[i])
    }

    pub fn forest_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z, |i| self.data[i * 4] as f32 / 255.0)
    }

    /// Ground height as the terrain mesh draws it near the camera: heightmap plus detail
    /// noise at the 1 m vertices of the finest CDLOD level, split into the same triangles
    /// as the patch index buffer (each quad along its (x+1, z)-(x, z+1) diagonal).
    pub fn ground_at(&self, x: f32, z: f32) -> f32 {
        let (x0, z0) = (x.floor(), z.floor());
        let (fx, fz) = (x - x0, z - z0);
        let h = |dx: f32, dz: f32| {
            let (px, pz) = (x0 + dx, z0 + dz);
            self.height_at(px, pz) + detail_h(Vec2::new(px, pz))
        };
        let (b, c) = (h(1.0, 0.0), h(0.0, 1.0));
        if fx + fz <= 1.0 {
            let a = h(0.0, 0.0);
            a + (b - a) * fx + (c - a) * fz
        } else {
            let d = h(1.0, 1.0);
            d + (c - d) * (1.0 - fx) + (b - d) * (1.0 - fz)
        }
    }

    /// Smooth surface normal from the heightmap alone, for walkable-slope tests.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let e = 2.0;
        let dx = self.height_at(x - e, z) - self.height_at(x + e, z);
        let dz = self.height_at(x, z - e) - self.height_at(x, z + e);
        Vec3::new(dx, 2.0 * e, dz).normalize()
    }
}

/// Lowest and highest heightmap value under every quadtree node, level 0 (the finest
/// patches) first. A level-0 node spans every texel `height_at` in globals.wgsl reads for a
/// point of the patch: its bilinear lookup between texel centres reaches one texel beyond
/// the patch on each side.
fn height_bounds(heights: &[f32], res: u32) -> Vec<Vec<[f32; 2]>> {
    let res = res as i32;
    let per = (LEAF / (WORLD / res as f32)) as i32;
    let n0 = (WORLD / LEAF) as usize;
    let span = |i: usize| ((i as i32 * per - 1).max(0), ((i as i32 + 1) * per).min(res - 1));
    let mut level0 = vec![[0.0f32; 2]; n0 * n0];
    for j in 0..n0 {
        let (z0, z1) = span(j);
        for i in 0..n0 {
            let (x0, x1) = span(i);
            let mut b = [f32::MAX, f32::MIN];
            for z in z0..=z1 {
                let row = (z * res) as usize;
                for &h in &heights[row + x0 as usize..=row + x1 as usize] {
                    b = [b[0].min(h), b[1].max(h)];
                }
            }
            level0[j * n0 + i] = b;
        }
    }
    let mut levels = vec![level0];
    for l in 1..LEVELS {
        let (prev, pn) = (&levels[l - 1], n0 >> (l - 1));
        let n = pn / 2;
        let mut cur = vec![[0.0f32; 2]; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut b = [f32::MAX, f32::MIN];
                for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = prev[(2 * j + dj) * pn + 2 * i + di];
                    b = [b[0].min(c[0]), b[1].max(c[1])];
                }
                cur[j * n + i] = b;
            }
        }
        levels.push(cur);
    }
    levels
}

/// True if the box lies wholly on the outside of one of the planes (inside: dot(n, p) + d >= 0).
fn outside(planes: &[Vec4; 6], lo: Vec3, hi: Vec3) -> bool {
    planes.iter().any(|p| {
        // the corner furthest along the plane normal
        let c = Vec3::new(if p.x > 0.0 { hi.x } else { lo.x }, if p.y > 0.0 { hi.y } else { lo.y }, if p.z > 0.0 { hi.z } else { lo.z });
        p.truncate().dot(c) + p.w < 0.0
    })
}

// CPU copy of `gnoise` / `detail_h` from shaders/noise.wgsl and globals.wgsl, so the
// player stands on the ground that is drawn. Keep the two in sync.

fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
    (w >> 22) ^ w
}

fn grad2(x: i32, y: i32) -> Vec2 {
    let h = pcg(x as u32 ^ pcg((y as u32).wrapping_add(0x9e3779b9)));
    let a = (h >> 8) as f32 * (1.0 / 16777216.0) * std::f32::consts::TAU;
    Vec2::new(a.cos(), a.sin())
}

fn gnoise(p: Vec2) -> f32 {
    let fl = p.floor();
    let (i, j) = (fl.x as i32, fl.y as i32);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = grad2(i, j).dot(f);
    let b = grad2(i + 1, j).dot(f - Vec2::X);
    let c = grad2(i, j + 1).dot(f - Vec2::Y);
    let d = grad2(i + 1, j + 1).dot(f - Vec2::ONE);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y) * 1.4
}

fn detail_h(p: Vec2) -> f32 {
    gnoise(p * 0.21) * 0.35 + gnoise(p * 0.73) * 0.12
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rough random heightmap.
    fn heightmap(res: u32) -> Vec<f32> {
        (0..res * res).map(|i| (pcg(i) % 100_000) as f32 * 0.01).collect()
    }

    /// `height_at` from globals.wgsl.
    fn height_at(heights: &[f32], res: u32, p: Vec2) -> f32 {
        let t = WORLD / res as f32;
        let uv = p / t - 0.5;
        let fl = uv.floor();
        let f = uv - fl;
        let max = res as i32 - 1;
        let at = |x: i32, z: i32| heights[(z.clamp(0, max) * res as i32 + x.clamp(0, max)) as usize];
        let (i, j) = (fl.x as i32, fl.y as i32);
        let top = at(i, j) + (at(i + 1, j) - at(i, j)) * f.x;
        let bot = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * f.x;
        top + (bot - top) * f.y
    }

    #[test]
    fn height_bounds_hold_every_sampled_height() {
        let heights = heightmap(HM_RES);
        let bounds = height_bounds(&heights, HM_RES);
        assert_eq!(bounds.len(), LEVELS);
        assert_eq!(bounds[LEVELS - 1].len(), 1);
        for (level, b) in bounds.iter().enumerate() {
            let size = LEAF * (1u32 << level) as f32;
            let n = (WORLD / size) as usize;
            assert_eq!(b.len(), n * n);
            for k in 0..2000u32 {
                // points of random patches, including their edges and corners
                let h = pcg(k * 7919 + level as u32);
                let (i, j) = ((h % n as u32) as f32, ((h >> 12) % n as u32) as f32);
                let fx = [0.0, 1.0, (pcg(h) % 1000) as f32 / 1000.0][k as usize % 3];
                let fz = [1.0, 0.0, (pcg(h + 1) % 1000) as f32 / 1000.0][k as usize % 3];
                let p = Vec2::new((i + fx) * size, (j + fz) * size);
                let y = height_at(&heights, HM_RES, p);
                let [lo, hi] = b[j as usize * n + i as usize];
                assert!(y >= lo - 1e-3 && y <= hi + 1e-3, "level {level} patch ({i}, {j}) at {p}: {y} outside {lo}..{hi}");
            }
        }
    }

    #[test]
    fn boxes_outside_a_frustum_are_culled() {
        let m = glam::camera::rh::proj::directx::orthographic(-10.0, 10.0, -10.0, 10.0, 0.0, 100.0) * glam::camera::rh::view::look_to_mat4(Vec3::new(0.0, 50.0, 0.0), Vec3::NEG_Y, Vec3::Z);
        let planes = frustum_planes(m);
        let unit = |c: Vec3| (c - Vec3::ONE, c + Vec3::ONE);
        let (lo, hi) = unit(Vec3::ZERO);
        assert!(!outside(&planes, lo, hi));
        let (lo, hi) = unit(Vec3::new(30.0, 0.0, 0.0));
        assert!(outside(&planes, lo, hi));
        // straddling an edge counts as inside
        let (lo, hi) = unit(Vec3::new(10.5, 0.0, 0.0));
        assert!(!outside(&planes, lo, hi));
        // beyond the far plane
        let (lo, hi) = unit(Vec3::new(0.0, -60.0, 0.0));
        assert!(outside(&planes, lo, hi));
    }
}
