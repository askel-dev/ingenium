//! Procedural terrain: generated once on the GPU, rendered with CDLOD
//! (quadtree patches whose vertices morph smoothly between detail levels).

use crate::util::*;
use bytemuck::cast_slice;
use glam::{Vec2, Vec4};
use wgpu::*;

pub const WORLD: f32 = 8192.0;
pub const HM_RES: u32 = 2048;
const LEAF: f32 = 32.0;
const PATCH_N: u32 = 32;
pub const LEVELS: usize = 9;
const MAX_PATCHES: usize = 4096;
pub const PATCH_TRIS: u32 = PATCH_N * PATCH_N * 2;

pub struct Terrain {
    pub world: f32,
    pub res: u32,
    pub height_view: TextureView,
    pub normal_view: TextureView,
    pub data_view: TextureView,
    _textures: Vec<Texture>,
    heights: Vec<f32>,
    data: Vec<u8>,
    index_buf: Buffer,
    index_count: u32,
    patch_buf: Buffer,
    pub patch_count: u32,
    pub shell_count: u32,
    ranges: [f32; LEVELS],
    pipe: Option<RenderPipeline>,
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
                storage_texture_entry(2, TextureFormat::Rgba16Float),
                storage_texture_entry(3, TextureFormat::Rgba8Unorm),
            ],
        );
        let bg = bind_group(
            device,
            "terrain gen",
            &layout,
            &[
                params_buf.as_entire_binding(),
                BindingResource::TextureView(&height_view),
                BindingResource::TextureView(&normal_view),
                BindingResource::TextureView(&data_view),
            ],
        );
        let module = shader(device, "terrain gen", &[NOISE_WGSL, include_str!("shaders/terrain_gen.wgsl")]);
        let pl = pipeline_layout(device, &[&layout]);
        let pipe = compute_pipeline(device, "terrain gen", &pl, &module, "main");

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
        let patch_buf = buffer(device, "terrain patches", (MAX_PATCHES * 16) as u64, BufferUsages::VERTEX | BufferUsages::COPY_DST);

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
            index_buf,
            index_count: idx.len() as u32,
            patch_buf,
            patch_count: 0,
            shell_count: 0,
            ranges,
            pipe: None,
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
        let make = |shadow: bool| {
            let targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::ALL })];
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(if shadow { "terrain shadow" } else { "terrain" }),
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
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &targets,
                    })
                },
                multiview_mask: None,
                cache: None,
            })
        };
        self.pipe = Some(make(false));
        self.pipe_shadow = Some(make(true));

        let targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::ALL })];
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

    /// Pick quadtree patches around `cam`; patches reaching beyond `shell_from` metres
    /// are also listed (after the regular ones) for the far canopy shell.
    pub fn select(&mut self, queue: &Queue, cam: Vec2, shell_from: f32) {
        let mut patches: Vec<[f32; 4]> = Vec::with_capacity(1024);
        self.select_node(0.0, 0.0, LEVELS - 1, cam, &mut patches);
        patches.truncate(MAX_PATCHES / 2);
        self.patch_count = patches.len() as u32;
        let shell: Vec<[f32; 4]> = patches
            .iter()
            .filter(|q| {
                let far = Vec2::new((cam.x - q[0]).abs().max((cam.x - q[0] - q[2]).abs()), (cam.y - q[1]).abs().max((cam.y - q[1] - q[2]).abs()));
                far.length() > shell_from
            })
            .copied()
            .collect();
        self.shell_count = shell.len() as u32;
        patches.extend(shell);
        queue.write_buffer(&self.patch_buf, 0, cast_slice(&patches));
    }

    pub fn draw_canopy(&self, pass: &mut RenderPass<'_>) {
        if self.shell_count == 0 {
            return;
        }
        pass.set_pipeline(self.pipe_canopy.as_ref().unwrap());
        pass.set_vertex_buffer(0, self.patch_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, self.patch_count..self.patch_count + self.shell_count);
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

    pub fn draw(&self, pass: &mut RenderPass<'_>, shadow: bool) {
        let pipe = if shadow { &self.pipe_shadow } else { &self.pipe };
        pass.set_pipeline(pipe.as_ref().unwrap());
        pass.set_vertex_buffer(0, self.patch_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.patch_count);
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
}
