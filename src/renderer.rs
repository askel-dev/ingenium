//! Frame orchestration: uniforms, cascaded sun shadows, the GPU-driven passes, sky,
//! tonemapping and GPU timing / stats readback.

use crate::camera::Camera;
use crate::forest::{Forest, ViewSetup, DRAWS};
use crate::grass::Grass;
use crate::leaftex;
use crate::terrain::{Terrain, HM_RES, PATCH_TRIS, WORLD};
use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use wgpu::*;

pub const HDR: TextureFormat = TextureFormat::Rgba16Float;
const DEPTH: TextureFormat = TextureFormat::Depth32Float;
const SHADOW_RES: u32 = 2048;
const CASCADES: [(f32, f32); 2] = [(0.1, 50.0), (35.0, 260.0)];
const LOD_DIST: Vec4 = Vec4::new(40.0, 140.0, 500.0, 0.0);
const MAX_TREE_DIST: f32 = 1600.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: Mat4,
    inv_view_proj: Mat4,
    shadow_vp: [Mat4; 2],
    cam_pos: Vec4,
    sun_dir: Vec4,
    sun_color: Vec4,
    sky_zenith: Vec4,
    sky_horizon: Vec4,
    fog: Vec4,
    wind: Vec4,
    screen: Vec4,
    world: Vec4,
    cascade: Vec4,
    lod: Vec4,
    debug: Vec4,
    morph: [Vec4; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PassUniform {
    view_proj: Mat4,
    ref_cam: Vec4,
    params: Vec4,
}

pub struct Atmosphere {
    pub name: &'static str,
    sun_elev: f32,
    sun_azim: f32,
    sun: Vec3,
    ambient: f32,
    zenith: Vec3,
    horizon: Vec3,
    fog_density: f32,
    fog_falloff: f32,
    clouds: f32,
    exposure: f32,
    wind: f32,
}

pub fn atmosphere(preset: usize) -> Atmosphere {
    match preset {
        0 => Atmosphere {
            name: "morning",
            sun_elev: 13.0,
            sun_azim: 105.0,
            sun: Vec3::new(1.0, 0.80, 0.58) * 3.0,
            ambient: 0.9,
            zenith: Vec3::new(0.10, 0.20, 0.42),
            horizon: Vec3::new(0.52, 0.55, 0.58),
            fog_density: 0.00085,
            fog_falloff: 0.0075,
            clouds: 0.3,
            exposure: 1.05,
            wind: 0.5,
        },
        2 => Atmosphere {
            name: "golden hour",
            sun_elev: 7.0,
            sun_azim: 250.0,
            sun: Vec3::new(1.0, 0.62, 0.33) * 3.1,
            ambient: 0.75,
            zenith: Vec3::new(0.09, 0.15, 0.32),
            horizon: Vec3::new(0.80, 0.58, 0.42),
            fog_density: 0.0005,
            fog_falloff: 0.006,
            clouds: 0.45,
            exposure: 1.1,
            wind: 0.8,
        },
        3 => Atmosphere {
            name: "overcast",
            sun_elev: 45.0,
            sun_azim: 160.0,
            sun: Vec3::new(0.92, 0.94, 0.98) * 0.55,
            ambient: 1.7,
            zenith: Vec3::new(0.40, 0.42, 0.46),
            horizon: Vec3::new(0.50, 0.52, 0.55),
            fog_density: 0.0011,
            fog_falloff: 0.005,
            clouds: 1.0,
            exposure: 1.25,
            wind: 1.6,
        },
        _ => Atmosphere {
            name: "midday",
            sun_elev: 50.0,
            sun_azim: 150.0,
            sun: Vec3::new(1.0, 0.96, 0.90) * 2.8,
            ambient: 0.7,
            zenith: Vec3::new(0.07, 0.19, 0.52),
            horizon: Vec3::new(0.55, 0.66, 0.80),
            fog_density: 0.00032,
            fog_falloff: 0.006,
            clouds: 0.5,
            exposure: 1.0,
            wind: 1.0,
        },
    }
}

pub struct Settings {
    pub lod_tint: bool,
    pub freeze: bool,
    pub grass: bool,
    pub atmos: usize,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub gpu_ms: f32,
    pub pass_ms: [f32; 4], // cull+grass compute, shadows, main, post
    pub visible: [u32; DRAWS],
    pub grass: u32,
    pub tris: u64,
    pub patches: u32,
}

struct Readback {
    buf: Buffer,
    state: Arc<AtomicU32>, // 0 idle, 1 in flight, 2 ready
}

struct Targets {
    color_msaa: Option<TextureView>,
    depth: TextureView,
    hdr: TextureView,
}

struct Frozen {
    planes: [Vec4; 6],
    pos: Vec3,
}

pub struct Renderer {
    pub width: u32,
    pub height: u32,
    msaa: u32,
    frame_buf: Buffer,
    global_bg: BindGroup,
    global_bg_shadow: BindGroup,
    pass_bufs: Vec<Buffer>,
    pass_bgs: Vec<BindGroup>,
    _shadow_tex: Texture,
    shadow_layers: Vec<TextureView>,
    targets: Targets,
    sky_pipe: RenderPipeline,
    post_pipe: RenderPipeline,
    post_bgl: BindGroupLayout,
    post_bg: BindGroup,
    post_buf: Buffer,
    pub terrain: Terrain,
    pub forest: Forest,
    grass: Grass,
    queries: Option<QuerySet>,
    query_resolve: Buffer,
    timestamp_period: f32,
    readbacks: Vec<Readback>,
    frame_index: u64,
    pub stats: Stats,
    frozen: Option<Frozen>,
    pub settings: Settings,
    _leaf: Texture,
}

fn make_targets(device: &Device, w: u32, h: u32, msaa: u32) -> Targets {
    let tex = |label: &str, format: TextureFormat, samples: u32, usage: TextureUsages| {
        device
            .create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let color_msaa = (msaa > 1).then(|| tex("color msaa", HDR, msaa, TextureUsages::RENDER_ATTACHMENT));
    let depth = tex("depth", DEPTH, msaa, TextureUsages::RENDER_ATTACHMENT);
    let hdr = tex("hdr", HDR, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    Targets { color_msaa, depth, hdr }
}

/// Stable cascade: bounding sphere of the frustum slice, snapped to shadow texels.
fn cascade_matrix(cam: &Camera, aspect: f32, near: f32, far: f32, sun: Vec3) -> (Mat4, f32) {
    let ty = (cam.fov_y * 0.5).tan();
    let tx = ty * aspect;
    let inv_view = cam.view().inverse();
    let mut corners = Vec::with_capacity(8);
    for z in [near, far] {
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            corners.push(inv_view.transform_point3(Vec3::new(sx * tx * z, sy * ty * z, -z)));
        }
    }
    let center = corners.iter().fold(Vec3::ZERO, |a, c| a + *c) / 8.0;
    let mut radius = corners.iter().map(|c| c.distance(center)).fold(0.0, f32::max);
    radius = (radius * 4.0).ceil() / 4.0;
    let texel = 2.0 * radius / SHADOW_RES as f32;
    let up = if sun.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let light_view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, -sun, up);
    let mut c = light_view.transform_point3(center);
    c.x = (c.x / texel).floor() * texel;
    c.y = (c.y / texel).floor() * texel;
    let proj = glam::camera::rh::proj::directx::orthographic(c.x - radius, c.x + radius, c.y - radius, c.y + radius, -c.z - radius - 1500.0, -c.z + radius + 20.0);
    (proj * light_view, texel)
}

impl Renderer {
    pub fn new(device: &Device, queue: &Queue, timestamps: bool, width: u32, height: u32, out_format: TextureFormat, msaa: u32) -> Renderer {
        let t0 = std::time::Instant::now();
        let terrain = Terrain::new(device, queue);
        let (leaf_tex, leaf_view) = leaftex::build(device, queue);

        let all = ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE;
        let global_bgl = bgl(
            device,
            "global",
            &[
                uniform_entry(0, all),
                texture_entry(1, all, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
                texture_entry(2, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(3, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(4, all, TextureSampleType::Depth, TextureViewDimension::D2Array),
                sampler_entry(5, all, SamplerBindingType::Comparison),
                sampler_entry(6, all, SamplerBindingType::Filtering),
                texture_entry(7, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2Array),
                sampler_entry(8, all, SamplerBindingType::Filtering),
            ],
        );
        let pass_bgl = bgl(device, "pass", &[uniform_entry(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)]);

        let frame_buf = buffer(device, "frame", std::mem::size_of::<FrameUniform>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let shadow_tex = device.create_texture(&TextureDescriptor {
            label: Some("shadow map"),
            size: Extent3d { width: SHADOW_RES, height: SHADOW_RES, depth_or_array_layers: 2 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DEPTH,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_array = shadow_tex.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
        let shadow_layers: Vec<TextureView> = (0..2)
            .map(|i| {
                shadow_tex.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let dummy_shadow = device.create_texture(&TextureDescriptor {
            label: Some("dummy shadow"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 2 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DEPTH,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_shadow.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });

        let s_shadow = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow cmp"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });
        let s_leaf = device.create_sampler(&SamplerDescriptor {
            label: Some("leaf"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let s_clamp = device.create_sampler(&SamplerDescriptor {
            label: Some("clamp"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let make_global = |shadow_view: &TextureView| {
            bind_group(
                device,
                "global",
                &global_bgl,
                &[
                    frame_buf.as_entire_binding(),
                    BindingResource::TextureView(&terrain.height_view),
                    BindingResource::TextureView(&terrain.data_view),
                    BindingResource::TextureView(&terrain.normal_view),
                    BindingResource::TextureView(shadow_view),
                    BindingResource::Sampler(&s_shadow),
                    BindingResource::Sampler(&s_leaf),
                    BindingResource::TextureView(&leaf_view),
                    BindingResource::Sampler(&s_clamp),
                ],
            )
        };
        let global_bg = make_global(&shadow_array);
        let global_bg_shadow = make_global(&dummy_view);

        let pass_bufs: Vec<Buffer> = (0..3)
            .map(|_| buffer(device, "pass", std::mem::size_of::<PassUniform>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST))
            .collect();
        let pass_bgs: Vec<BindGroup> = pass_bufs.iter().map(|b| bind_group(device, "pass", &pass_bgl, &[b.as_entire_binding()])).collect();

        // The global bind group needs a valid frame uniform before the scatter pass runs.
        let mut init = FrameUniform::zeroed();
        init.world = Vec4::new(WORLD, HM_RES as f32, WORLD / HM_RES as f32, SHADOW_RES as f32);
        queue.write_buffer(&frame_buf, 0, bytemuck::bytes_of(&init));

        let mut terrain = terrain;
        terrain.create_pipelines(device, &global_bgl, &pass_bgl, HDR, msaa);
        let forest = Forest::new(device, queue, &global_bgl, &global_bg, &pass_bgl, HDR, msaa, WORLD);
        let grass = Grass::new(device, &global_bgl, &pass_bgl, HDR, msaa);

        // sky
        let module = shader(device, "sky", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/sky.wgsl")]);
        let targets_hdr = [Some(ColorTargetState { format: HDR, blend: None, write_mask: ColorWrites::ALL })];
        let sky_pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&pipeline_layout(device, &[&global_bgl])),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state(DEPTH, false, CompareFunction::GreaterEqual)),
            multisample: MultisampleState { count: msaa, ..Default::default() },
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &targets_hdr }),
            multiview_mask: None,
            cache: None,
        });

        // post
        let post_bgl = bgl(
            device,
            "post",
            &[
                texture_entry(0, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
                uniform_entry(1, ShaderStages::FRAGMENT),
            ],
        );
        let post_buf = buffer(device, "post", 16, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let module = shader(device, "post", &[NOISE_WGSL, include_str!("shaders/post.wgsl")]);
        let out_targets = [Some(ColorTargetState { format: out_format, blend: None, write_mask: ColorWrites::ALL })];
        let post_pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("post"),
            layout: Some(&pipeline_layout(device, &[&post_bgl])),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &out_targets }),
            multiview_mask: None,
            cache: None,
        });
        let targets = make_targets(device, width, height, msaa);
        let post_bg = bind_group(device, "post", &post_bgl, &[BindingResource::TextureView(&targets.hdr), post_buf.as_entire_binding()]);

        let queries = timestamps.then(|| device.create_query_set(&QuerySetDescriptor { label: Some("timestamps"), ty: QueryType::Timestamp, count: 5 }));
        let query_resolve = buffer(device, "query resolve", 256, BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC);
        let readbacks = (0..3)
            .map(|_| Readback { buf: buffer(device, "stats readback", 512, BufferUsages::COPY_DST | BufferUsages::MAP_READ), state: Arc::new(AtomicU32::new(0)) })
            .collect();

        println!("renderer ready in {:.2}s", t0.elapsed().as_secs_f32());
        Renderer {
            width,
            height,
            msaa,
            frame_buf,
            global_bg,
            global_bg_shadow,
            pass_bufs,
            pass_bgs,
            _shadow_tex: shadow_tex,
            shadow_layers,
            targets,
            sky_pipe,
            post_pipe,
            post_bgl,
            post_bg,
            post_buf,
            terrain,
            forest,
            grass,
            queries,
            query_resolve,
            timestamp_period: queue.get_timestamp_period(),
            readbacks,
            frame_index: 0,
            stats: Stats::default(),
            frozen: None,
            settings: Settings { lod_tint: false, freeze: false, grass: true, atmos: 1 },
            _leaf: leaf_tex,
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.targets = make_targets(device, self.width, self.height, self.msaa);
        self.post_bg = bind_group(device, "post", &self.post_bgl, &[BindingResource::TextureView(&self.targets.hdr), self.post_buf.as_entire_binding()]);
    }

    pub fn render(&mut self, device: &Device, queue: &Queue, out: &TextureView, cam: &Camera, time: f32) {
        let aspect = self.width as f32 / self.height as f32;
        let atm = atmosphere(self.settings.atmos);
        let (el, az) = (atm.sun_elev.to_radians(), atm.sun_azim.to_radians());
        let sun = Vec3::new(el.cos() * az.sin(), el.sin(), -el.cos() * az.cos()).normalize();

        let view_proj = cam.proj(aspect) * cam.view();
        let (svp0, texel0) = cascade_matrix(cam, aspect, CASCADES[0].0, CASCADES[0].1, sun);
        let (svp1, texel1) = cascade_matrix(cam, aspect, CASCADES[1].0, CASCADES[1].1, sun);
        let main_planes = frustum_planes(view_proj);

        // freeze: keep culling / LOD / terrain selection from the frozen camera
        if self.settings.freeze && self.frozen.is_none() {
            self.frozen = Some(Frozen { planes: main_planes, pos: cam.pos });
        } else if !self.settings.freeze {
            self.frozen = None;
        }
        let (cull_planes, ref_pos) = match &self.frozen {
            Some(f) => (f.planes, f.pos),
            None => (main_planes, cam.pos),
        };

        let wind_dir = Vec2::new(0.8, 0.6).normalize();
        let fu = FrameUniform {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            shadow_vp: [svp0, svp1],
            cam_pos: cam.pos.extend(time),
            sun_dir: sun.extend(0.0),
            sun_color: atm.sun.extend(atm.ambient),
            sky_zenith: atm.zenith.extend(0.0),
            sky_horizon: atm.horizon.extend(0.0),
            fog: Vec4::new(atm.fog_density, atm.fog_falloff, 0.0, atm.clouds),
            wind: Vec4::new(wind_dir.x, wind_dir.y, atm.wind, 0.0),
            screen: Vec4::new(self.width as f32, self.height as f32, 1.0 / self.width as f32, 1.0 / self.height as f32),
            world: Vec4::new(WORLD, HM_RES as f32, WORLD / HM_RES as f32, SHADOW_RES as f32),
            cascade: Vec4::new(CASCADES[0].1, CASCADES[1].1, texel0, texel1),
            lod: Vec4::new(LOD_DIST.x, LOD_DIST.y, LOD_DIST.z, MAX_TREE_DIST),
            debug: Vec4::new(if self.settings.lod_tint { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
            morph: self.terrain.morph_ranges(),
        };
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&fu));
        let passes = [
            PassUniform { view_proj, ref_cam: ref_pos.extend(0.0), params: Vec4::ZERO },
            PassUniform { view_proj: svp0, ref_cam: ref_pos.extend(0.0), params: Vec4::new(1.0, 0.0, 0.0, 0.0) },
            PassUniform { view_proj: svp1, ref_cam: ref_pos.extend(0.0), params: Vec4::new(1.0, 1.0, 0.0, 0.0) },
        ];
        for (b, p) in self.pass_bufs.iter().zip(passes.iter()) {
            queue.write_buffer(b, 0, bytemuck::bytes_of(p));
        }
        queue.write_buffer(&self.post_buf, 0, cast_slice(&[atm.exposure, 0.0, 0.0, 0.0]));

        self.terrain.select(queue, Vec2::new(ref_pos.x, ref_pos.z), MAX_TREE_DIST * 0.6);
        self.forest.update_view(queue, 0, &ViewSetup { planes: cull_planes, ref_pos, max_dist: MAX_TREE_DIST, lod: LOD_DIST });
        for (i, svp) in [svp0, svp1].iter().enumerate() {
            let mut lod = LOD_DIST;
            lod.w = [0.0f32, 3.0][i]; // far cascade casts from the cheap hull LOD only
            self.forest.update_view(queue, 1 + i, &ViewSetup { planes: frustum_planes(*svp), ref_pos, max_dist: CASCADES[i].1 + 60.0, lod });
        }
        self.grass.update(queue, cull_planes, ref_pos);

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let tw = self.queries.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("cull"), timestamp_writes: tw });
            for v in 0..3 {
                self.forest.cull(&mut cp, &self.global_bg, v);
            }
            if self.settings.grass {
                self.grass.generate(&mut cp, &self.global_bg);
            }
        }
        for c in 0..2 {
            let tw = self.queries.as_ref().filter(|_| c == 1).map(|q| RenderPassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(2) });
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.shadow_layers[c],
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: tw,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, &self.global_bg_shadow, &[]);
            rp.set_bind_group(1, &self.pass_bgs[1 + c], &[]);
            self.terrain.draw(&mut rp, true);
            rp.set_bind_group(2, &self.pass_bgs[1 + c], &[]);
            self.forest.draw(&mut rp, 1 + c, true, 3);
        }
        {
            let (view, resolve) = match &self.targets.color_msaa {
                Some(m) => (m, Some(&self.targets.hdr)),
                None => (&self.targets.hdr, None),
            };
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: if resolve.is_some() { StoreOp::Discard } else { StoreOp::Store } },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.queries.as_ref().map(|q| RenderPassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(3) }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, &self.global_bg, &[]);
            rp.set_bind_group(1, &self.pass_bgs[0], &[]);
            self.terrain.draw(&mut rp, false);
            self.terrain.draw_canopy(&mut rp);
            rp.set_bind_group(2, &self.pass_bgs[0], &[]);
            self.forest.draw(&mut rp, 0, false, 3);
            if self.settings.grass {
                self.grass.draw(&mut rp);
            }
            rp.set_pipeline(&self.sky_pipe);
            rp.set_bind_group(0, &self.global_bg, &[]);
            rp.draw(0..3, 0..1);
        }
        {
            let tw = self.queries.as_ref().map(|q| RenderPassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(4) });
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: tw,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.post_pipe);
            rp.set_bind_group(0, &self.post_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // stats readback (non-blocking ring)
        let slot = (self.frame_index % self.readbacks.len() as u64) as usize;
        let staged = self.readbacks[slot].state.load(Ordering::Acquire) == 0;
        if staged {
            let rb = &self.readbacks[slot];
            if let Some(q) = &self.queries {
                enc.resolve_query_set(q, 0..5, &self.query_resolve, 0);
                enc.copy_buffer_to_buffer(&self.query_resolve, 0, &rb.buf, 0, 40);
            }
            enc.copy_buffer_to_buffer(&self.forest.views[0].args_buf, 0, &rb.buf, 256, (DRAWS * 20) as u64);
            enc.copy_buffer_to_buffer(&self.grass.args_buf, 0, &rb.buf, 496, 16);
        }
        queue.submit([enc.finish()]);
        if staged {
            let rb = &self.readbacks[slot];
            rb.state.store(1, Ordering::Release);
            let st = rb.state.clone();
            rb.buf.map_async(MapMode::Read, .., move |r| st.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release));
        }
        let _ = device.poll(PollType::Poll);
        for rb in &self.readbacks {
            if rb.state.load(Ordering::Acquire) == 2 {
                {
                    let view = rb.buf.get_mapped_range(..).expect("mapped range");
                    let words: &[u32] = cast_slice(&view);
                    if self.queries.is_some() {
                        let ts: Vec<u64> = (0..5).map(|i| words[i * 2] as u64 | ((words[i * 2 + 1] as u64) << 32)).collect();
                        if ts[..4].windows(2).all(|w| w[1] >= w[0]) && ts[3] > ts[0] {
                            let ms = |a: u64, b: u64| (b - a) as f32 * self.timestamp_period / 1.0e6;
                            self.stats.gpu_ms = ms(ts[0], ts[3]);
                            self.stats.pass_ms = [ms(ts[0], ts[1]), ms(ts[1], ts[2]), ms(ts[2], ts[3]), 0.0];
                        }
                    }
                    let mut tris = self.terrain.patch_count as u64 * PATCH_TRIS as u64;
                    for d in 0..DRAWS {
                        let n = words[64 + d * 5 + 1];
                        self.stats.visible[d] = n;
                        tris += n as u64 * self.forest.draws[d].tris as u64;
                    }
                    self.stats.grass = if self.settings.grass { words[124 + 1] } else { 0 };
                    tris += self.stats.grass as u64 * 5;
                    self.stats.tris = tris;
                    self.stats.patches = self.terrain.patch_count;
                }
                rb.buf.unmap();
                rb.state.store(0, Ordering::Release);
            }
        }
        self.frame_index += 1;
    }
}
