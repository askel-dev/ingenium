//! Frame orchestration: uniforms, cascaded sun shadows, the GPU-driven passes, sky,
//! bloom, tonemapping and GPU timing / stats readback.

use crate::bloom::Bloom;
use crate::camera::Camera;
use crate::clouds::Clouds;
use crate::forest::{Forest, ViewSetup, DRAWS};
use crate::grass::Grass;
use crate::leaftex;
use crate::sky::{Sky, SkyParams};
use crate::ssao::Ssao;
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
/// Sun illuminance above the atmosphere, in the renderer's light units.
const SUN_ILLUM: Vec3 = Vec3::new(3.4, 3.4, 3.4);
/// Light on flat ground (sun + sky) at which a preset's exposure applies unchanged; the post
/// pass opens the exposure up partway as the light falls below it.
const EXPOSURE_KEY: f32 = 2.6;
const BLOOM: f32 = 0.05;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: Mat4,
    inv_view_proj_rot: Mat4,
    shadow_vp: [Mat4; 2],
    cam_pos: Vec4,
    sun_dir: Vec4,
    sun_color: Vec4,
    atmos: Vec4,
    fog: Vec4,
    wind: Vec4,
    screen: Vec4,
    world: Vec4,
    cascade: Vec4,
    lod: Vec4,
    debug: Vec4,
    clouds: Vec4,
    cloud_type: Vec4,
    cloud_shadow: Vec4,
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
    /// Aerosol density (1 = clear air); thickens the glow around the sun and the horizon haze.
    haze: f32,
    /// 0..1: greys the sky light under a cloud deck (the clouds themselves shade the sun).
    overcast: f32,
    /// Gain on the sky's ambient light for the ground and trees. Above 1 it stands in for the
    /// light bounced off sunlit terrain and foliage, which is not modelled.
    ambient: f32,
    /// Ground haze, an aerosol layer added to the atmosphere: extinction per metre at sea
    /// level and its falloff per metre of height.
    fog_density: f32,
    fog_falloff: f32,
    /// Cloud layer: base and top altitude (m), coverage 0..1, density, and the range of cloud
    /// types the weather map picks from (0 flat stratus .. 1 towering cumulus).
    cloud_base: f32,
    cloud_top: f32,
    cloud_cover: f32,
    cloud_density: f32,
    cloud_type: (f32, f32),
    exposure: f32,
    wind: f32,
}

pub fn atmosphere(preset: usize) -> Atmosphere {
    match preset {
        0 => Atmosphere {
            name: "morning",
            sun_elev: 13.0,
            sun_azim: 105.0,
            haze: 1.6,
            overcast: 0.0,
            ambient: 2.4,
            fog_density: 0.00085,
            fog_falloff: 0.0075,
            cloud_base: 1400.0,
            cloud_top: 3400.0,
            cloud_cover: 0.38,
            cloud_density: 0.9,
            cloud_type: (0.1, 0.6),
            exposure: 1.0,
            wind: 0.5,
        },
        2 => Atmosphere {
            name: "golden hour",
            sun_elev: 7.0,
            sun_azim: 250.0,
            haze: 1.3,
            overcast: 0.0,
            ambient: 2.4,
            fog_density: 0.0005,
            fog_falloff: 0.006,
            cloud_base: 1600.0,
            cloud_top: 4200.0,
            cloud_cover: 0.42,
            cloud_density: 1.0,
            cloud_type: (0.3, 1.0),
            exposure: 1.0,
            wind: 0.8,
        },
        3 => Atmosphere {
            name: "overcast",
            sun_elev: 45.0,
            sun_azim: 160.0,
            haze: 2.5,
            overcast: 1.0,
            ambient: 1.5,
            fog_density: 0.0008,
            fog_falloff: 0.005,
            cloud_base: 1200.0,
            cloud_top: 3400.0,
            cloud_cover: 0.9,
            cloud_density: 0.7,
            cloud_type: (0.05, 0.5),
            exposure: 1.0,
            wind: 1.6,
        },
        _ => Atmosphere {
            name: "midday",
            sun_elev: 50.0,
            sun_azim: 150.0,
            haze: 1.0,
            overcast: 0.0,
            ambient: 1.7,
            fog_density: 0.00032,
            fog_falloff: 0.006,
            cloud_base: 1600.0,
            cloud_top: 4400.0,
            cloud_cover: 0.5,
            cloud_density: 1.0,
            cloud_type: (0.45, 1.0),
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
    /// Degrees added to the preset's sun elevation (`[` / `]`).
    pub sun_shift: f32,
    /// Screen-space ambient occlusion (`O`).
    pub ssao: bool,
}

#[derive(Default, Clone)]
pub struct Stats {
    /// Whole-frame GPU time of the latest timed frame.
    pub gpu_ms: f32,
    /// Light on flat ground from the sun and sky (what the exposure adapts to).
    pub ground_light: f32,
    pub visible: [u32; DRAWS],
    pub grass: u32,
    pub tris: u64,
    pub patches: u32,
}

// GPU timestamps of a timed frame, in submission order: the start of the frame, then the
// end of each section in GPU_SECTIONS.
const TS_START: u32 = 0;
const TS_CULL: u32 = 1;
const TS_CLOUDS: u32 = 2;
const TS_SHADOWS: u32 = 3;
const TS_MAIN: u32 = 4;
const TS_SSAO: u32 = 5;
const TS_BLOOM: u32 = 6;
const TS_POST: u32 = 7;
const TIMESTAMPS: u32 = 8;
/// What each timed section covers. A section runs from the end of the one before it, so
/// the atmosphere tables count with the clouds and the near cascade with the shadows.
pub const GPU_SECTIONS: [&str; TIMESTAMPS as usize - 1] = ["cull+grass", "sky+clouds", "shadows", "main", "ssao", "bloom", "post"];

/// GPU time of one frame, from its timestamps.
#[derive(Clone, Copy, Debug)]
pub struct GpuTiming {
    /// The frame the timestamps were taken in.
    pub frame: u64,
    /// Start of the culling pass to the end of the tonemap pass: the whole frame.
    pub total_ms: f32,
    /// Start of the culling pass to the end of the main pass, the span the timer covered
    /// before SSAO, bloom and post were timed (to compare with older measurements).
    pub through_main_ms: Option<f32>,
    /// Time each section of GPU_SECTIONS adds after the ones before it finished. None if
    /// its timestamp was not taken (SSAO off) or came back invalid; its time then counts
    /// with the next section.
    pub sections: [Option<f32>; TIMESTAMPS as usize - 1],
    /// Every timestamp taken came back valid and in submission order, so the sections add
    /// up to the total.
    pub ordered: bool,
}

impl GpuTiming {
    /// `ts`: the raw timestamps, `written`: a bit per timestamp the frame took,
    /// `period`: nanoseconds per tick. None if the frame start or end is unusable.
    fn parse(frame: u64, ts: &[u64], written: u32, period: f32) -> Option<GpuTiming> {
        let ms = |a: u64, b: u64| ((b - a) as f64 * period as f64 / 1.0e6) as f32;
        let start = ts[TS_START as usize];
        if written & (1 << TS_START) == 0 || start == 0 || start == u64::MAX {
            return None;
        }
        // A timestamp from this frame lies after its start and within a minute of it (slow
        // enough for a software rasterizer). One that was not written (stale, from an earlier
        // frame, or zero) or an error value does not.
        let taken = |i: usize| written & (1 << i) != 0;
        let valid = |i: usize| taken(i) && ts[i] != u64::MAX && ts[i] >= start && ms(start, ts[i]) < 60_000.0;
        let end = TS_POST as usize;
        if !valid(end) {
            return None;
        }
        let mut sections = [None; TIMESTAMPS as usize - 1];
        let mut ordered = true;
        let mut done = start;
        for i in 1..=end {
            if !valid(i) {
                ordered &= !taken(i);
                continue;
            }
            ordered &= ts[i] >= done;
            sections[i - 1] = Some(ms(done, ts[i].max(done)));
            done = done.max(ts[i]);
        }
        let main = TS_MAIN as usize;
        Some(GpuTiming { frame, total_ms: ms(start, ts[end]), through_main_ms: valid(main).then(|| ms(start, ts[main])), sections, ordered })
    }
}

struct Readback {
    buf: Buffer,
    /// The timestamps of the frame that fills this slot (None without timestamp support),
    /// resolved into `resolve` and copied into `buf` one frame later.
    queries: Option<QuerySet>,
    resolve: Buffer,
    /// Filled by frame `frame`, waiting for the next frame to resolve its timestamps.
    pending: bool,
    frame: u64,
    /// A bit per timestamp that frame took.
    written: u32,
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
    post_sampler: Sampler,
    sky: Sky,
    bloom: Bloom,
    ssao: Ssao,
    pub terrain: Terrain,
    pub forest: Forest,
    grass: Grass,
    clouds: Clouds,
    timestamp_period: f32,
    readbacks: Vec<Readback>,
    /// Timed frames read back since the last `take_timings`.
    timings: Vec<GpuTiming>,
    frame_index: u64,
    pub stats: Stats,
    frozen: Option<Frozen>,
    pub settings: Settings,
    _leaf: Texture,
    _dummies: [Texture; 3],
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
    // read back by the SSAO pass
    let depth = tex("depth", DEPTH, msaa, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    let hdr = tex("hdr", HDR, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    Targets { color_msaa, depth, hdr }
}

fn post_bind_group(device: &Device, bgl: &BindGroupLayout, hdr: &TextureView, buf: &Buffer, bloom: &Bloom, sampler: &Sampler, sky: &Sky, ssao: &Ssao) -> BindGroup {
    bind_group(
        device,
        "post",
        bgl,
        &[
            BindingResource::TextureView(hdr),
            buf.as_entire_binding(),
            BindingResource::TextureView(bloom.output()),
            BindingResource::Sampler(sampler),
            sky.light_buf.as_entire_binding(),
            BindingResource::TextureView(ssao.output()),
        ],
    )
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
                uniform_entry(9, all),
                texture_entry(10, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(11, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(12, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(13, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D3),
                texture_entry(14, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D3),
            ],
        );
        let clouds = Clouds::new(device, queue, &global_bgl, width, height);
        let sky = Sky::new(device, &global_bgl, &clouds.shadow_view);
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
        // stand-ins for the passes that write the cloud shadow map and the aerial volume
        let dummy = |label: &str, dimension: TextureDimension, format: TextureFormat| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let dummy_2d = dummy("dummy cloud shadow", TextureDimension::D2, TextureFormat::Rgba8Unorm);
        let dummy_3d = dummy("dummy aerial", TextureDimension::D3, HDR);
        let (dummy_2d_view, dummy_3d_view) = (dummy_2d.create_view(&Default::default()), dummy_3d.create_view(&Default::default()));

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
        let make_global = |shadow_view: &TextureView, cloud_shadow: &TextureView, ap_scat: &TextureView, ap_trans: &TextureView| {
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
                    sky.light_buf.as_entire_binding(),
                    BindingResource::TextureView(&sky.trans_view),
                    BindingResource::TextureView(&sky.sky_view),
                    BindingResource::TextureView(cloud_shadow),
                    BindingResource::TextureView(ap_scat),
                    BindingResource::TextureView(ap_trans),
                ],
            )
        };
        let global_bg = make_global(&shadow_array, &clouds.shadow_view, &sky.ap_scat_view, &sky.ap_trans_view);
        // for the passes that write the sun shadow map, the cloud shadow map or the aerial volume
        let global_bg_shadow = make_global(&dummy_view, &dummy_2d_view, &dummy_3d_view, &dummy_3d_view);

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
            layout: Some(&pipeline_layout(device, &[&global_bgl, &clouds.sample_bgl])),
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
                texture_entry(2, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                sampler_entry(3, ShaderStages::FRAGMENT, SamplerBindingType::Filtering),
                uniform_entry(4, ShaderStages::FRAGMENT),
                texture_entry(5, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
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
        let bloom = Bloom::new(device, HDR, &targets.hdr, width, height);
        let ssao = Ssao::new(device, &global_bgl, &targets.depth, msaa, width, height);
        let post_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("post"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let post_bg = post_bind_group(device, &post_bgl, &targets.hdr, &post_buf, &bloom, &post_sampler, &sky, &ssao);

        let readbacks = (0..3)
            .map(|_| Readback {
                buf: buffer(device, "stats readback", 512, BufferUsages::COPY_DST | BufferUsages::MAP_READ),
                queries: timestamps.then(|| device.create_query_set(&QuerySetDescriptor { label: Some("timestamps"), ty: QueryType::Timestamp, count: TIMESTAMPS })),
                resolve: buffer(device, "query resolve", 256, BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC),
                pending: false,
                frame: 0,
                written: 0,
                state: Arc::new(AtomicU32::new(0)),
            })
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
            post_sampler,
            sky,
            bloom,
            ssao,
            terrain,
            forest,
            grass,
            clouds,
            timestamp_period: queue.get_timestamp_period(),
            readbacks,
            timings: Vec::new(),
            frame_index: 0,
            stats: Stats::default(),
            frozen: None,
            settings: Settings { lod_tint: false, freeze: false, grass: true, atmos: 1, sun_shift: 0.0, ssao: true },
            _leaf: leaf_tex,
            _dummies: [dummy_shadow, dummy_2d, dummy_3d],
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.targets = make_targets(device, self.width, self.height, self.msaa);
        self.clouds.resize(device, self.width, self.height);
        self.bloom.resize(device, HDR, &self.targets.hdr, self.width, self.height);
        self.ssao.resize(device, &self.targets.depth, self.width, self.height);
        self.post_bg = post_bind_group(device, &self.post_bgl, &self.targets.hdr, &self.post_buf, &self.bloom, &self.post_sampler, &self.sky, &self.ssao);
    }

    pub fn render(&mut self, device: &Device, queue: &Queue, out: &TextureView, cam: &Camera, time: f32) {
        let aspect = self.width as f32 / self.height as f32;
        let atm = atmosphere(self.settings.atmos);
        let el = (atm.sun_elev + self.settings.sun_shift).clamp(-10.0, 89.0).to_radians();
        let az = atm.sun_azim.to_radians();
        let sun = Vec3::new(el.cos() * az.sin(), el.sin(), -el.cos() * az.cos()).normalize();

        let view_proj = cam.proj(aspect) * cam.view();
        // Sky rays come from a rotation-only inverse: unprojecting through the full matrix and
        // then subtracting a camera position kilometres from the origin loses most of the f32
        // precision of a point 0.1 m away, so the reconstructed ray (sun, clouds) jittered as
        // the camera moved.
        let mut view_rot = cam.view();
        view_rot.w_axis = Vec4::W;
        let vp_rot = cam.proj(aspect) * view_rot;
        let inv_view_proj_rot = vp_rot.inverse();
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
            inv_view_proj_rot,
            shadow_vp: [svp0, svp1],
            cam_pos: cam.pos.extend(time),
            sun_dir: sun.extend(0.0),
            sun_color: SUN_ILLUM.extend(atm.ambient),
            atmos: Vec4::new(atm.haze, atm.overcast, 0.0, 0.0),
            fog: Vec4::new(atm.fog_density, atm.fog_falloff, 0.0, 0.0),
            wind: Vec4::new(wind_dir.x, wind_dir.y, atm.wind, 0.0),
            screen: Vec4::new(self.width as f32, self.height as f32, 1.0 / self.width as f32, 1.0 / self.height as f32),
            world: Vec4::new(WORLD, HM_RES as f32, WORLD / HM_RES as f32, SHADOW_RES as f32),
            cascade: Vec4::new(CASCADES[0].1, CASCADES[1].1, texel0, texel1),
            lod: Vec4::new(LOD_DIST.x, LOD_DIST.y, LOD_DIST.z, MAX_TREE_DIST),
            debug: Vec4::new(if self.settings.lod_tint { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
            clouds: Vec4::new(atm.cloud_base, atm.cloud_top, atm.cloud_cover, atm.cloud_density),
            cloud_type: Vec4::new(atm.cloud_type.0, atm.cloud_type.1, atm.wind * 9.0, 0.0),
            cloud_shadow: Clouds::shadow_params(cam.pos, sun, atm.cloud_base, atm.cloud_cover),
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
        queue.write_buffer(&self.post_buf, 0, cast_slice(&[atm.exposure, BLOOM, 1.0 / self.bloom.levels() as f32, EXPOSURE_KEY]));

        self.terrain.select(queue, Vec2::new(ref_pos.x, ref_pos.z), MAX_TREE_DIST * 0.6);
        self.forest.update_view(queue, 0, &ViewSetup { planes: cull_planes, ref_pos, max_dist: MAX_TREE_DIST, lod: LOD_DIST });
        for (i, svp) in [svp0, svp1].iter().enumerate() {
            let mut lod = LOD_DIST;
            lod.w = [0.0f32, 3.0][i]; // far cascade casts from the cheap hull LOD only
            self.forest.update_view(queue, 1 + i, &ViewSetup { planes: frustum_planes(*svp), ref_pos, max_dist: CASCADES[i].1 + 60.0, lod });
        }
        self.grass.update(queue, cull_planes, ref_pos);
        self.clouds.update(queue, vp_rot, cam, sun, fu.cloud_shadow, fu.clouds);

        // Stats readback (non-blocking ring): a frame whose slot is free is timed and read back.
        let slot = (self.frame_index % self.readbacks.len() as u64) as usize;
        let staged = !self.readbacks[slot].pending && self.readbacks[slot].state.load(Ordering::Acquire) == 0;
        let qs = if staged { self.readbacks[slot].queries.clone() } else { None };
        let written = match (&qs, self.settings.ssao) {
            (None, _) => 0,
            (Some(_), true) => (1 << TIMESTAMPS) - 1,
            (Some(_), false) => ((1 << TIMESTAMPS) - 1) & !(1 << TS_SSAO),
        };
        let pass_end = |i: u32| qs.as_ref().map(|q| RenderPassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(i) });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let tw = qs.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(TS_START), end_of_pass_write_index: Some(TS_CULL) });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("cull"), timestamp_writes: tw });
            for v in 0..3 {
                self.forest.cull(&mut cp, &self.global_bg, v);
            }
            if self.settings.grass {
                self.grass.generate(&mut cp, &self.global_bg);
            }
        }
        // the atmosphere tables; timed together with the clouds
        let sky_params = SkyParams { sun_dir: sun, sun_illum: SUN_ILLUM, haze: atm.haze, cam_alt_m: cam.pos.y, overcast: atm.overcast, ground_albedo: 0.12, fog: fu.fog };
        self.sky.update(queue, &mut enc, &sky_params);
        {
            let tw = qs.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(TS_CLOUDS) });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("clouds"), timestamp_writes: tw });
            self.clouds.render_shadow(&mut cp, &self.global_bg_shadow);
            self.sky.aerial(&mut cp, &self.global_bg_shadow);
            self.clouds.march(&mut cp, &self.global_bg);
        }
        for c in 0..2 {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.shadow_layers[c],
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: if c == 1 { pass_end(TS_SHADOWS) } else { None },
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
                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: pass_end(TS_MAIN),
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
            rp.set_bind_group(1, self.clouds.output(), &[]);
            rp.draw(0..3, 0..1);
        }
        if self.settings.ssao {
            self.ssao.record(&mut enc, &self.global_bg, pass_end(TS_SSAO));
        } else {
            // not timed: it draws nothing, and an empty pass may not take a timestamp on Metal
            self.ssao.clear(&mut enc);
        }
        self.bloom.record(&mut enc, pass_end(TS_BLOOM));
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: pass_end(TS_POST),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.post_pipe);
            rp.set_bind_group(0, &self.post_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // This frame's counts are copied now, but its timestamps are resolved by the next
        // frame. Resolved in the frame's own command buffer, the timestamps of the last
        // passes (SSAO, bloom, post) came back stale on Metal, probably because the resolve
        // does not wait for those passes to write them. A frame later they are final: that
        // frame's main pass overwrites the image those passes read, so it runs after them,
        // and in capture mode the GPU is idle between frames anyway.
        if staged {
            let rb = &self.readbacks[slot];
            enc.copy_buffer_to_buffer(&self.forest.views[0].args_buf, 0, &rb.buf, 256, (DRAWS * 20) as u64);
            enc.copy_buffer_to_buffer(&self.grass.args_buf, 0, &rb.buf, 496, 16);
            enc.copy_buffer_to_buffer(&self.sky.light_buf, 0, &rb.buf, 64, 64);
        }
        let mut resolved = Vec::new();
        for (k, rb) in self.readbacks.iter_mut().enumerate() {
            if rb.pending {
                if let Some(q) = &rb.queries {
                    enc.resolve_query_set(q, 0..TIMESTAMPS, &rb.resolve, 0);
                    enc.copy_buffer_to_buffer(&rb.resolve, 0, &rb.buf, 0, TIMESTAMPS as u64 * 8);
                }
                rb.pending = false;
                resolved.push(k);
            }
        }
        if staged {
            let rb = &mut self.readbacks[slot];
            rb.pending = true;
            rb.frame = self.frame_index;
            rb.written = written;
        }
        queue.submit([enc.finish()]);
        for k in resolved {
            let rb = &self.readbacks[k];
            rb.state.store(1, Ordering::Release);
            let st = rb.state.clone();
            rb.buf.map_async(MapMode::Read, .., move |r| st.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release));
        }
        let _ = device.poll(PollType::Poll);
        let mut ready: Vec<usize> = (0..self.readbacks.len()).filter(|&k| self.readbacks[k].state.load(Ordering::Acquire) == 2).collect();
        ready.sort_by_key(|&k| self.readbacks[k].frame);
        for k in ready {
            let rb = &self.readbacks[k];
            {
                let view = rb.buf.get_mapped_range(..).expect("mapped range");
                let words: &[u32] = cast_slice(&view);
                if rb.queries.is_some() {
                    let ts: Vec<u64> = (0..TIMESTAMPS as usize).map(|i| words[i * 2] as u64 | ((words[i * 2 + 1] as u64) << 32)).collect();
                    if let Some(t) = GpuTiming::parse(rb.frame, &ts, rb.written, self.timestamp_period) {
                        self.stats.gpu_ms = t.total_ms;
                        if self.timings.len() >= 1024 {
                            self.timings.drain(..512); // nobody is collecting them
                        }
                        self.timings.push(t);
                    }
                }
                self.stats.ground_light = f32::from_bits(words[16 + 15]);
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
        self.frame_index += 1;
    }

    /// GPU timings read back since the last call, oldest first. A frame's timings arrive two
    /// frames after it is rendered.
    pub fn take_timings(&mut self) -> Vec<GpuTiming> {
        std::mem::take(&mut self.timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: u32 = (1 << TIMESTAMPS) - 1;
    // 1000 ns per tick, so 1000 ticks are a millisecond
    const PERIOD: f32 = 1000.0;
    const IN_ORDER: [u64; 8] = [10_000, 10_100, 10_300, 10_600, 11_600, 11_700, 11_750, 11_800];

    fn close(a: Option<f32>, b: f32) -> bool {
        a.is_some_and(|a| (a - b).abs() < 1e-4)
    }

    #[test]
    fn sections_add_up_when_in_order() {
        let t = GpuTiming::parse(7, &IN_ORDER, ALL, PERIOD).unwrap();
        assert!(t.ordered);
        assert_eq!(t.frame, 7);
        assert!(close(Some(t.total_ms), 1.8));
        assert!(close(t.through_main_ms, 1.6));
        for (s, want) in t.sections.iter().zip([0.1, 0.2, 0.3, 1.0, 0.1, 0.05, 0.05]) {
            assert!(close(*s, want), "{s:?} != {want}");
        }
    }

    #[test]
    fn stale_end_of_frame_is_rejected() {
        let mut ts = IN_ORDER;
        ts[TS_POST as usize] = 9_000; // left over from an earlier frame
        assert!(GpuTiming::parse(0, &ts, ALL, PERIOD).is_none());
        ts[TS_POST as usize] = u64::MAX; // Metal's error value
        assert!(GpuTiming::parse(0, &ts, ALL, PERIOD).is_none());
        let mut ts = IN_ORDER;
        ts[TS_START as usize] = 0;
        assert!(GpuTiming::parse(0, &ts, ALL, PERIOD).is_none());
    }

    #[test]
    fn untimed_ssao_counts_with_bloom() {
        let mut ts = IN_ORDER;
        ts[TS_SSAO as usize] = 3; // not written this frame
        let t = GpuTiming::parse(0, &ts, ALL & !(1 << TS_SSAO), PERIOD).unwrap();
        assert!(t.ordered);
        assert_eq!(t.sections[TS_SSAO as usize - 1], None);
        assert!(close(t.sections[TS_BLOOM as usize - 1], 0.15));
        assert!(close(Some(t.total_ms), 1.8));
    }

    #[test]
    fn invalid_timestamp_inside_the_frame_is_flagged() {
        let mut ts = IN_ORDER;
        ts[TS_SHADOWS as usize] = 9_000; // stale
        let t = GpuTiming::parse(0, &ts, ALL, PERIOD).unwrap();
        assert!(!t.ordered);
        assert_eq!(t.sections[TS_SHADOWS as usize - 1], None);
        assert!(close(t.sections[TS_MAIN as usize - 1], 1.3)); // shadows + main
    }

    #[test]
    fn overlapping_passes_are_flagged_but_still_add_up() {
        // SSAO finishes after bloom (they don't depend on each other)
        let mut ts = IN_ORDER;
        ts[TS_SSAO as usize] = 11_760;
        let t = GpuTiming::parse(0, &ts, ALL, PERIOD).unwrap();
        assert!(!t.ordered);
        assert!(close(t.sections[TS_BLOOM as usize - 1], 0.0));
        let sum: f32 = t.sections.iter().flatten().sum();
        assert!((sum - t.total_ms).abs() < 1e-4);
    }
}
