//! Frame orchestration: uniforms, cascaded sun shadows, the GPU-driven passes, sky,
//! bloom, tonemapping and GPU timing / stats readback.

use crate::bloom::Bloom;
use crate::camera::Camera;
use crate::clouds::Clouds;
use crate::forest::{Forest, TreePass, ViewSetup, CULL_PLANES, DRAWS, LODS, OCCLUDER_LODS};
use crate::height_shadow::HeightShadow;
use crate::hiz::Hiz;
use crate::exposure::Exposure;
use crate::grass::Grass;
use crate::undergrowth::Undergrowth;
use crate::leaftex;
use crate::sky::{Sky, SkyParams};
use crate::ssilvb::Ssilvb;
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
/// The stats readback: timestamps at 0, the sky light at 64, the grass draw args at 128, the
/// tree draw args from 256.
const READBACK_SIZE: u64 = (256 + DRAWS as u64 * 20).next_multiple_of(256);

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
    sun_grid: Vec4,
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
    /// light bounced off sunlit terrain and foliage further away than the screen-space indirect
    /// light (ssilvb.wgsl) reaches.
    ambient: f32,
    /// Ground haze, an aerosol layer added to the atmosphere: extinction per metre at sea
    /// level and its falloff per metre of height.
    fog_density: f32,
    fog_falloff: f32,
    /// Haze in the air among the trees, under the canopy (extinction per metre): moisture and
    /// dust, what the shafts of light show up in.
    forest_haze: f32,
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
            ambient: 1.6,
            fog_density: 0.00085,
            fog_falloff: 0.0075,
            forest_haze: 0.0025,
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
            ambient: 1.6,
            fog_density: 0.0005,
            fog_falloff: 0.006,
            forest_haze: 0.0016,
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
            ambient: 1.25,
            fog_density: 0.0008,
            fog_falloff: 0.005,
            forest_haze: 0.0015,
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
            ambient: 1.2,
            fog_density: 0.00032,
            fog_falloff: 0.006,
            forest_haze: 0.0011,
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
    /// Screen-space indirect light, ambient occlusion and contact shadows (`O`).
    pub ssilvb: bool,
    /// Lay down the trees' depth before shading them, so each covered sample is shaded once
    /// (`--no-prepass` turns it off to compare).
    pub prepass: bool,
    /// Draw the terrain and the near trees first, then cull the distant trees hidden behind
    /// them (`--no-occlusion` turns it off to compare). Needs the prepass.
    pub occlusion: bool,
    /// Show one term of the lighting instead of the image (`--debug-view`): 1 the ambient
    /// light's visibility (SSILVB), 2 the contact shadows, 3 the indirect light added (0.5 where
    /// it doubles a pixel's light), 4 the share of the light that is direct sun, 5 the albedo,
    /// 6 the share of the haze in front that lies in shadow (red) and the haze itself (green).
    pub debug_view: u32,
}

#[derive(Clone)]
pub struct Stats {
    /// Whole-frame GPU time of the latest timed frame.
    pub gpu_ms: f32,
    /// Light on flat ground from the sun and sky (what the exposure adapts to).
    pub ground_light: f32,
    pub visible: [u32; DRAWS],
    pub grass: u32,
    pub tris: u64,
    /// Terrain patches picked by the LOD selection, then drawn by the camera, the near and
    /// the far shadow cascade.
    pub patches: [u32; 4],
}

impl Default for Stats {
    fn default() -> Stats {
        Stats { gpu_ms: 0.0, ground_light: 0.0, visible: [0; DRAWS], grass: 0, tris: 0, patches: [0; 4] }
    }
}

// GPU timestamps of a timed frame, in submission order: the start of the frame, then the
// end of each section in GPU_SECTIONS.
const TS_START: u32 = 0;
const TS_CULL: u32 = 1;
const TS_CLOUDS: u32 = 2;
const TS_SHADOWS: u32 = 3;
const TS_MAIN: u32 = 4;
const TS_GI: u32 = 5;
const TS_BLOOM: u32 = 6;
const TS_POST: u32 = 7;
const TIMESTAMPS: u32 = 8;
/// What each timed section covers. A section runs from the end of the one before it, so
/// the terrain shadow map (when the sun moved) and the atmosphere tables count with the
/// clouds, and the near cascade with the shadows, which end with the shadowed haze
/// (aerial_near). The first covers the culling, the grass and, with occlusion culling, the
/// occluder pass, the depth pyramid and the second culling pass.
pub const GPU_SECTIONS: [&str; TIMESTAMPS as usize - 1] = ["cull+occluders", "sky+clouds", "shadows+haze", "main", "ssilvb", "bloom", "post"];

/// GPU time of one frame, from its timestamps.
#[derive(Clone, Copy, Debug)]
pub struct GpuTiming {
    /// The frame the timestamps were taken in.
    pub frame: u64,
    /// Start of the culling pass to the end of the tonemap pass: the whole frame.
    pub total_ms: f32,
    /// Start of the culling pass to the end of the main pass, the span the timer covered
    /// before the screen-space lighting, bloom and post were timed (to compare with older measurements).
    pub through_main_ms: Option<f32>,
    /// Time each section of GPU_SECTIONS adds after the ones before it finished. None if
    /// its timestamp was not taken (SSILVB off) or came back invalid; its time then counts
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
    /// Albedo and the direct sun's share of the light (util::AUX), multisampled and resolved.
    aux_msaa: Option<TextureView>,
    aux: TextureView,
}

impl Targets {
    /// The main pass's colour attachments: (view, resolve target) for the HDR colour and aux.
    fn scene(&self) -> [(&TextureView, Option<&TextureView>); 2] {
        match (&self.color_msaa, &self.aux_msaa) {
            (Some(c), Some(a)) => [(c, Some(&self.hdr)), (a, Some(&self.aux))],
            _ => [(&self.hdr, None), (&self.aux, None)],
        }
    }
}

struct Frozen {
    planes: [Vec4; 6],
    pos: Vec3,
}

pub struct Renderer {
    /// The size the scene is rendered at: the output size times `scale`.
    pub width: u32,
    pub height: u32,
    /// The size of the image `render` writes (the window or the capture).
    pub out_width: u32,
    pub out_height: u32,
    /// Render scale: the post pass upscales the scene to the output size when below 1.
    pub scale: f32,
    msaa: u32,
    frame_buf: Buffer,
    global_bg: BindGroup,
    global_bg_shadow: BindGroup,
    /// With the shadow maps but without the near haze volume, for the pass that writes it.
    global_bg_haze: BindGroup,
    pass_bufs: Vec<Buffer>,
    pass_bgs: Vec<BindGroup>,
    _shadow_tex: Texture,
    shadow_layers: Vec<TextureView>,
    targets: Targets,
    sky_pipe: RenderPipeline,
    post_pipe: RenderPipeline,
    exposure: Exposure,
    post_bgl: BindGroupLayout,
    post_bg: BindGroup,
    post_buf: Buffer,
    post_sampler: Sampler,
    sky: Sky,
    bloom: Bloom,
    ssilvb: Ssilvb,
    hiz: Hiz,
    pub terrain: Terrain,
    pub forest: Forest,
    grass: Grass,
    undergrowth: Undergrowth,
    clouds: Clouds,
    height_shadow: HeightShadow,
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
    // read back by the screen-space lighting (ssilvb.rs)
    let depth = tex("depth", DEPTH, msaa, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    let hdr = tex("hdr", HDR, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    let aux_msaa = (msaa > 1).then(|| tex("aux msaa", AUX, msaa, TextureUsages::RENDER_ATTACHMENT));
    let aux = tex("aux", AUX, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING);
    Targets { color_msaa, depth, hdr, aux_msaa, aux }
}

#[allow(clippy::too_many_arguments)]
fn post_bind_group(device: &Device, bgl: &BindGroupLayout, targets: &Targets, buf: &Buffer, bloom: &Bloom, sampler: &Sampler, sky: &Sky, gi: &Ssilvb, exposure: &Exposure) -> BindGroup {
    bind_group(
        device,
        "post",
        bgl,
        &[
            BindingResource::TextureView(&targets.hdr),
            buf.as_entire_binding(),
            BindingResource::TextureView(bloom.output()),
            BindingResource::Sampler(sampler),
            sky.light_buf.as_entire_binding(),
            BindingResource::TextureView(gi.output()),
            exposure.state.as_entire_binding(),
            BindingResource::TextureView(gi.contact()),
            BindingResource::TextureView(&targets.aux),
        ],
    )
}

/// Corners of the camera frustum between the view depths `near` and `far`.
fn slice_corners(cam: &Camera, aspect: f32, near: f32, far: f32) -> [Vec3; 8] {
    let ty = (cam.fov_y * 0.5).tan();
    let tx = ty * aspect;
    let inv_view = cam.view().inverse();
    let mut corners = [Vec3::ZERO; 8];
    for (i, z) in [near, far].into_iter().enumerate() {
        for (j, (sx, sy)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].into_iter().enumerate() {
            corners[i * 4 + j] = inv_view.transform_point3(Vec3::new(sx * tx * z, sy * ty * z, -z));
        }
    }
    corners
}

/// The sun's view: a rotation only, looking along the sunlight.
fn light_view(sun: Vec3) -> Mat4 {
    let up = if sun.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    glam::camera::rh::view::look_to_mat4(Vec3::ZERO, -sun, up)
}

/// Stable cascade: bounding sphere of the frustum slice, snapped to shadow texels.
fn cascade_matrix(cam: &Camera, aspect: f32, near: f32, far: f32, sun: Vec3) -> (Mat4, f32) {
    let corners = slice_corners(cam, aspect, near, far);
    let center = corners.iter().fold(Vec3::ZERO, |a, c| a + *c) / 8.0;
    let mut radius = corners.iter().map(|c| c.distance(center)).fold(0.0, f32::max);
    radius = (radius * 4.0).ceil() / 4.0;
    let texel = 2.0 * radius / SHADOW_RES as f32;
    let light_view = light_view(sun);
    let mut c = light_view.transform_point3(center);
    c.x = (c.x / texel).floor() * texel;
    c.y = (c.y / texel).floor() * texel;
    let proj = glam::camera::rh::proj::directx::orthographic(c.x - radius, c.x + radius, c.y - radius, c.y + radius, -c.z - radius - 1500.0, -c.z + radius + 20.0);
    (proj * light_view, texel)
}

/// Convex hull of a few points, counter-clockwise (monotone chain).
fn convex_hull(mut pts: Vec<Vec2>) -> Vec<Vec2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let half = |pts: &mut dyn Iterator<Item = Vec2>| {
        let mut h: Vec<Vec2> = Vec::new();
        for p in pts {
            while h.len() >= 2 && (h[h.len() - 1] - h[h.len() - 2]).perp_dot(p - h[h.len() - 2]) <= 0.0 {
                h.pop();
            }
            h.push(p);
        }
        h.pop();
        h
    };
    let mut hull = half(&mut pts.iter().copied());
    hull.extend(half(&mut pts.iter().rev().copied()));
    hull
}

/// Culling planes for the trees a cascade draws. A tree can only shade the part of the view
/// the cascade covers if it stands between that part and the sun, but the cascade's box is a
/// square around a sphere that holds it: most trees in the box cast their shadows off screen,
/// behind the camera or onto ground that is shaded by the other cascade. So the planes are the
/// outline of the view, between the distances `d0` and `d1` from the camera, as the sun sees
/// it (swept towards the sun), plus one plane past its far side from the sun, all moved out by
/// `margin`. `None` if the outline doesn't fit in the planes (a degenerate view).
fn caster_planes(cam: &Camera, aspect: f32, d0: f32, d1: f32, sun: Vec3, margin: f32) -> Option<[Vec4; CULL_PLANES]> {
    // a point at distance d inside the frustum is at least d / (length of a corner ray) deep
    let ty = (cam.fov_y * 0.5).tan();
    let tx = ty * aspect;
    let near = (d0 / (1.0 + tx * tx + ty * ty).sqrt()).max(0.1);
    let corners = slice_corners(cam, aspect, near, d1);
    let lv = light_view(sun);
    let hull = convex_hull(corners.iter().map(|c| lv.transform_point3(*c).truncate()).collect());
    if hull.len() < 3 || hull.len() > CULL_PLANES - 1 {
        return None;
    }
    let inv = lv.inverse();
    let mut planes = [Vec4::new(0.0, 0.0, 0.0, 1.0); CULL_PLANES];
    for (i, &a) in hull.iter().enumerate() {
        let e = hull[(i + 1) % hull.len()] - a;
        if e.length() < 1e-4 {
            continue; // leaving a plane out only keeps more trees
        }
        let n = Vec2::new(-e.y, e.x).normalize(); // inwards: the hull runs counter-clockwise
        planes[i] = inv.transform_vector3(n.extend(0.0)).extend(margin - n.dot(a));
    }
    let back = corners.iter().map(|c| c.dot(sun)).fold(f32::INFINITY, f32::min);
    planes[hull.len()] = sun.extend(margin - back);
    Some(planes)
}

/// A frustum's planes, padded with planes that keep everything.
fn pad_planes(p: [Vec4; 6]) -> [Vec4; CULL_PLANES] {
    let mut out = [Vec4::new(0.0, 0.0, 0.0, 1.0); CULL_PLANES];
    out[..6].copy_from_slice(&p);
    out
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
                texture_entry(15, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(16, all, TextureSampleType::Float { filterable: true }, TextureViewDimension::D3),
            ],
        );
        let clouds = Clouds::new(device, queue, &global_bgl, width, height);
        let height_shadow = HeightShadow::new(device, &terrain);
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
        let make_global = |shadow_view: &TextureView, cloud_shadow: &TextureView, ap_scat: &TextureView, ap_trans: &TextureView, ap_shade: &TextureView| {
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
                    BindingResource::TextureView(&height_shadow.view),
                    BindingResource::TextureView(ap_shade),
                ],
            )
        };
        let global_bg = make_global(&shadow_array, &clouds.shadow_view, &sky.ap_scat_view, &sky.ap_trans_view, &sky.ap_shade_view);
        // for the passes that write the sun shadow map, the cloud shadow map or the aerial volume
        let global_bg_shadow = make_global(&dummy_view, &dummy_2d_view, &dummy_3d_view, &dummy_3d_view, &dummy_3d_view);
        let global_bg_haze = make_global(&shadow_array, &clouds.shadow_view, &sky.ap_scat_view, &sky.ap_trans_view, &dummy_3d_view);

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
        let mut forest = Forest::new(device, queue, &global_bgl, &global_bg, &pass_bgl, HDR, msaa, WORLD);
        let undergrowth = Undergrowth::new(device, &global_bgl, &forest);
        let grass = Grass::new(device, &global_bgl, &pass_bgl, HDR, msaa);

        // sky
        let module = shader(device, "sky", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/sky.wgsl")]);
        let targets_hdr = scene_targets(HDR, true);
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
                texture_entry(0, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                uniform_entry(1, ShaderStages::FRAGMENT),
                texture_entry(2, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                sampler_entry(3, ShaderStages::FRAGMENT, SamplerBindingType::Filtering),
                uniform_entry(4, ShaderStages::FRAGMENT),
                texture_entry(5, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                storage_entry(6, ShaderStages::FRAGMENT, true),
                texture_entry(7, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                texture_entry(8, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
            ],
        );
        let post_buf = buffer(device, "post", 32, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
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
        let ssilvb = Ssilvb::new(device, &global_bgl, &targets.depth, &targets.hdr, msaa, width, height);
        let hiz = Hiz::new(device, &targets.depth, msaa, width, height);
        forest.set_hiz(device, hiz.view(), hiz.info());
        let post_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("post"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let exposure = Exposure::new(device, &targets.hdr);
        let post_bg = post_bind_group(device, &post_bgl, &targets, &post_buf, &bloom, &post_sampler, &sky, &ssilvb, &exposure);

        let readbacks = (0..3)
            .map(|_| Readback {
                buf: buffer(device, "stats readback", READBACK_SIZE, BufferUsages::COPY_DST | BufferUsages::MAP_READ),
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
            out_width: width,
            out_height: height,
            scale: 1.0,
            msaa,
            frame_buf,
            global_bg,
            global_bg_shadow,
            global_bg_haze,
            pass_bufs,
            pass_bgs,
            _shadow_tex: shadow_tex,
            shadow_layers,
            targets,
            sky_pipe,
            post_pipe,
            exposure,
            post_bgl,
            post_bg,
            post_buf,
            post_sampler,
            sky,
            bloom,
            ssilvb,
            hiz,
            terrain,
            forest,
            grass,
            undergrowth,
            clouds,
            height_shadow,
            timestamp_period: queue.get_timestamp_period(),
            readbacks,
            timings: Vec::new(),
            frame_index: 0,
            stats: Stats::default(),
            frozen: None,
            settings: Settings { lod_tint: false, freeze: false, grass: true, atmos: 1, sun_shift: 0.0, ssilvb: true, prepass: true, occlusion: true, debug_view: 0 },
            _leaf: leaf_tex,
            _dummies: [dummy_shadow, dummy_2d, dummy_3d],
        }
    }

    /// The output size changed (window resize).
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.out_width = width.max(1);
        self.out_height = height.max(1);
        self.rebuild_targets(device);
    }

    /// Render the scene at `scale` times the output size (clamped to 0.25..1).
    pub fn set_scale(&mut self, device: &Device, scale: f32) {
        self.scale = scale.clamp(0.25, 1.0);
        self.rebuild_targets(device);
    }

    fn rebuild_targets(&mut self, device: &Device) {
        let scaled = |n: u32| ((n as f32 * self.scale).round() as u32).clamp(1, n);
        self.width = scaled(self.out_width);
        self.height = scaled(self.out_height);
        self.targets = make_targets(device, self.width, self.height, self.msaa);
        self.clouds.resize(device, self.width, self.height);
        self.bloom.resize(device, HDR, &self.targets.hdr, self.width, self.height);
        self.ssilvb.resize(device, &self.targets.depth, &self.targets.hdr, self.width, self.height);
        self.hiz.resize(device, &self.targets.depth, self.width, self.height);
        self.forest.set_hiz(device, self.hiz.view(), self.hiz.info());
        self.exposure.set_target(device, &self.targets.hdr);
        self.post_bg = post_bind_group(device, &self.post_bgl, &self.targets, &self.post_buf, &self.bloom, &self.post_sampler, &self.sky, &self.ssilvb, &self.exposure);
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
            fog: Vec4::new(atm.fog_density, atm.fog_falloff, 0.0, atm.forest_haze),
            wind: Vec4::new(wind_dir.x, wind_dir.y, atm.wind, 0.0),
            screen: Vec4::new(self.width as f32, self.height as f32, 1.0 / self.width as f32, 1.0 / self.height as f32),
            world: Vec4::new(WORLD, HM_RES as f32, WORLD / HM_RES as f32, SHADOW_RES as f32),
            cascade: Vec4::new(CASCADES[0].1, CASCADES[1].1, texel0, texel1),
            lod: Vec4::new(LOD_DIST.x, LOD_DIST.y, LOD_DIST.z, MAX_TREE_DIST),
            debug: Vec4::new(if self.settings.lod_tint { 1.0 } else { 0.0 }, (self.settings.debug_view == 6) as u32 as f32, 0.0, 0.0),
            clouds: Vec4::new(atm.cloud_base, atm.cloud_top, atm.cloud_cover, atm.cloud_density),
            cloud_type: Vec4::new(atm.cloud_type.0, atm.cloud_type.1, atm.wind * 9.0, 0.0),
            cloud_shadow: Clouds::shadow_params(cam.pos, sun, atm.cloud_base, atm.cloud_cover),
            sun_grid: self.height_shadow.grid(sun),
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
        let upscale = (self.width, self.height) != (self.out_width, self.out_height);
        queue.write_buffer(&self.post_buf, 0, cast_slice(&[atm.exposure, BLOOM, 1.0 / self.bloom.levels() as f32, EXPOSURE_KEY, upscale as u32 as f32, self.settings.debug_view as f32, 0.0, 0.0]));

        let shadow_planes = [frustum_planes(svp0), frustum_planes(svp1)];
        self.terrain.select(queue, Vec2::new(ref_pos.x, ref_pos.z), MAX_TREE_DIST * 0.6, &[cull_planes, shadow_planes[0], shadow_planes[1]]);
        // Occlusion culling tests against the depth the current camera sees, so it is off while
        // culling is frozen at another camera.
        let occlusion = self.settings.occlusion && self.settings.prepass && self.frozen.is_none();
        let proj = cam.proj(aspect);
        self.forest.update_view(
            queue,
            0,
            &ViewSetup { planes: pad_planes(cull_planes), ref_pos, max_dist: MAX_TREE_DIST, lod: LOD_DIST, occlusion: occlusion.then(|| (cam.view(), Vec2::new(proj.x_axis.x, proj.y_axis.y))) },
        );
        // what each cascade shades (shadow_at in globals.wgsl, and the haze in aerial_near in
        // aerial.wgsl): the near one up to its far distance, the far one from where the blend
        // between them starts
        let receivers = [(0.0, CASCADES[0].1), (0.8 * CASCADES[0].1, CASCADES[1].1)];
        for (i, planes) in shadow_planes.iter().enumerate() {
            let mut lod = LOD_DIST;
            lod.w = [0.0f32, 3.0][i]; // the far cascade casts impostors only
            // a metre for wind sway and the normal offset, plus the filter's reach
            let margin = 1.0 + 3.0 * [texel0, texel1][i];
            let planes = caster_planes(cam, aspect, receivers[i].0, receivers[i].1, sun, margin).unwrap_or(pad_planes(*planes));
            self.forest.update_view(queue, 1 + i, &ViewSetup { planes, ref_pos, max_dist: CASCADES[i].1 + 60.0, lod, occlusion: None });
        }
        self.grass.update(queue, cull_planes, ref_pos);
        self.undergrowth.update(queue, cull_planes, ref_pos);
        self.clouds.update(queue, vp_rot, cam, sun, fu.cloud_shadow, fu.clouds);

        // Stats readback (non-blocking ring): a frame whose slot is free is timed and read back.
        let slot = (self.frame_index % self.readbacks.len() as u64) as usize;
        let staged = !self.readbacks[slot].pending && self.readbacks[slot].state.load(Ordering::Acquire) == 0;
        let qs = if staged { self.readbacks[slot].queries.clone() } else { None };
        let written = match (&qs, self.settings.ssilvb) {
            (None, _) => 0,
            (Some(_), true) => (1 << TIMESTAMPS) - 1,
            (Some(_), false) => ((1 << TIMESTAMPS) - 1) & !(1 << TS_GI),
        };
        let pass_end = |i: u32| qs.as_ref().map(|q| RenderPassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(i) });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let end = (!occlusion).then_some(TS_CULL);
            let tw = qs.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(TS_START), end_of_pass_write_index: end });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("cull"), timestamp_writes: tw });
            for v in 0..3 {
                self.forest.cull(&mut cp, &self.global_bg, v);
            }
            if self.settings.grass {
                self.grass.generate(&mut cp, &self.global_bg);
                self.undergrowth.generate(&mut cp, &self.global_bg);
            }
        }
        let scene = self.targets.scene();
        if occlusion {
            // Under the canopy almost every distant tree is hidden behind the near ones, yet
            // they are most of the triangles. So the terrain and the near trees lay down their
            // depth first, a depth pyramid is built from it, and the distant trees are culled
            // against that. Nothing is carried over from the last frame, so a tree is never late.
            // It all happens before the sky and the shadows, which don't depend on it, so the
            // GPU can overlap them with it instead of the main pass waiting on it.
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("occluders"),
                // only there for the leaves' alpha-to-coverage: nothing is written to them
                color_attachments: &scene.map(|(view, _)| {
                    Some(RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Discard } })
                }),
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // the bind group without the shadow, cloud and haze maps: depth needs none of them,
            // and binding them would make this wait for the passes that write them
            rp.set_bind_group(0, &self.global_bg_shadow, &[]);
            rp.set_bind_group(1, &self.pass_bgs[0], &[]);
            self.terrain.draw_depth(&mut rp);
            rp.set_bind_group(2, &self.pass_bgs[0], &[]);
            self.forest.draw(&mut rp, 0, TreePass::Depth, 0..OCCLUDER_LODS);
            drop(rp);
            let tw = qs.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(TS_CULL) });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("occlusion cull"), timestamp_writes: tw });
            self.hiz.build(&mut cp);
            self.forest.cull_occluded(&mut cp, &self.global_bg_shadow);
        }
        // the terrain's shadow over the whole world when the sun has moved, and the atmosphere
        // tables; timed together with the clouds
        self.height_shadow.update(queue, &mut enc, sun);
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, &self.global_bg_shadow, &[]);
            rp.set_bind_group(1, &self.pass_bgs[1 + c], &[]);
            self.terrain.draw(&mut rp, 1 + c);
            rp.set_bind_group(2, &self.pass_bgs[1 + c], &[]);
            self.forest.draw(&mut rp, 1 + c, TreePass::Shadow, 0..LODS);
        }
        {
            // the haze in the trees' shadows; timed with the shadows
            let tw = qs.as_ref().map(|q| ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(TS_SHADOWS) });
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("shadowed haze"), timestamp_writes: tw });
            self.sky.aerial_near(&mut cp, &self.global_bg_haze);
        }
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &scene.map(|(view, resolve_target)| {
                    Some(RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target,
                        ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: if resolve_target.is_some() { StoreOp::Discard } else { StoreOp::Store } },
                    })
                }),
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(Operations { load: if occlusion { LoadOp::Load } else { LoadOp::Clear(0.0) }, store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: pass_end(TS_MAIN),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, &self.global_bg, &[]);
            rp.set_bind_group(2, &self.pass_bgs[0], &[]);
            // Leaves and impostors are cut out by alpha-to-coverage, which turns off the GPU's
            // hidden surface removal: every fragment that passes the depth test so far is
            // shaded, in whatever order the culling appended the trees. Laying down their depth
            // first (cheap: a texture read or four) lets the colour pass shade only the nearest
            // surface. (with occlusion culling the near LODs' depth is already there)
            if self.settings.prepass {
                self.forest.draw(&mut rp, 0, TreePass::Depth, if occlusion { OCCLUDER_LODS..LODS } else { 0..LODS });
            }
            rp.set_bind_group(1, &self.pass_bgs[0], &[]);
            self.terrain.draw(&mut rp, 0);
            self.terrain.draw_canopy(&mut rp);
            if self.settings.prepass {
                self.forest.draw(&mut rp, 0, TreePass::ColorEqual, 0..LODS);
            } else {
                self.forest.draw(&mut rp, 0, TreePass::Color, 0..LODS);
            }
            if self.settings.grass {
                self.undergrowth.draw(&mut rp, self.forest.color_pipeline());
                self.grass.draw(&mut rp);
            }
            rp.set_pipeline(&self.sky_pipe);
            rp.set_bind_group(0, &self.global_bg, &[]);
            rp.set_bind_group(1, self.clouds.output(), &[]);
            rp.draw(0..3, 0..1);
        }
        if self.settings.ssilvb {
            self.ssilvb.record(&mut enc, &self.global_bg, pass_end(TS_GI));
        } else {
            // not timed: it draws nothing, and an empty pass may not take a timestamp on Metal
            self.ssilvb.clear(&mut enc);
        }
        self.bloom.record(&mut enc, pass_end(TS_BLOOM));
        self.exposure.record(queue, &mut enc, time);
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
        // passes (SSILVB, bloom, post) came back stale on Metal, probably because the resolve
        // does not wait for those passes to write them. A frame later they are final: that
        // frame's main pass overwrites the image those passes read, so it runs after them,
        // and in capture mode the GPU is idle between frames anyway.
        if staged {
            let rb = &self.readbacks[slot];
            enc.copy_buffer_to_buffer(&self.forest.views[0].args_buf, 0, &rb.buf, 256, (DRAWS * 20) as u64);
            enc.copy_buffer_to_buffer(&self.grass.args_buf, 0, &rb.buf, 128, 16);
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
                let mut tris = self.terrain.drawn(0) as u64 * PATCH_TRIS as u64;
                for d in 0..DRAWS {
                    let n = words[64 + d * 5 + 1];
                    self.stats.visible[d] = n;
                    tris += n as u64 * self.forest.draws[d].tris as u64;
                }
                self.stats.grass = if self.settings.grass { words[32 + 1] } else { 0 };
                tris += self.stats.grass as u64 * 5;
                self.stats.tris = tris;
                self.stats.patches = [self.terrain.selected, self.terrain.drawn(0), self.terrain.drawn(1), self.terrain.drawn(2)];
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
    fn hull_drops_inner_points_and_runs_counter_clockwise() {
        let pts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0), (1.0, 0.0)];
        let hull = convex_hull(pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect());
        assert_eq!(hull.len(), 4);
        let area: f32 = (0..4).map(|i| hull[i].perp_dot(hull[(i + 1) % 4])).sum::<f32>() * 0.5;
        assert!((area - 4.0).abs() < 1e-5, "signed area {area}");
    }

    fn inside(planes: &[Vec4; CULL_PLANES], p: Vec3) -> bool {
        planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= 0.0)
    }

    #[test]
    fn caster_planes_keep_every_tree_that_can_shade_the_view() {
        let cam = Camera { pos: Vec3::new(100.0, 50.0, 200.0), yaw: 0.7, pitch: -0.2, fov_y: 0.95 };
        let aspect = 16.0 / 9.0;
        let (ty, inv_view) = ((cam.fov_y * 0.5).tan(), cam.view().inverse());
        let mut seed = 1u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        // the sun ahead of the camera and low, behind it, and overhead
        for sun in [Vec3::new(0.6, 0.15, -0.5), Vec3::new(-0.5, 0.6, 0.5), Vec3::new(0.0, 1.0, 0.001)] {
            let sun = sun.normalize();
            for (d0, d1) in [(0.0, 50.0), (40.0, 260.0)] {
                let planes = caster_planes(&cam, aspect, d0, d1, sun, 0.01).expect("planes");
                for _ in 0..500 {
                    // a point the cascade shades: inside the view, between d0 and d1 away
                    let dir = Vec3::new((rnd() * 2.0 - 1.0) * ty * aspect, (rnd() * 2.0 - 1.0) * ty, -1.0).normalize();
                    let p = inv_view.transform_point3(dir * (d0 + (d1 - d0) * rnd()).max(0.2));
                    for t in [0.0, 5.0, 30.0, 120.0] {
                        assert!(inside(&planes, p + sun * t), "sun {sun} d {d0}..{d1}: lost a caster {t} m towards the sun");
                    }
                }
            }
        }
        // the sun ahead: nothing behind the camera or far off to the side casts into the view
        let sun = Vec3::new(0.6, 0.15, -0.5).normalize();
        let planes = caster_planes(&cam, aspect, 0.0, 50.0, sun, 1.0).unwrap();
        assert!(!inside(&planes, cam.pos - cam.forward() * 40.0));
        assert!(!inside(&planes, cam.pos + cam.forward() * 25.0 + cam.right() * 80.0));
        assert!(inside(&planes, cam.pos + cam.forward() * 25.0));
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
    fn untimed_ssilvb_counts_with_bloom() {
        let mut ts = IN_ORDER;
        ts[TS_GI as usize] = 3; // not written this frame
        let t = GpuTiming::parse(0, &ts, ALL & !(1 << TS_GI), PERIOD).unwrap();
        assert!(t.ordered);
        assert_eq!(t.sections[TS_GI as usize - 1], None);
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
        // SSILVB finishes after bloom (they don't depend on each other)
        let mut ts = IN_ORDER;
        ts[TS_GI as usize] = 11_760;
        let t = GpuTiming::parse(0, &ts, ALL, PERIOD).unwrap();
        assert!(!t.ordered);
        assert!(close(t.sections[TS_BLOOM as usize - 1], 0.0));
        let sum: f32 = t.sections.iter().flatten().sum();
        assert!((sum - t.total_ms).abs() < 1e-4);
    }
}
