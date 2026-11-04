//! Physically based sky (Hillaire 2020): transmittance and multiple-scattering tables rebuilt
//! when the haze changes, a sky-view table and the sun / ambient light rebuilt every frame
//! (`shaders/atmosphere.wgsl`). Everything else reads the sky, the sun colour and the ambient
//! light from these, so terrain, trees, fog and clouds all agree with the sky.

use crate::util::*;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

const FMT: TextureFormat = TextureFormat::Rgba16Float;
const TRANS: (u32, u32) = (256, 64);
const MS: u32 = 32;
const SKYVIEW: (u32, u32) = (192, 108);
/// Size of the `Sky` uniform the irradiance pass fills (4 x vec4).
const SKY_OUT: u64 = 64;
/// Aerial perspective froxels: across, down, depth slices (AP_DIST in globals.wgsl).
const AERIAL: (u32, u32, u32) = (64, 64, 32);
/// The shadowed haze near the camera: across, down, depth slices over NEAR_DIST (globals.wgsl).
const NEAR: (u32, u32, u32) = (160, 90, 64);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AtmoUniform {
    sun_dir: Vec4,
    sun_illum: Vec4,
    params: Vec4,
    fog: Vec4,
}

pub struct SkyParams {
    pub sun_dir: Vec3,
    /// Sun illuminance above the atmosphere.
    pub sun_illum: Vec3,
    pub haze: f32,
    pub cam_alt_m: f32,
    pub overcast: f32,
    pub ground_albedo: f32,
    /// Ground haze: extinction per metre at height `fog.z`, falloff per metre.
    pub fog: Vec4,
}

pub struct Sky {
    _textures: Vec<Texture>,
    pub trans_view: TextureView,
    pub sky_view: TextureView,
    uniform: Buffer,
    out: Buffer,
    /// The sun / ambient light as a uniform for bind group 0.
    pub light_buf: Buffer,
    pipes: [ComputePipeline; 4],
    bgs: [BindGroup; 4],
    /// Aerial perspective volume: light scattered towards the camera, and transmittance.
    pub ap_scat_view: TextureView,
    pub ap_trans_view: TextureView,
    ap_pipe: ComputePipeline,
    ap_bg: BindGroup,
    /// The sunlight the shadows take out of the haze near the camera (`aerial_near`).
    pub ap_shade_view: TextureView,
    near_pipe: ComputePipeline,
    near_bg: BindGroup,
    built_for: Option<(f32, f32)>,
}

fn lut(device: &Device, label: &str, w: u32, h: u32) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: FMT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn volume(device: &Device, label: &str, (w, h, d): (u32, u32, u32)) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: w, height: h, depth_or_array_layers: d },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: FMT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

impl Sky {
    /// `global_bgl` is bind group 0 for the aerial perspective pass, which also reads the cloud
    /// shadow map.
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, cloud_shadow: &TextureView) -> Sky {
        let trans = lut(device, "transmittance lut", TRANS.0, TRANS.1);
        let ms = lut(device, "multi-scatter lut", MS, MS);
        let sky = lut(device, "sky-view lut", SKYVIEW.0, SKYVIEW.1);
        let (trans_view, ms_view, sky_view) = (trans.create_view(&Default::default()), ms.create_view(&Default::default()), sky.create_view(&Default::default()));
        let uniform = buffer(device, "atmosphere", std::mem::size_of::<AtmoUniform>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let out = buffer(device, "sky light out", SKY_OUT, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let light_buf = buffer(device, "sky light", SKY_OUT, BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC);
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("atmosphere lut"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let module = shader(device, "atmosphere", &[NOISE_WGSL, ATMO_WGSL, include_str!("shaders/atmosphere.wgsl")]);
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = [pipe("transmittance"), pipe("multiscatter"), pipe("skyview"), pipe("irradiance")];
        let tex = |v| BindingResource::TextureView(v);
        let entries: [Vec<(u32, BindingResource)>; 4] = [
            vec![(0, uniform.as_entire_binding()), (5, tex(&trans_view))],
            vec![(0, uniform.as_entire_binding()), (1, BindingResource::Sampler(&sampler)), (2, tex(&trans_view)), (6, tex(&ms_view))],
            vec![(0, uniform.as_entire_binding()), (1, BindingResource::Sampler(&sampler)), (2, tex(&trans_view)), (3, tex(&ms_view)), (7, tex(&sky_view))],
            vec![(0, uniform.as_entire_binding()), (1, BindingResource::Sampler(&sampler)), (2, tex(&trans_view)), (4, tex(&sky_view)), (8, out.as_entire_binding())],
        ];
        let bgs = std::array::from_fn(|i| {
            let e: Vec<BindGroupEntry> = entries[i].iter().map(|(b, r)| BindGroupEntry { binding: *b, resource: r.clone() }).collect();
            device.create_bind_group(&BindGroupDescriptor { label: Some("atmosphere"), layout: &pipes[i].get_bind_group_layout(0), entries: &e })
        });

        let (ap_scat, ap_trans) = (volume(device, "aerial scattering", AERIAL), volume(device, "aerial transmittance", AERIAL));
        let (ap_scat_view, ap_trans_view) = (ap_scat.create_view(&Default::default()), ap_trans.create_view(&Default::default()));
        let ap_shade = volume(device, "aerial shade", NEAR);
        let ap_shade_view = ap_shade.create_view(&Default::default());
        let cs = ShaderStages::COMPUTE;
        let float = TextureSampleType::Float { filterable: true };
        let storage_3d = |binding| BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: FMT, view_dimension: TextureViewDimension::D3 },
            count: None,
        };
        let ap_bgl = bgl(
            device,
            "aerial",
            &[texture_entry(0, cs, float, TextureViewDimension::D2), texture_entry(1, cs, float, TextureViewDimension::D2), storage_3d(2), storage_3d(3)],
        );
        let ap_module = shader(device, "aerial", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/aerial.wgsl")]);
        let ap_pipe = compute_pipeline(device, "aerial", &pipeline_layout(device, &[global_bgl, &ap_bgl]), &ap_module, "aerial");
        let ap_bg = bind_group(device, "aerial", &ap_bgl, &[tex(&ms_view), tex(cloud_shadow), tex(&ap_scat_view), tex(&ap_trans_view)]);
        let near_bgl = bgl(device, "aerial near", &[storage_3d(4)]);
        let near_pipe = compute_pipeline(device, "aerial near", &pipeline_layout(device, &[global_bgl, &near_bgl]), &ap_module, "aerial_near");
        let near_bg = device.create_bind_group(&BindGroupDescriptor { label: Some("aerial near"), layout: &near_bgl, entries: &[BindGroupEntry { binding: 4, resource: tex(&ap_shade_view) }] });
        Sky {
            _textures: vec![trans, ms, sky, ap_scat, ap_trans, ap_shade],
            trans_view,
            sky_view,
            uniform,
            out,
            light_buf,
            pipes,
            bgs,
            ap_scat_view,
            ap_trans_view,
            ap_pipe,
            ap_bg,
            ap_shade_view,
            near_pipe,
            near_bg,
            built_for: None,
        }
    }

    /// Record the table updates for this frame. Run before anything that reads the sky.
    pub fn update(&mut self, queue: &Queue, enc: &mut CommandEncoder, p: &SkyParams) {
        let u = AtmoUniform {
            sun_dir: p.sun_dir.extend(0.0),
            sun_illum: p.sun_illum.extend(p.haze),
            params: Vec4::new(6360.0 + p.cam_alt_m.max(1.0) * 0.001, p.overcast, p.ground_albedo, 0.0),
            fog: p.fog,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let rebuild = self.built_for != Some((p.haze, p.ground_albedo));
        self.built_for = Some((p.haze, p.ground_albedo));
        {
            let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("atmosphere"), timestamp_writes: None });
            let mut run = |i: usize, x: u32, y: u32| {
                cp.set_pipeline(&self.pipes[i]);
                cp.set_bind_group(0, &self.bgs[i], &[]);
                cp.dispatch_workgroups(x, y, 1);
            };
            if rebuild {
                run(0, TRANS.0 / 8, TRANS.1 / 8);
                run(1, MS / 8, MS / 8);
            }
            run(2, SKYVIEW.0.div_ceil(8), SKYVIEW.1.div_ceil(8));
            run(3, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.light_buf, 0, SKY_OUT);
    }

    /// Rebuild the aerial perspective volume. Run after `update` and the cloud shadow map;
    /// `global_bg` must not hold the volume itself.
    pub fn aerial(&self, cp: &mut ComputePass, global_bg: &BindGroup) {
        cp.set_pipeline(&self.ap_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.ap_bg, &[]);
        cp.dispatch_workgroups(AERIAL.0.div_ceil(8), AERIAL.1.div_ceil(8), 1);
    }

    /// Rebuild the shadowed haze near the camera. Run after the shadow maps; `global_bg` must
    /// hold them, but not the volume itself.
    pub fn aerial_near(&self, cp: &mut ComputePass, global_bg: &BindGroup) {
        cp.set_pipeline(&self.near_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.near_bg, &[]);
        cp.dispatch_workgroups(NEAR.0.div_ceil(8), NEAR.1.div_ceil(8), 1);
    }
}
