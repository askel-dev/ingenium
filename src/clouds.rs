//! Volumetric clouds: tiling noise volumes and a weather map baked once on the GPU, a raymarch
//! that refreshes a quarter of a reduced-resolution buffer each frame, and temporal reprojection
//! for the rest (`shaders/clouds.wgsl`). The sky pass composites the buffer.

use crate::camera::Camera;
use crate::util::*;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::*;

const SHAPE_RES: u32 = 128;
const DETAIL_RES: u32 = 64;
const WEATHER_RES: u32 = 512;
const CLOUD_FMT: TextureFormat = TextureFormat::Rgba16Float;
/// Cloud shadow map: texels and the ground it covers (m), centred where the camera's sun ray
/// crosses the cloud base. It is anchored to the world and wraps around, so a moving camera
/// keeps what is already in it, and a quarter of it (in 8x8 tiles) is refreshed per frame.
const SHADOW_RES: u32 = 512;
const SHADOW_EXTENT: f32 = 32000.0;
/// Matches CLOUD_SHADOW_MIN_MU in globals.wgsl.
const SHADOW_MIN_MU: f32 = 0.1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CloudUniform {
    prev_vp_rot: Mat4,
    cam_delta: Vec4,
    size: Vec4,
    params: Vec4,
}

pub struct Clouds {
    _textures: Vec<Texture>,
    shape_view: TextureView,
    detail_view: TextureView,
    weather_view: TextureView,
    sampler: Sampler,
    uniform: Buffer,
    pipe: ComputePipeline,
    bgl: BindGroupLayout,
    pub sample_bgl: BindGroupLayout,
    shadow_pipe: ComputePipeline,
    shadow_bg: BindGroup,
    /// Sun transmittance through the clouds (r), for bind group 0.
    pub shadow_view: TextureView,
    /// march_bgs[i] reads buffer 1 - i and writes buffer i; sample_bgs[i] reads buffer i.
    march_bgs: Vec<BindGroup>,
    sample_bgs: Vec<BindGroup>,
    width: u32,
    height: u32,
    frame: u64,
    prev: Option<(Mat4, Vec3)>,
    /// What the shadow map was last fully built for: centre, sun, cloud settings.
    shadow_key: Option<(Vec4, Vec3, Vec4)>,
    shadow_full: bool,
}

fn volume(device: &Device, label: &str, res: u32, mips: u32) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d { width: res, height: res, depth_or_array_layers: res },
        mip_level_count: mips,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn mip_view(tex: &Texture, level: u32) -> TextureView {
    tex.create_view(&TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() })
}

/// Bake the shape and detail volumes (with mips) and the weather map.
fn bake(device: &Device, queue: &Queue, shape: &Texture, detail: &Texture, weather: &Texture) {
    let module = shader(device, "cloud noise", &[include_str!("shaders/cloud_noise.wgsl")]);
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
    let (gen_shape, gen_detail, gen_weather, down) = (pipe("gen_shape"), pipe("gen_detail"), pipe("gen_weather"), pipe("downsample"));
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("cloud noise"), timestamp_writes: None });
        for (p, tex, res) in [(&gen_shape, shape, SHAPE_RES), (&gen_detail, detail, DETAIL_RES)] {
            let bg = bind_group(device, "cloud volume", &p.get_bind_group_layout(0), &[BindingResource::TextureView(&mip_view(tex, 0))]);
            cp.set_pipeline(p);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(res / 4, res / 4, res / 4);
        }
        let weather_view = weather.create_view(&Default::default());
        let bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("weather"),
            layout: &gen_weather.get_bind_group_layout(0),
            entries: &[BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&weather_view) }],
        });
        cp.set_pipeline(&gen_weather);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(WEATHER_RES / 8, WEATHER_RES / 8, 1);

        // mip chains: each level is a box filter of the one above
        let layout = down.get_bind_group_layout(0);
        cp.set_pipeline(&down);
        for tex in [shape, detail] {
            for level in 1..tex.mip_level_count() {
                let entries = [
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&mip_view(tex, level)) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&mip_view(tex, level - 1)) },
                ];
                let bg = device.create_bind_group(&BindGroupDescriptor { label: Some("cloud mip"), layout: &layout, entries: &entries });
                let n = (tex.width() >> level).max(1).div_ceil(4);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(n, n, n);
            }
        }
    }
    queue.submit([enc.finish()]);
}

impl Clouds {
    pub fn new(device: &Device, queue: &Queue, global_bgl: &BindGroupLayout, width: u32, height: u32) -> Clouds {
        let shape = volume(device, "cloud shape", SHAPE_RES, 5);
        let detail = volume(device, "cloud detail", DETAIL_RES, 4);
        let weather = device.create_texture(&TextureDescriptor {
            label: Some("cloud weather"),
            size: Extent3d { width: WEATHER_RES, height: WEATHER_RES, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        bake(device, queue, &shape, &detail, &weather);

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud wrap"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let cs = ShaderStages::COMPUTE;
        let float = TextureSampleType::Float { filterable: true };
        let bgl = bgl(
            device,
            "clouds",
            &[
                uniform_entry(0, cs),
                texture_entry(1, cs, float, TextureViewDimension::D3),
                texture_entry(2, cs, float, TextureViewDimension::D3),
                texture_entry(3, cs, float, TextureViewDimension::D2),
                sampler_entry(4, cs, SamplerBindingType::Filtering),
                texture_entry(5, cs, float, TextureViewDimension::D2),
                storage_texture_entry(6, CLOUD_FMT),
            ],
        );
        let sample_bgl = crate::util::bgl(device, "cloud sample", &[texture_entry(0, ShaderStages::FRAGMENT, float, TextureViewDimension::D2)]);
        let module = shader(device, "clouds", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/clouds.wgsl")]);
        let pipe = compute_pipeline(device, "clouds", &pipeline_layout(device, &[global_bgl, &bgl]), &module, "march_clouds");

        let shadow_tex = device.create_texture(&TextureDescriptor {
            label: Some("cloud shadow"),
            size: Extent3d { width: SHADOW_RES, height: SHADOW_RES, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let shadow_bgl = crate::util::bgl(
            device,
            "cloud shadow",
            &[
                uniform_entry(0, cs),
                texture_entry(1, cs, float, TextureViewDimension::D3),
                texture_entry(2, cs, float, TextureViewDimension::D3),
                texture_entry(3, cs, float, TextureViewDimension::D2),
                sampler_entry(4, cs, SamplerBindingType::Filtering),
                storage_texture_entry(7, TextureFormat::Rgba8Unorm),
            ],
        );
        let shadow_pipe = compute_pipeline(device, "cloud shadow", &pipeline_layout(device, &[global_bgl, &shadow_bgl]), &module, "cloud_shadow_map");
        let (shape_view, detail_view, weather_view) = (shape.create_view(&Default::default()), detail.create_view(&Default::default()), weather.create_view(&Default::default()));
        let uniform = buffer(device, "clouds", std::mem::size_of::<CloudUniform>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let shadow_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud shadow"),
            layout: &shadow_bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&shape_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&detail_view) },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&weather_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry { binding: 7, resource: BindingResource::TextureView(&shadow_view) },
            ],
        });

        let mut clouds = Clouds {
            shape_view,
            detail_view,
            weather_view,
            _textures: vec![shape, detail, weather, shadow_tex],
            shadow_pipe,
            shadow_bg,
            shadow_view,
            sampler,
            uniform,
            pipe,
            bgl,
            sample_bgl,
            march_bgs: Vec::new(),
            sample_bgs: Vec::new(),
            width: 0,
            height: 0,
            frame: 0,
            prev: None,
            shadow_key: None,
            shadow_full: true,
        };
        clouds.resize(device, width, height);
        clouds
    }

    /// (Re)create the ping-pong buffers, a half to a quarter of the screen resolution so they
    /// stay around 450-540 rows. History restarts.
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        let scale = (height as f32 / 480.0).round().max(2.0) as u32;
        self.width = width.div_ceil(scale).max(1);
        self.height = height.div_ceil(scale).max(1);
        let views: Vec<TextureView> = (0..2)
            .map(|_| {
                device
                    .create_texture(&TextureDescriptor {
                        label: Some("clouds"),
                        size: Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: CLOUD_FMT,
                        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            })
            .collect();
        self.march_bgs = (0..2)
            .map(|i| {
                bind_group(
                    device,
                    "clouds",
                    &self.bgl,
                    &[
                        self.uniform.as_entire_binding(),
                        BindingResource::TextureView(&self.shape_view),
                        BindingResource::TextureView(&self.detail_view),
                        BindingResource::TextureView(&self.weather_view),
                        BindingResource::Sampler(&self.sampler),
                        BindingResource::TextureView(&views[1 - i]),
                        BindingResource::TextureView(&views[i]),
                    ],
                )
            })
            .collect();
        self.sample_bgs = views.iter().map(|v| bind_group(device, "cloud sample", &self.sample_bgl, &[BindingResource::TextureView(v)])).collect();
        self.prev = None;
    }

    /// Per-frame uniforms. `vp_rot` is proj * view rotation, as used for the sky rays;
    /// `shadow` is `shadow_params`, `layer` the frame's cloud settings.
    pub fn update(&mut self, queue: &Queue, vp_rot: Mat4, cam: &Camera, sun: Vec3, shadow: Vec4, layer: Vec4) {
        self.frame += 1;
        // the map is refreshed a quarter at a time unless what it shows changed at once
        self.shadow_full = match self.shadow_key {
            Some((c, s, l)) => c.truncate().distance(shadow.truncate()) > 1500.0 || s.dot(sun) < 0.99999 || l != layer,
            None => true,
        };
        if self.shadow_full {
            self.shadow_key = Some((shadow, sun, layer));
        } else if let Some(k) = &mut self.shadow_key {
            // the sun creeps (`[` / `]`): rebuild once it has moved a little, not every frame
            k.0 = shadow;
        }
        let (prev_vp, delta, valid) = match self.prev {
            Some((vp, pos)) => (vp, cam.pos - pos, 1.0),
            None => (vp_rot, Vec3::ZERO, 0.0),
        };
        let pixel_angle = 2.0 * (cam.fov_y * 0.5).tan() / self.height as f32;
        let u = CloudUniform {
            prev_vp_rot: prev_vp,
            cam_delta: delta.extend(valid),
            size: Vec4::new(self.width as f32, self.height as f32, 1.0 / self.width as f32, 1.0 / self.height as f32),
            params: Vec4::new(self.frame as f32, pixel_angle, if self.shadow_full { 0.0 } else { 1.0 }, 0.0),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        self.prev = Some((vp_rot, cam.pos));
    }

    /// `cloud_shadow` for the frame uniform: map centre, 1 / extent, and the light assumed
    /// beyond the map's edge.
    pub fn shadow_params(cam: Vec3, sun: Vec3, cloud_base: f32, cover: f32) -> Vec4 {
        let k = (cloud_base - cam.y) / sun.y.max(SHADOW_MIN_MU);
        let beyond = 1.0 - 0.8 * ((cover - 0.3) / 0.6).clamp(0.0, 1.0);
        Vec4::new(cam.x + sun.x * k, cam.z + sun.z * k, 1.0 / SHADOW_EXTENT, beyond)
    }

    /// Rebuild the cloud shadow map. `global_bg` must not hold the map itself.
    pub fn render_shadow(&self, cp: &mut ComputePass, global_bg: &BindGroup) {
        cp.set_pipeline(&self.shadow_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.shadow_bg, &[]);
        let n = if self.shadow_full { SHADOW_RES } else { SHADOW_RES / 2 };
        cp.dispatch_workgroups(n / 8, n / 8, 1);
    }

    fn current(&self) -> usize {
        (self.frame % 2) as usize
    }

    pub fn march(&self, cp: &mut ComputePass, global_bg: &BindGroup) {
        cp.set_pipeline(&self.pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.march_bgs[self.current()], &[]);
        cp.dispatch_workgroups(self.width.div_ceil(2).div_ceil(8), self.height.div_ceil(2).div_ceil(8), 1);
    }

    /// The buffer this frame's march wrote, for the sky pass.
    pub fn output(&self) -> &BindGroup {
        &self.sample_bgs[self.current()]
    }
}
