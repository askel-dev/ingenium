//! The terrain's and the forest canopy's shadow over the whole world (`shaders/sun_height.wgsl`).
//! The shadow cascades end 260 m out, and towards the sun they only see 1.5 km; this map holds,
//! for every point of the world, the height at which the sun's rays clear the terrain (and the
//! terrain with the canopy on top). Mountains then shade the valleys at any distance, and forest
//! masses the meadows next to them. It is rebuilt only when the sun moves: one sweep down each
//! column of a map laid out along the sun.

use crate::terrain::Terrain;
use crate::util::*;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

/// Texels per side. The map is a square around the world turned to face the sun, so a texel
/// covers 4 to 5.7 m depending on the sun's azimuth (the heightmap's are 4 m).
pub const RES: u32 = 2048;
const FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The steepest the map takes the sun: higher than this, nothing on the heightmap casts a
/// shadow worth having.
const MAX_TAN: f32 = 1000.0;
/// The lowest: the sun is gone below about this (and its light with it).
const MIN_TAN: f32 = 0.01;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    world: f32,
    res: f32,
    extent: f32,
    tan_el: f32,
    sun: [f32; 2],
    pad: [f32; 2],
}

pub struct HeightShadow {
    _tex: Texture,
    pub view: TextureView,
    params: Buffer,
    pipe: ComputePipeline,
    bg: BindGroup,
    world: f32,
    hm_res: u32,
    built_for: Option<Vec3>,
}

impl HeightShadow {
    pub fn new(device: &Device, terrain: &Terrain) -> HeightShadow {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("height shadow"),
            size: Extent3d { width: RES, height: RES, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: FORMAT,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let params = buffer(device, "height shadow params", std::mem::size_of::<Params>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let cs = ShaderStages::COMPUTE;
        let unfiltered = TextureSampleType::Float { filterable: false };
        let layout = bgl(
            device,
            "height shadow",
            &[
                uniform_entry(0, cs),
                texture_entry(1, cs, unfiltered, TextureViewDimension::D2),
                texture_entry(2, cs, unfiltered, TextureViewDimension::D2),
                storage_texture_entry(3, FORMAT),
            ],
        );
        let bg = bind_group(
            device,
            "height shadow",
            &layout,
            &[params.as_entire_binding(), BindingResource::TextureView(&terrain.height_view), BindingResource::TextureView(&terrain.data_view), BindingResource::TextureView(&view)],
        );
        let module = shader(device, "height shadow", &[include_str!("shaders/sun_height.wgsl")]);
        let pipe = compute_pipeline(device, "height shadow", &pipeline_layout(device, &[&layout]), &module, "main");
        HeightShadow { _tex: tex, view, params, pipe, bg, world: terrain.world, hm_res: terrain.res, built_for: None }
    }

    /// The map's layout for the frame uniform (`sun_grid` in globals.wgsl): the horizontal
    /// direction towards the sun, the tangent of its elevation, and the metres the map covers
    /// along each axis.
    pub fn grid(&self, sun: Vec3) -> Vec4 {
        let h = glam::Vec2::new(sun.x, sun.z);
        let len = h.length();
        let (dir, tan) = if len < 1e-5 { (glam::Vec2::X, MAX_TAN) } else { (h / len, (sun.y / len).clamp(MIN_TAN, MAX_TAN)) };
        // the world's square seen along the sun, plus a texel each side
        let extent = self.world * (dir.x.abs() + dir.y.abs()) * (1.0 + 2.0 / RES as f32);
        Vec4::new(dir.x, dir.y, tan, extent)
    }

    /// Record a rebuild if the sun moved since the last one.
    pub fn update(&mut self, queue: &Queue, enc: &mut CommandEncoder, sun: Vec3) {
        if self.built_for.is_some_and(|s| s.distance(sun) < 1e-5) {
            return;
        }
        self.built_for = Some(sun);
        let g = self.grid(sun);
        let p = Params { world: self.world, res: self.hm_res as f32, extent: g.w, tan_el: g.z, sun: [g.x, g.y], pad: [0.0; 2] };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("height shadow"), timestamp_writes: None });
        cp.set_pipeline(&self.pipe);
        cp.set_bind_group(0, &self.bg, &[]);
        cp.dispatch_workgroups(RES.div_ceil(64), 1, 1);
    }
}
