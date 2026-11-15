//! The meadow: tufts of grass blades, cards of scanned grass tufts, and flowers and seed heads
//! on stalks among the blades. Regenerated around the camera every frame by a compute shader
//! (`shaders/grass_gen.wgsl`) and drawn with two indirect calls (`shaders/grass.wgsl`).

use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrassParams {
    planes: [Vec4; 6],
    origin: Vec4,
    origin2: Vec4,
    dims: [u32; 4],
}

/// A grid of cells around the camera.
struct Grid {
    cell: f32,
    radius: f32,
    dim: u32,
}

impl Grid {
    fn new(cell: f32, radius: f32) -> Grid {
        Grid { cell, radius, dim: (radius * 2.0 / cell).ceil() as u32 + 2 }
    }
    fn origin(&self, cam: Vec3) -> Vec4 {
        let ox = ((cam.x - self.radius) / self.cell).floor() * self.cell;
        let oz = ((cam.z - self.radius) / self.cell).floor() * self.cell;
        Vec4::new(ox, oz, self.cell, self.radius)
    }
}

/// Vertices per blade (a strip of 5 triangles) and per card.
const BLADE_VERTS: u32 = 7;
const CARD_VERTS: u32 = 4;

pub struct Grass {
    params_buf: Buffer,
    pub args_buf: Buffer,
    gen_bg: BindGroup,
    draw_bg: BindGroup,
    gen_blades: ComputePipeline,
    gen_cards: ComputePipeline,
    blade_pipe: RenderPipeline,
    card_pipe: RenderPipeline,
    blade_cap: u32,
    card_cap: u32,
    blades: Grid,
    cards: Grid,
}

impl Grass {
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, pass_bgl: &BindGroupLayout, hdr: TextureFormat, msaa: u32) -> Grass {
        let blade_cap = 420_000u32;
        let card_cap = 160_000u32;
        // the blades thin out from GRASS_NEAR to GRASS_FAR (globals.wgsl)
        let blades = Grid::new(0.12, 26.0);
        let cards = Grid::new(0.25, 60.0);
        let params_buf = buffer(device, "grass params", std::mem::size_of::<GrassParams>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let blade_buf = buffer(device, "grass blades", blade_cap as u64 * 48, BufferUsages::STORAGE);
        let card_buf = buffer(device, "grass cards", card_cap as u64 * 32, BufferUsages::STORAGE);
        let args_buf = buffer(device, "grass args", 32, BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC);

        let gen_bgl = bgl(
            device,
            "grass gen",
            &[
                uniform_entry(0, ShaderStages::COMPUTE),
                storage_entry(1, ShaderStages::COMPUTE, false),
                storage_entry(2, ShaderStages::COMPUTE, false),
                storage_entry(3, ShaderStages::COMPUTE, false),
            ],
        );
        let gen_bg = bind_group(
            device,
            "grass gen",
            &gen_bgl,
            &[params_buf.as_entire_binding(), blade_buf.as_entire_binding(), card_buf.as_entire_binding(), args_buf.as_entire_binding()],
        );
        let module = shader(device, "grass gen", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/grass_gen.wgsl")]);
        let layout = pipeline_layout(device, &[global_bgl, &gen_bgl]);
        let gen_blades = compute_pipeline(device, "grass gen blades", &layout, &module, "blades_main");
        let gen_cards = compute_pipeline(device, "grass gen cards", &layout, &module, "cards_main");

        let draw_bgl = bgl(device, "grass draw", &[storage_entry(0, ShaderStages::VERTEX, true), storage_entry(1, ShaderStages::VERTEX, true)]);
        let draw_bg = bind_group(device, "grass draw", &draw_bgl, &[blade_buf.as_entire_binding(), card_buf.as_entire_binding()]);
        let module = shader(device, "grass", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/grass.wgsl")]);
        let layout = pipeline_layout(device, &[global_bgl, &draw_bgl, pass_bgl]);
        let targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::ALL })];
        let pipe = |label: &str, vs: &str, fs: &str, cutout: bool| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState { module: &module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                primitive: PrimitiveState { topology: PrimitiveTopology::TriangleStrip, ..Default::default() },
                depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual)),
                multisample: MultisampleState { count: msaa, mask: !0, alpha_to_coverage_enabled: cutout && msaa > 1 },
                fragment: Some(FragmentState { module: &module, entry_point: Some(fs), compilation_options: Default::default(), targets: &targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let blade_pipe = pipe("grass blades", "vs_blade", "fs_blade", false);
        // without MSAA there is no coverage to cut the cards with: an alpha test instead
        let card_pipe = pipe("grass cards", "vs_card", if msaa > 1 { "fs_card" } else { "fs_card_test" }, true);
        Grass { params_buf, args_buf, gen_bg, draw_bg, gen_blades, gen_cards, blade_pipe, card_pipe, blade_cap, card_cap, blades, cards }
    }

    pub fn update(&self, queue: &Queue, planes: [Vec4; 6], cam: Vec3) {
        let p = GrassParams {
            planes,
            origin: self.blades.origin(cam),
            origin2: self.cards.origin(cam),
            dims: [self.blades.dim, self.blade_cap, self.cards.dim, self.card_cap],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        queue.write_buffer(&self.args_buf, 0, cast_slice(&[BLADE_VERTS, 0, 0, 0, CARD_VERTS, 0, 0, 0]));
    }

    pub fn generate(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup) {
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.gen_bg, &[]);
        cp.set_pipeline(&self.gen_blades);
        cp.dispatch_workgroups(self.blades.dim.div_ceil(8), self.blades.dim.div_ceil(8), 1);
        cp.set_pipeline(&self.gen_cards);
        cp.dispatch_workgroups(self.cards.dim.div_ceil(8), self.cards.dim.div_ceil(8), 1);
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>) {
        pass.set_bind_group(1, &self.draw_bg, &[]);
        pass.set_pipeline(&self.blade_pipe);
        pass.draw_indirect(&self.args_buf, 0);
        pass.set_pipeline(&self.card_pipe);
        pass.draw_indirect(&self.args_buf, 16);
    }
}
