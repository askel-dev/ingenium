//! Grass: regenerated around the camera every frame by a compute shader, drawn with
//! one indirect call.

use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrassParams {
    planes: [Vec4; 6],
    origin: Vec4,
    dims: [u32; 4],
}

pub struct Grass {
    params_buf: Buffer,
    pub args_buf: Buffer,
    gen_bg: BindGroup,
    draw_bg: BindGroup,
    gen_pipe: ComputePipeline,
    draw_pipe: RenderPipeline,
    cap: u32,
    cell: f32,
    radius: f32,
    dim: u32,
}

impl Grass {
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, pass_bgl: &BindGroupLayout, hdr: TextureFormat, msaa: u32) -> Grass {
        let cap = 600_000u32;
        let cell = 0.16f32;
        let radius = 60.0f32;
        let dim = (radius * 2.0 / cell).ceil() as u32 + 2;
        let params_buf = buffer(device, "grass params", std::mem::size_of::<GrassParams>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let blades = buffer(device, "grass blades", cap as u64 * 32, BufferUsages::STORAGE);
        let args_buf = buffer(device, "grass args", 16, BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC);

        let gen_bgl = bgl(
            device,
            "grass gen",
            &[uniform_entry(0, ShaderStages::COMPUTE), storage_entry(1, ShaderStages::COMPUTE, false), storage_entry(2, ShaderStages::COMPUTE, false)],
        );
        let gen_bg = bind_group(device, "grass gen", &gen_bgl, &[params_buf.as_entire_binding(), blades.as_entire_binding(), args_buf.as_entire_binding()]);
        let module = shader(device, "grass gen", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/grass_gen.wgsl")]);
        let gen_pipe = compute_pipeline(device, "grass gen", &pipeline_layout(device, &[global_bgl, &gen_bgl]), &module, "main");

        let draw_bgl = bgl(device, "grass draw", &[storage_entry(0, ShaderStages::VERTEX, true)]);
        let draw_bg = bind_group(device, "grass draw", &draw_bgl, &[blades.as_entire_binding()]);
        let module = shader(device, "grass", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/grass.wgsl")]);
        let layout = pipeline_layout(device, &[global_bgl, &draw_bgl, pass_bgl]);
        let targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::ALL })];
        let draw_pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("grass"),
            layout: Some(&layout),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual)),
            multisample: MultisampleState { count: msaa, ..Default::default() },
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &targets }),
            multiview_mask: None,
            cache: None,
        });
        Grass { params_buf, args_buf, gen_bg, draw_bg, gen_pipe, draw_pipe, cap, cell, radius, dim }
    }

    pub fn update(&self, queue: &Queue, planes: [Vec4; 6], cam: Vec3) {
        let ox = ((cam.x - self.radius) / self.cell).floor() * self.cell;
        let oz = ((cam.z - self.radius) / self.cell).floor() * self.cell;
        let p = GrassParams { planes, origin: Vec4::new(ox, oz, self.cell, self.radius), dims: [self.dim, self.cap, 0, 0] };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        queue.write_buffer(&self.args_buf, 0, cast_slice(&[15u32, 0, 0, 0]));
    }

    pub fn generate(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup) {
        cp.set_pipeline(&self.gen_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.gen_bg, &[]);
        cp.dispatch_workgroups(self.dim.div_ceil(8), self.dim.div_ceil(8), 1);
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>) {
        pass.set_pipeline(&self.draw_pipe);
        pass.set_bind_group(1, &self.draw_bg, &[]);
        pass.draw_indirect(&self.args_buf, 0);
    }
}
