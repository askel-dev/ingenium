//! Forest-floor plants (ferns, shrubs, ground cover): like the grass, regenerated around the
//! camera every frame by a compute shader in a world-anchored grid, so they don't swim. The
//! records have the same layout as the trees', so they are drawn with the tree pipeline: one
//! indirect draw per kind.

use crate::forest::Forest;
use crate::treegen::{self, TreeVertex};
use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

pub const KINDS: usize = 3;
/// Plants per kind at most.
const CAP: u32 = 12_000;
/// Grid cell (one plant at most per cell) and how far out plants grow.
const CELL: f32 = 0.85;
const RADIUS: f32 = 42.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    planes: [Vec4; 6],
    origin: Vec4,        // x, z of the grid's first cell, cell size, radius
    dims: [u32; 4],      // cells per side, capacity per kind
    kinds: [Vec4; KINDS], // x = texture layer, y = bounding radius
}

pub struct Undergrowth {
    params_buf: Buffer,
    pub args_buf: Buffer,
    reset_args: Vec<u32>,
    kinds: [Vec4; KINDS],
    vbuf: Buffer,
    ibuf: Buffer,
    ids: Buffer,
    gen_bg: BindGroup,
    draw_bg: BindGroup,
    gen_pipe: ComputePipeline,
    dim: u32,
}

impl Undergrowth {
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, forest: &Forest) -> Undergrowth {
        let plants = treegen::build_plants();
        let mut verts: Vec<TreeVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut reset_args = vec![0u32; KINDS * 5];
        let mut kinds = [Vec4::ZERO; KINDS];
        for (k, p) in plants.iter().enumerate() {
            reset_args[k * 5] = p.mesh.idx.len() as u32;
            reset_args[k * 5 + 2] = idx.len() as u32;
            reset_args[k * 5 + 3] = verts.len() as u32;
            kinds[k] = Vec4::new(p.layer as f32, p.radius, 0.0, 0.0);
            verts.extend_from_slice(&p.mesh.verts);
            idx.extend_from_slice(&p.mesh.idx);
        }
        let vbuf = buffer_init(device, "plant verts", cast_slice(&verts), BufferUsages::VERTEX);
        let ibuf = buffer_init(device, "plant idx", cast_slice(&idx), BufferUsages::INDEX);
        // instance i of kind k reads record k * CAP + i (see `draw`)
        let ids: Vec<u32> = (0..KINDS as u32 * CAP).collect();
        let ids = buffer_init(device, "plant ids", cast_slice(&ids), BufferUsages::VERTEX);
        let records = buffer(device, "plants", KINDS as u64 * CAP as u64 * 16, BufferUsages::STORAGE);
        let params_buf = buffer(device, "plant params", std::mem::size_of::<Params>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let args_buf = buffer(
            device,
            "plant args",
            (KINDS * 20) as u64,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let gen_bgl = bgl(
            device,
            "plant gen",
            &[uniform_entry(0, ShaderStages::COMPUTE), storage_entry(1, ShaderStages::COMPUTE, false), storage_entry(2, ShaderStages::COMPUTE, false)],
        );
        let gen_bg = bind_group(device, "plant gen", &gen_bgl, &[params_buf.as_entire_binding(), records.as_entire_binding(), args_buf.as_entire_binding()]);
        let module = shader(device, "plant gen", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/undergrowth.wgsl")]);
        let gen_pipe = compute_pipeline(device, "plant gen", &pipeline_layout(device, &[global_bgl, &gen_bgl]), &module, "main");
        let draw_bg = forest.records_bind_group(device, &records);
        let dim = (RADIUS * 2.0 / CELL).ceil() as u32 + 2;
        Undergrowth { params_buf, args_buf, reset_args, kinds, vbuf, ibuf, ids, gen_bg, draw_bg, gen_pipe, dim }
    }

    pub fn update(&self, queue: &Queue, planes: [Vec4; 6], cam: Vec3) {
        let ox = ((cam.x - RADIUS) / CELL).floor() * CELL;
        let oz = ((cam.z - RADIUS) / CELL).floor() * CELL;
        let p = Params { planes, origin: Vec4::new(ox, oz, CELL, RADIUS), dims: [self.dim, CAP, 0, 0], kinds: self.kinds };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
        queue.write_buffer(&self.args_buf, 0, cast_slice(&self.reset_args));
    }

    pub fn generate(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup) {
        cp.set_pipeline(&self.gen_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.gen_bg, &[]);
        cp.dispatch_workgroups(self.dim.div_ceil(8), self.dim.div_ceil(8), 1);
    }

    /// Draws with the tree colour pipeline (`Forest::color_pipeline`); group 2 must be set.
    pub fn draw(&self, pass: &mut RenderPass<'_>, pipe: &RenderPipeline) {
        pass.set_pipeline(pipe);
        pass.set_bind_group(1, &self.draw_bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), IndexFormat::Uint32);
        for k in 0..KINDS {
            pass.set_vertex_buffer(1, self.ids.slice(k as u64 * CAP as u64 * 4..));
            pass.draw_indexed_indirect(&self.args_buf, (k * 20) as u64);
        }
    }
}
