//! The forest: GPU scatter of ~a million trees, per-view GPU culling + LOD selection
//! into indirect draw buckets, and instanced rendering of the culled lists.

use crate::treegen::{self, TreeVertex};
use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::*;

pub const SPECIES: usize = 3;
pub const LODS: usize = 4;
pub const DRAWS: usize = 12; // 3 species * 4 LODs (fits the 12-entry uniform arrays)
const SCATTER_CELL: f32 = 5.5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    planes: [Vec4; 6],
    cam: Vec4,
    lod: Vec4,
    counts: [u32; 4],
    regions: [[u32; 4]; 12],
    bounds: [Vec4; 4],
}

#[derive(Clone, Copy, Default)]
pub struct DrawRange {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub tris: u32,
}

pub struct CullView {
    params_buf: Buffer,
    pub args_buf: Buffer,
    visible_buf: Buffer,
    bind_group: BindGroup,
    regions: [[u32; 4]; 12],
    reset_args: Vec<u32>,
}

pub struct Forest {
    vbuf: Buffer,
    ibuf: Buffer,
    pub draws: [DrawRange; DRAWS],
    pub tree_count: u32,
    pub views: Vec<CullView>,
    cull_pipe: ComputePipeline,
    tree_bg: BindGroup,
    pipe_color: RenderPipeline,
    pipe_shadow: RenderPipeline,
    bounds: [Vec4; 4],
}

pub struct ViewSetup {
    pub planes: [Vec4; 6],
    pub ref_pos: Vec3,
    pub max_dist: f32,
    pub lod: Vec4,
}

impl Forest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        queue: &Queue,
        global_bgl: &BindGroupLayout,
        global_bg: &BindGroup,
        pass_bgl: &BindGroupLayout,
        hdr: TextureFormat,
        msaa: u32,
        world: f32,
    ) -> Forest {
        // ---- meshes
        let species = treegen::build_all();
        let mut verts: Vec<TreeVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut draws = [DrawRange::default(); DRAWS];
        let mut bounds = [Vec4::ZERO; 4];
        for (s, sp) in species.iter().enumerate() {
            bounds[s] = Vec4::new(sp.center_y, sp.radius, 0.0, 0.0);
            for (l, m) in sp.lods.iter().enumerate() {
                draws[s * LODS + l] = DrawRange {
                    index_count: m.idx.len() as u32,
                    first_index: idx.len() as u32,
                    base_vertex: verts.len() as i32,
                    tris: m.tris(),
                };
                verts.extend_from_slice(&m.verts);
                idx.extend_from_slice(&m.idx);
            }
        }
        let vbuf = buffer_init(device, "tree verts", cast_slice(&verts), BufferUsages::VERTEX);
        let ibuf = buffer_init(device, "tree idx", cast_slice(&idx), BufferUsages::INDEX);

        // ---- scatter
        let dim = (world / SCATTER_CELL) as u32;
        let cap = dim * dim;
        let trees_buf = buffer(device, "trees", cap as u64 * 16, BufferUsages::STORAGE);
        let counter = buffer(device, "tree counter", 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST);
        let counter_rb = buffer(device, "tree counter rb", 4, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        let sp: [u32; 4] = [SCATTER_CELL.to_bits(), dim, cap, 0];
        let sp_buf = buffer_init(device, "scatter params", cast_slice(&sp), BufferUsages::UNIFORM);
        let scatter_bgl = bgl(
            device,
            "scatter",
            &[uniform_entry(0, ShaderStages::COMPUTE), storage_entry(1, ShaderStages::COMPUTE, false), storage_entry(2, ShaderStages::COMPUTE, false)],
        );
        let scatter_bg = bind_group(
            device,
            "scatter",
            &scatter_bgl,
            &[sp_buf.as_entire_binding(), trees_buf.as_entire_binding(), counter.as_entire_binding()],
        );
        let module = shader(device, "scatter", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/scatter.wgsl")]);
        let pl = pipeline_layout(device, &[global_bgl, &scatter_bgl]);
        let scatter_pipe = compute_pipeline(device, "scatter", &pl, &module, "main");
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&counter, 0, None);
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&scatter_pipe);
            cp.set_bind_group(0, global_bg, &[]);
            cp.set_bind_group(1, &scatter_bg, &[]);
            cp.dispatch_workgroups(dim.div_ceil(8), dim.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&counter, 0, &counter_rb, 0, 4);
        queue.submit([enc.finish()]);
        let n: u32 = cast_slice::<u8, u32>(&read_buffer_blocking(device, &counter_rb, 4))[0];
        let tree_count = n.min(cap);
        println!("forest: scattered {tree_count} trees from {cap} candidate cells");

        // ---- culling
        let cull_bgl = bgl(
            device,
            "cull",
            &[
                uniform_entry(0, ShaderStages::COMPUTE),
                storage_entry(1, ShaderStages::COMPUTE, true),
                storage_entry(2, ShaderStages::COMPUTE, false),
                storage_entry(3, ShaderStages::COMPUTE, false),
            ],
        );
        let module = shader(device, "cull", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/cull.wgsl")]);
        let pl = pipeline_layout(device, &[global_bgl, &cull_bgl]);
        let cull_pipe = compute_pipeline(device, "cull", &pl, &module, "main");

        let main_caps = [8_000u32, 50_000, 220_000, 700_000];
        let shadow_caps = [8_000u32, 50_000, 120_000, 120_000];
        let mut views = Vec::new();
        for caps in [main_caps, shadow_caps, shadow_caps] {
            let mut regions = [[0u32; 4]; 12];
            let mut cursor = 0;
            for s in 0..SPECIES {
                for l in 0..LODS {
                    regions[s * LODS + l] = [cursor, caps[l], 0, 0];
                    cursor += caps[l];
                }
            }
            let mut reset_args = vec![0u32; DRAWS * 5];
            for d in 0..DRAWS {
                reset_args[d * 5] = draws[d].index_count;
                reset_args[d * 5 + 1] = 0;
                reset_args[d * 5 + 2] = draws[d].first_index;
                reset_args[d * 5 + 3] = draws[d].base_vertex as u32;
                reset_args[d * 5 + 4] = 0;
            }
            let params_buf = buffer(device, "cull params", std::mem::size_of::<CullParams>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
            let args_buf = buffer(
                device,
                "tree draw args",
                (DRAWS * 20) as u64,
                BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            );
            let visible_buf = buffer(device, "visible trees", cursor as u64 * 4, BufferUsages::STORAGE | BufferUsages::VERTEX);
            let bind_group = bind_group(
                device,
                "cull",
                &cull_bgl,
                &[params_buf.as_entire_binding(), trees_buf.as_entire_binding(), args_buf.as_entire_binding(), visible_buf.as_entire_binding()],
            );
            views.push(CullView { params_buf, args_buf, visible_buf, bind_group, regions, reset_args });
        }

        // ---- rendering
        let tree_bgl = bgl(device, "trees", &[storage_entry(0, ShaderStages::VERTEX, true)]);
        let tree_bg = bind_group(device, "trees", &tree_bgl, &[trees_buf.as_entire_binding()]);
        let module = shader(device, "tree", &[NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/tree.wgsl")]);
        let layout = pipeline_layout(device, &[global_bgl, &tree_bgl, pass_bgl]);
        let vattrs = [
            VertexAttribute { format: VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
            VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 3 },
        ];
        let iattrs = [VertexAttribute { format: VertexFormat::Uint32, offset: 0, shader_location: 4 }];
        let buffers = [
            Some(VertexBufferLayout { array_stride: std::mem::size_of::<TreeVertex>() as u64, step_mode: VertexStepMode::Vertex, attributes: &vattrs }),
            Some(VertexBufferLayout { array_stride: 4, step_mode: VertexStepMode::Instance, attributes: &iattrs }),
        ];
        let targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::ALL })];
        let pipe_color = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("tree"),
            layout: Some(&layout),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &buffers },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual)),
            multisample: MultisampleState { count: msaa, mask: !0, alpha_to_coverage_enabled: msaa > 1 },
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &targets }),
            multiview_mask: None,
            cache: None,
        });
        let pipe_shadow = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("tree shadow"),
            layout: Some(&layout),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &buffers },
            primitive: PrimitiveState::default(),
            depth_stencil: Some(shadow_depth_state()),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState { module: &module, entry_point: Some("fs_shadow"), compilation_options: Default::default(), targets: &[] }),
            multiview_mask: None,
            cache: None,
        });

        Forest { vbuf, ibuf, draws, tree_count, views, cull_pipe, tree_bg, pipe_color, pipe_shadow, bounds }
    }

    pub fn update_view(&self, queue: &Queue, view: usize, setup: &ViewSetup) {
        let v = &self.views[view];
        let params = CullParams {
            planes: setup.planes,
            cam: setup.ref_pos.extend(setup.max_dist),
            lod: setup.lod,
            counts: [self.tree_count, 0, 0, 0],
            regions: v.regions,
            bounds: self.bounds,
        };
        queue.write_buffer(&v.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&v.args_buf, 0, cast_slice(&v.reset_args));
    }

    pub fn cull(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup, view: usize) {
        let groups = self.tree_count.div_ceil(256);
        let (x, y) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups, 1) };
        cp.set_pipeline(&self.cull_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, &self.views[view].bind_group, &[]);
        cp.dispatch_workgroups(x, y, 1);
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>, view: usize, shadow: bool, max_lod: usize) {
        let v = &self.views[view];
        pass.set_pipeline(if shadow { &self.pipe_shadow } else { &self.pipe_color });
        pass.set_bind_group(1, &self.tree_bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), IndexFormat::Uint32);
        for d in 0..DRAWS {
            if d % LODS > max_lod {
                continue;
            }
            let start = v.regions[d][0] as u64 * 4;
            pass.set_vertex_buffer(1, v.visible_buf.slice(start..));
            pass.draw_indexed_indirect(&v.args_buf, (d * 20) as u64);
        }
    }
}
