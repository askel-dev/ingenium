//! The forest: GPU scatter of ~a million trees, per-view GPU culling + LOD selection into
//! indirect draw buckets, and instanced rendering of the culled lists. Near trees are meshes,
//! with a bucket per tree shape and mesh LOD; from LOD2 on every tree is an impostor
//! (impostor.rs), one bucket per LOD for all shapes.

use crate::impostor::{self, Atlas, BakeItem};
use crate::treegen::{self, Shape, TreeVertex, KIND_LEAF, NO_SLOT, SHAPES, SLOTS, VARIANTS};
use crate::util::*;
use bytemuck::{cast_slice, Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::*;

pub const LODS: usize = 4;
/// LODs below this are meshes, the rest impostors.
pub const MESH_LODS: usize = 2;
/// Draw buckets: one per shape and mesh LOD, then one per impostor LOD.
pub const DRAWS: usize = SHAPES * MESH_LODS + (LODS - MESH_LODS);
pub const SCATTER_CELL: f32 = 5.5;
/// Planes a tree is culled against in each view (a frustum uses 6; the rest are padding).
pub const CULL_PLANES: usize = 8;
/// With occlusion culling, the camera's LODs below this are drawn first, as occluders; the
/// rest are culled against the depth they leave (see cull.wgsl).
pub const OCCLUDER_LODS: usize = 2;

/// A tree's scale from its record (bits 10-17), as `tree_scale` in globals.wgsl decodes it.
pub fn tree_scale(packed: u32) -> f32 {
    0.45 + ((packed >> 10) & 255) as f32 / 255.0 * 1.3
}

// cull.wgsl's modes
const MODE_ALL: u32 = 0;
const MODE_OCCLUDERS: u32 = 1;
const MODE_OCCLUDED: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    planes: [Vec4; CULL_PLANES],
    cam: Vec4,
    lod: Vec4,
    counts: [u32; 4],
    shapes: [Vec4; SLOTS],
    regions: [[u32; 4]; DRAWS],
    view: Mat4,
    proj: Vec4,
    hiz: Vec4,
}

/// One tree shape for the shaders (`Shape` in tree_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeGpu {
    crown_c: Vec4,
    crown_r: Vec4,
    imp: Vec4,
}

#[derive(Clone, Copy, Default)]
pub struct DrawRange {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub tris: u32,
    pub lod: usize,
}

impl DrawRange {
    fn impostor(&self) -> bool {
        self.lod >= MESH_LODS
    }
}

pub struct CullView {
    params_buf: Buffer,
    pub args_buf: Buffer,
    visible_buf: Buffer,
    bind_group: BindGroup,
    regions: [[u32; 4]; DRAWS],
    reset_args: Vec<u32>,
}

/// Which pipeline `Forest::draw` uses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TreePass {
    /// Depth into a shadow cascade.
    Shadow,
    /// Shaded, cut out by alpha-to-coverage, writing depth (no prepass).
    Color,
    /// Depth only, cut out like `Color` (the prepass).
    Depth,
    /// Shaded where the prepass left this surface's depth: no cut-out, no depth writes.
    ColorEqual,
}

/// The four pipelines of one kind of tree geometry (meshes or impostors), by `TreePass`.
struct Pipes {
    shadow: RenderPipeline,
    color: RenderPipeline,
    depth: RenderPipeline,
    color_equal: RenderPipeline,
}

impl Pipes {
    fn get(&self, kind: TreePass) -> &RenderPipeline {
        match kind {
            TreePass::Shadow => &self.shadow,
            TreePass::Color => &self.color,
            TreePass::Depth => &self.depth,
            TreePass::ColorEqual => &self.color_equal,
        }
    }
}

pub struct Forest {
    vbuf: Buffer,
    ibuf: Buffer,
    pub draws: [DrawRange; DRAWS],
    pub tree_count: u32,
    pub views: Vec<CullView>,
    cull_pipe: ComputePipeline,
    cull_bgl: BindGroupLayout,
    /// The camera's second culling pass (`MODE_OCCLUDED`): its own parameters, the camera's
    /// lists, and the depth pyramid (`None` until `set_hiz`).
    late_params: Buffer,
    late_bg: Option<BindGroup>,
    hiz_info: Vec4,
    _dummy_hiz: Texture,
    tree_bg: BindGroup,
    /// group 1 of the tree pipelines: the instance records (`trees` in tree.wgsl), the shape
    /// table and the impostor atlas
    tree_bgl: BindGroupLayout,
    shapes_buf: Buffer,
    atlas: Atlas,
    mesh: Pipes,
    impostor: Pipes,
    cull_shapes: [Vec4; SLOTS],
    trees_buf: Buffer,
}

pub struct ViewSetup {
    pub planes: [Vec4; CULL_PLANES],
    pub ref_pos: Vec3,
    pub max_dist: f32,
    pub lod: Vec4,
    /// The camera's view matrix and projection scale (x, y), to cull its distant LODs in a
    /// second pass against the depth of the near ones (`cull_occluded`). Only for view 0.
    pub occlusion: Option<(Mat4, glam::Vec2)>,
}

/// Constants the tree and culling shaders are built with.
fn shader_consts() -> String {
    format!(
        "const SLOTS: u32 = {SLOTS}u;\nconst DRAWS: u32 = {DRAWS}u;\nconst IMP_FRAMES: u32 = {}u;\nconst IMP_WARP: f32 = {:?};\n",
        impostor::FRAMES,
        impostor::WARP
    )
}

/// For every slot, the shape that fills it: a slot without a shape of its own falls back to
/// its species' first (the scatter never picks one).
fn slot_shapes(shapes: &[Shape]) -> [usize; SLOTS] {
    let mut out = [usize::MAX; SLOTS];
    for (i, s) in shapes.iter().enumerate() {
        out[s.slot()] = i;
    }
    for slot in 0..SLOTS {
        if out[slot] == usize::MAX {
            out[slot] = out[slot / VARIANTS * VARIANTS];
        }
    }
    out
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
        // ---- meshes: each shape's two mesh LODs, then the impostor quad
        let shapes = treegen::build_shapes();
        let mut verts: Vec<TreeVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut draws = [DrawRange::default(); DRAWS];
        for (s, shape) in shapes.iter().enumerate() {
            for (l, m) in shape.lods.iter().enumerate() {
                draws[s * MESH_LODS + l] = DrawRange {
                    index_count: m.idx.len() as u32,
                    first_index: idx.len() as u32,
                    base_vertex: verts.len() as i32,
                    tris: m.tris(),
                    lod: l,
                };
                verts.extend_from_slice(&m.verts);
                idx.extend_from_slice(&m.idx);
            }
        }
        let quad = DrawRange { index_count: 6, first_index: idx.len() as u32, base_vertex: verts.len() as i32, tris: 2, lod: 0 };
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            verts.push(TreeVertex { pos: [x, y, 0.0], nrm: [0.0, 0.0, 1.0], uv: [0.0; 2], attr: [KIND_LEAF, 0.0, 1.0, NO_SLOT] });
        }
        idx.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        for l in MESH_LODS..LODS {
            draws[SHAPES * MESH_LODS + l - MESH_LODS] = DrawRange { lod: l, ..quad };
        }
        let vbuf = buffer_init(device, "tree verts", cast_slice(&verts), BufferUsages::VERTEX);
        let ibuf = buffer_init(device, "tree idx", cast_slice(&idx), BufferUsages::INDEX);

        // ---- the shape table, and what culling needs of it
        let fits: Vec<impostor::Fit> = shapes.iter().map(|s| impostor::fit(&s.lods[0])).collect();
        let by_slot = slot_shapes(&shapes);
        let mut gpu_shapes = [ShapeGpu::zeroed(); SLOTS];
        let mut cull_shapes = [Vec4::ZERO; SLOTS];
        for slot in 0..SLOTS {
            let i = by_slot[slot];
            let (s, f) = (&shapes[i], &fits[i]);
            gpu_shapes[slot] = ShapeGpu {
                crown_c: s.crown_center.extend(s.density),
                crown_r: s.crown_radii.extend(s.species as f32),
                imp: Vec4::new(f.center_y, f.half, f.depth, i as f32),
            };
            cull_shapes[slot] = Vec4::new(s.center_y, s.radius, (i * MESH_LODS) as f32, 0.0);
        }
        let shapes_buf = buffer_init(device, "tree shapes", cast_slice(&gpu_shapes), BufferUsages::UNIFORM);

        // ---- impostors
        let consts = shader_consts();
        let atlas = Atlas::new(device, SHAPES as u32);
        let items: Vec<BakeItem> = (0..SHAPES)
            .map(|i| {
                let d = draws[i * MESH_LODS];
                BakeItem { layer: i as u32, index_count: d.index_count, first_index: d.first_index, base_vertex: d.base_vertex }
            })
            .collect();
        let tree_common = include_str!("shaders/tree_common.wgsl");
        impostor::bake(
            device,
            queue,
            global_bgl,
            global_bg,
            &[&consts, NOISE_WGSL, GLOBALS_WGSL, tree_common, include_str!("shaders/impostor_bake.wgsl")],
            &shapes_buf,
            &vbuf,
            &ibuf,
            &items,
            &atlas,
        );

        // ---- scatter
        let dim = (world / SCATTER_CELL) as u32;
        let cap = dim * dim;
        let trees_buf = buffer(device, "trees", cap as u64 * 16, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
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
        println!("forest: scattered {tree_count} trees and shrubs from {cap} candidate cells");

        // ---- culling
        let cull_bgl = bgl(
            device,
            "cull",
            &[
                uniform_entry(0, ShaderStages::COMPUTE),
                storage_entry(1, ShaderStages::COMPUTE, true),
                storage_entry(2, ShaderStages::COMPUTE, false),
                storage_entry(3, ShaderStages::COMPUTE, false),
                texture_entry(4, ShaderStages::COMPUTE, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
            ],
        );
        // stands in for the depth pyramid in the passes that don't test against it
        let dummy_hiz = device.create_texture(&TextureDescriptor {
            label: Some("dummy hiz"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_hiz_view = dummy_hiz.create_view(&Default::default());
        let module = shader(device, "cull", &[&consts, NOISE_WGSL, GLOBALS_WGSL, include_str!("shaders/cull.wgsl")]);
        let pl = pipeline_layout(device, &[global_bgl, &cull_bgl]);
        let cull_pipe = compute_pipeline(device, "cull", &pl, &module, "main");

        // Room in each bucket: a mesh bucket only ever holds the trees of one shape near the
        // view; an impostor bucket may need every tree.
        let main_caps = [3_000u32, 20_000];
        let shadow_caps = [3_000u32, 12_000];
        let mut views = Vec::new();
        for caps in [main_caps, shadow_caps, shadow_caps] {
            let mut regions = [[0u32; 4]; DRAWS];
            let mut cursor = 0;
            for (d, r) in regions.iter_mut().enumerate() {
                let cap = if draws[d].impostor() { tree_count.max(1) } else { caps[draws[d].lod] };
                *r = [cursor, cap, 0, 0];
                cursor += cap;
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
                &[
                    params_buf.as_entire_binding(),
                    trees_buf.as_entire_binding(),
                    args_buf.as_entire_binding(),
                    visible_buf.as_entire_binding(),
                    BindingResource::TextureView(&dummy_hiz_view),
                ],
            );
            views.push(CullView { params_buf, args_buf, visible_buf, bind_group, regions, reset_args });
        }

        // ---- rendering
        let tree_bgl = bgl(
            device,
            "trees",
            &[
                storage_entry(0, ShaderStages::VERTEX, true),
                uniform_entry(1, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
                texture_entry(2, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2Array),
                texture_entry(3, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2Array),
            ],
        );
        let tree_bg = Self::records_bg(device, &tree_bgl, &trees_buf, &shapes_buf, &atlas);
        let module = shader(device, "tree", &[&consts, NOISE_WGSL, GLOBALS_WGSL, tree_common, include_str!("shaders/tree.wgsl")]);
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
        // Without MSAA there is no coverage to cut leaves with, so the prepass discards instead.
        let depth_targets = [Some(ColorTargetState { format: hdr, blend: None, write_mask: ColorWrites::empty() })];
        let pipe = |label: &str, vs: &str, fs: &str, depth: DepthStencilState, cutout: bool, targets: &[Option<ColorTargetState>]| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState { module: &module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &buffers },
                primitive: PrimitiveState::default(),
                depth_stencil: Some(depth),
                multisample: MultisampleState { count: if targets.is_empty() { 1 } else { msaa }, mask: !0, alpha_to_coverage_enabled: cutout && msaa > 1 },
                fragment: Some(FragmentState { module: &module, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let write_depth = depth_state(TextureFormat::Depth32Float, true, CompareFunction::GreaterEqual);
        let equal = depth_state(TextureFormat::Depth32Float, false, CompareFunction::Equal);
        let pipes = |name: &str, vs: &str, fs: &str| {
            let depth_fs = if msaa > 1 { format!("{fs}_depth") } else { format!("{fs}_depth_test") };
            Pipes {
                shadow: pipe(&format!("{name} shadow"), vs, &format!("{fs}_shadow"), shadow_depth_state(), false, &[]),
                color: pipe(name, vs, fs, write_depth.clone(), true, &targets),
                depth: pipe(&format!("{name} depth"), vs, &depth_fs, write_depth.clone(), true, &depth_targets),
                color_equal: pipe(&format!("{name} shade"), vs, fs, equal.clone(), false, &targets),
            }
        };
        let mesh = pipes("tree", "vs", "fs");
        let impostor = pipes("tree impostor", "vs_impostor", "fs_impostor");

        let late_params = buffer(device, "cull params late", std::mem::size_of::<CullParams>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        Forest {
            vbuf,
            ibuf,
            draws,
            tree_count,
            views,
            cull_pipe,
            cull_bgl,
            late_params,
            late_bg: None,
            hiz_info: Vec4::ZERO,
            _dummy_hiz: dummy_hiz,
            tree_bg,
            tree_bgl,
            shapes_buf,
            atlas,
            mesh,
            impostor,
            cull_shapes,
            trees_buf,
        }
    }

    fn records_bg(device: &Device, layout: &BindGroupLayout, records: &Buffer, shapes: &Buffer, atlas: &Atlas) -> BindGroup {
        bind_group(
            device,
            "trees",
            layout,
            &[
                records.as_entire_binding(),
                shapes.as_entire_binding(),
                BindingResource::TextureView(&atlas.color_view),
                BindingResource::TextureView(&atlas.data_view),
            ],
        )
    }

    /// Writes the impostor atlas as PNGs (`--impostor-png`).
    pub fn debug_impostors(&self, device: &Device, queue: &Queue, path: &str) {
        self.atlas.debug_png(device, queue, path);
    }

    /// Group 1 of the tree pipelines for other records in the trees' layout (undergrowth.rs).
    pub fn records_bind_group(&self, device: &Device, records: &Buffer) -> BindGroup {
        Self::records_bg(device, &self.tree_bgl, records, &self.shapes_buf, &self.atlas)
    }

    /// One-time copy of the tree records (see scatter.wgsl) to the CPU, for trunk collision.
    pub fn read_trees(&self, device: &Device, queue: &Queue) -> Vec<[u32; 4]> {
        let size = self.tree_count as u64 * 16;
        if size == 0 {
            return Vec::new();
        }
        let rb = buffer(device, "trees rb", size, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.trees_buf, 0, &rb, 0, size);
        queue.submit([enc.finish()]);
        read_buffer_blocking(device, &rb, size).as_chunks::<16>().0.iter().map(|c| bytemuck::cast(*c)).collect()
    }

    /// The camera's second culling pass tests against this depth pyramid (`Hiz`); call again
    /// whenever it is rebuilt.
    pub fn set_hiz(&mut self, device: &Device, view: &TextureView, info: Vec4) {
        let v = &self.views[0];
        self.late_bg = Some(bind_group(
            device,
            "cull late",
            &self.cull_bgl,
            &[
                self.late_params.as_entire_binding(),
                self.trees_buf.as_entire_binding(),
                v.args_buf.as_entire_binding(),
                v.visible_buf.as_entire_binding(),
                BindingResource::TextureView(view),
            ],
        ));
        self.hiz_info = info;
    }

    pub fn update_view(&self, queue: &Queue, view: usize, setup: &ViewSetup) {
        let v = &self.views[view];
        let first_impostor = (SHAPES * MESH_LODS) as u32;
        let mut params = CullParams {
            planes: setup.planes,
            cam: setup.ref_pos.extend(setup.max_dist),
            lod: setup.lod,
            counts: [self.tree_count, MODE_ALL, OCCLUDER_LODS as u32, first_impostor],
            shapes: self.cull_shapes,
            regions: v.regions,
            view: Mat4::IDENTITY,
            proj: Vec4::ZERO,
            hiz: Vec4::ZERO,
        };
        if let Some((cam_view, scale)) = setup.occlusion {
            assert_eq!(view, 0, "only the camera has a second culling pass");
            params.counts[1] = MODE_OCCLUDERS;
            let late = CullParams {
                counts: [self.tree_count, MODE_OCCLUDED, OCCLUDER_LODS as u32, first_impostor],
                view: cam_view,
                proj: Vec4::new(scale.x, scale.y, crate::camera::NEAR, 0.0),
                hiz: self.hiz_info,
                ..params
            };
            queue.write_buffer(&self.late_params, 0, bytemuck::bytes_of(&late));
        }
        queue.write_buffer(&v.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&v.args_buf, 0, cast_slice(&v.reset_args));
    }

    pub fn cull(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup, view: usize) {
        self.dispatch_cull(cp, global_bg, &self.views[view].bind_group);
    }

    /// The camera's second pass: appends its distant LODs that the depth pyramid doesn't hide.
    pub fn cull_occluded(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup) {
        self.dispatch_cull(cp, global_bg, self.late_bg.as_ref().expect("set_hiz before cull_occluded"));
    }

    fn dispatch_cull(&self, cp: &mut ComputePass<'_>, global_bg: &BindGroup, bg: &BindGroup) {
        let groups = self.tree_count.div_ceil(256);
        let (x, y) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups, 1) };
        cp.set_pipeline(&self.cull_pipe);
        cp.set_bind_group(0, global_bg, &[]);
        cp.set_bind_group(1, bg, &[]);
        cp.dispatch_workgroups(x, y, 1);
    }

    /// The shaded, alpha-to-coverage mesh pipeline, for other instanced foliage (undergrowth.rs).
    pub fn color_pipeline(&self) -> &RenderPipeline {
        &self.mesh.color
    }

    /// Draws the LODs in `lods` (0 = full detail, 2 and 3 = impostors).
    pub fn draw(&self, pass: &mut RenderPass<'_>, view: usize, kind: TreePass, lods: std::ops::Range<usize>) {
        let v = &self.views[view];
        pass.set_bind_group(1, &self.tree_bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), IndexFormat::Uint32);
        let mut bound = None;
        for (d, draw) in self.draws.iter().enumerate() {
            if !lods.contains(&draw.lod) {
                continue;
            }
            if bound != Some(draw.impostor()) {
                pass.set_pipeline(if draw.impostor() { self.impostor.get(kind) } else { self.mesh.get(kind) });
                bound = Some(draw.impostor());
            }
            let start = v.regions[d][0] as u64 * 4;
            pass.set_vertex_buffer(1, v.visible_buf.slice(start..));
            pass.draw_indexed_indirect(&v.args_buf, (d * 20) as u64);
        }
    }
}
