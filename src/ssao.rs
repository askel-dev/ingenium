//! Screen-space ambient occlusion at half resolution (`shaders/ssao.wgsl`): a half-res copy
//! of the depth buffer, one pass that samples it and darkens only the ambient share of each
//! pixel's light, and a depth-aware blur. The post pass multiplies the image by the result.

use crate::util::*;
use wgpu::*;

const HALF_DEPTH: TextureFormat = TextureFormat::R32Float;
const RAW: TextureFormat = TextureFormat::Rg16Float;
const OUT: TextureFormat = TextureFormat::R8Unorm;

pub struct Ssao {
    depth_bgl: BindGroupLayout,
    hd_bgl: BindGroupLayout,
    blur_bgl: BindGroupLayout,
    prep_pipe: RenderPipeline,
    ao_pipe: RenderPipeline,
    blur_pipe: RenderPipeline,
    t: Targets,
}

struct Targets {
    hd: TextureView,
    raw: TextureView,
    out: TextureView,
    depth_bg: BindGroup,
    hd_bg: BindGroup,
    blur_bg: BindGroup,
}

fn float_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn single(device: &Device, label: &str, layout: &BindGroupLayout, binding: u32, view: &TextureView) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor { label: Some(label), layout, entries: &[BindGroupEntry { binding, resource: BindingResource::TextureView(view) }] })
}

fn target(device: &Device, label: &str, format: TextureFormat, w: u32, h: u32) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl Ssao {
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, depth: &TextureView, msaa: u32, width: u32, height: u32) -> Ssao {
        let depth_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssao depth"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: msaa > 1 },
                count: None,
            }],
        });
        let blur_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("ssao blur"), entries: &[float_entry(1)] });
        let hd_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("ssao half depth"), entries: &[float_entry(2)] });
        let depth_decl = if msaa > 1 {
            "@group(1) @binding(0) var t_depth: texture_depth_multisampled_2d;\n"
        } else {
            "@group(1) @binding(0) var t_depth: texture_depth_2d;\n"
        };
        let module = shader(device, "ssao", &[NOISE_WGSL, GLOBALS_WGSL, depth_decl, include_str!("shaders/ssao.wgsl")]);
        let pipe = |label: &str, entry: &str, layout: &PipelineLayout, format: TextureFormat| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let prep_pipe = pipe("ssao depth", "fs_prep", &pipeline_layout(device, &[global_bgl, &depth_bgl]), HALF_DEPTH);
        let ao_pipe = pipe("ssao", "fs_ao", &pipeline_layout(device, &[global_bgl, &hd_bgl]), RAW);
        let blur_pipe = pipe("ssao blur", "fs_blur", &pipeline_layout(device, &[global_bgl, &blur_bgl]), OUT);
        let t = Self::make(device, &depth_bgl, &hd_bgl, &blur_bgl, depth, width, height);
        Ssao { depth_bgl, hd_bgl, blur_bgl, prep_pipe, ao_pipe, blur_pipe, t }
    }

    fn make(device: &Device, depth_bgl: &BindGroupLayout, hd_bgl: &BindGroupLayout, blur_bgl: &BindGroupLayout, depth: &TextureView, width: u32, height: u32) -> Targets {
        let (w, h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
        let hd = target(device, "ssao half depth", HALF_DEPTH, w, h);
        let raw = target(device, "ssao raw", RAW, w, h);
        let out = target(device, "ssao", OUT, w, h);
        let depth_bg = single(device, "ssao depth", depth_bgl, 0, depth);
        let hd_bg = single(device, "ssao half depth", hd_bgl, 2, &hd);
        let blur_bg = single(device, "ssao blur", blur_bgl, 1, &raw);
        Targets { hd, raw, out, depth_bg, hd_bg, blur_bg }
    }

    pub fn resize(&mut self, device: &Device, depth: &TextureView, width: u32, height: u32) {
        self.t = Self::make(device, &self.depth_bgl, &self.hd_bgl, &self.blur_bgl, depth, width, height);
    }

    /// The occlusion multiplier, at half resolution.
    pub fn output(&self) -> &TextureView {
        &self.t.out
    }

    /// Switched off: leave no occlusion.
    pub fn clear(&self, enc: &mut CommandEncoder) {
        enc.begin_render_pass(&RenderPassDescriptor {
            label: Some("ssao off"),
            color_attachments: &[Some(RenderPassColorAttachment { view: &self.t.out, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    /// `timestamp` is written by the last of the three passes.
    pub fn record(&self, enc: &mut CommandEncoder, global: &BindGroup, mut timestamp: Option<RenderPassTimestampWrites>) {
        let t = &self.t;
        let passes = [(&t.hd, &self.prep_pipe, &t.depth_bg), (&t.raw, &self.ao_pipe, &t.hd_bg), (&t.out, &self.blur_pipe, &t.blur_bg)];
        for (k, (view, pipe, bg)) in passes.into_iter().enumerate() {
            let last = k + 1 == passes.len();
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssao"),
                color_attachments: &[Some(RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: if last { timestamp.take() } else { None },
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, global, &[]);
            rp.set_bind_group(1, bg, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
