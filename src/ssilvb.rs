//! Screen-space indirect light with a visibility bitmask, and contact shadows, at half
//! resolution (`shaders/ssilvb.wgsl`): a half-res copy of the depth buffer with normals, one pass
//! that gathers the light the surfaces around each pixel send it and how much of its sky they
//! hide, and marches a short ray towards the sun, and a depth-aware blur. The post pass
//! composites the result with the main pass's aux target (albedo and direct share).

use crate::util::*;
use wgpu::*;

const PREP: TextureFormat = TextureFormat::Rgba32Float;
const GI: TextureFormat = TextureFormat::Rgba16Float;
const CONTACT: TextureFormat = TextureFormat::R8Unorm;

pub struct Ssilvb {
    depth_bgl: BindGroupLayout,
    gi_bgl: BindGroupLayout,
    blur_bgl: BindGroupLayout,
    prep_pipe: RenderPipeline,
    gi_pipe: RenderPipeline,
    blur_pipe: RenderPipeline,
    t: Targets,
}

struct Targets {
    prep: TextureView,
    raw: TextureView,
    raw_cs: TextureView,
    out: TextureView,
    out_cs: TextureView,
    depth_bg: BindGroup,
    gi_bg: BindGroup,
    blur_bg: BindGroup,
}

fn tex_entry(binding: u32, filterable: bool) -> BindGroupLayoutEntry {
    texture_entry(binding, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable }, TextureViewDimension::D2)
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

fn group(device: &Device, label: &str, layout: &BindGroupLayout, views: &[(u32, &TextureView)]) -> BindGroup {
    let entries: Vec<BindGroupEntry> = views.iter().map(|(binding, v)| BindGroupEntry { binding: *binding, resource: BindingResource::TextureView(v) }).collect();
    device.create_bind_group(&BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

impl Ssilvb {
    pub fn new(device: &Device, global_bgl: &BindGroupLayout, depth: &TextureView, hdr: &TextureView, msaa: u32, width: u32, height: u32) -> Ssilvb {
        let depth_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssilvb depth"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: msaa > 1 },
                count: None,
            }],
        });
        let gi_bgl = bgl(device, "ssilvb", &[tex_entry(1, false), tex_entry(2, true)]);
        let blur_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some("ssilvb blur"), entries: &[tex_entry(1, false), tex_entry(3, false), tex_entry(4, false)] });
        let depth_decl = if msaa > 1 {
            "@group(1) @binding(0) var t_depth: texture_depth_multisampled_2d;\n"
        } else {
            "@group(1) @binding(0) var t_depth: texture_depth_2d;\n"
        };
        let module = shader(device, "ssilvb", &[NOISE_WGSL, GLOBALS_WGSL, depth_decl, include_str!("shaders/ssilvb.wgsl")]);
        let pipe = |label: &str, entry: &str, layout: &PipelineLayout, formats: &[TextureFormat]| {
            let targets: Vec<Option<ColorTargetState>> = formats.iter().map(|&format| Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })).collect();
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState { module: &module, entry_point: Some(entry), compilation_options: Default::default(), targets: &targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let prep_pipe = pipe("ssilvb depth", "fs_prep", &pipeline_layout(device, &[global_bgl, &depth_bgl]), &[PREP]);
        let gi_pipe = pipe("ssilvb", "fs_gi", &pipeline_layout(device, &[global_bgl, &gi_bgl]), &[GI, CONTACT]);
        let blur_pipe = pipe("ssilvb blur", "fs_blur", &pipeline_layout(device, &[global_bgl, &blur_bgl]), &[GI, CONTACT]);
        let t = Self::make(device, &depth_bgl, &gi_bgl, &blur_bgl, depth, hdr, width, height);
        Ssilvb { depth_bgl, gi_bgl, blur_bgl, prep_pipe, gi_pipe, blur_pipe, t }
    }

    #[allow(clippy::too_many_arguments)]
    fn make(device: &Device, depth_bgl: &BindGroupLayout, gi_bgl: &BindGroupLayout, blur_bgl: &BindGroupLayout, depth: &TextureView, hdr: &TextureView, width: u32, height: u32) -> Targets {
        let (w, h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
        let prep = target(device, "ssilvb half depth", PREP, w, h);
        let raw = target(device, "ssilvb raw", GI, w, h);
        let raw_cs = target(device, "ssilvb raw contact", CONTACT, w, h);
        let out = target(device, "ssilvb", GI, w, h);
        let out_cs = target(device, "ssilvb contact", CONTACT, w, h);
        let depth_bg = group(device, "ssilvb depth", depth_bgl, &[(0, depth)]);
        let gi_bg = group(device, "ssilvb", gi_bgl, &[(1, &prep), (2, hdr)]);
        let blur_bg = group(device, "ssilvb blur", blur_bgl, &[(1, &prep), (3, &raw), (4, &raw_cs)]);
        Targets { prep, raw, raw_cs, out, out_cs, depth_bg, gi_bg, blur_bg }
    }

    pub fn resize(&mut self, device: &Device, depth: &TextureView, hdr: &TextureView, width: u32, height: u32) {
        self.t = Self::make(device, &self.depth_bgl, &self.gi_bgl, &self.blur_bgl, depth, hdr, width, height);
    }

    /// Indirect light (rgb, times the albedo) and the visibility of the ambient light (a), at
    /// half resolution.
    pub fn output(&self) -> &TextureView {
        &self.t.out
    }

    /// The share of the direct sunlight left after the contact shadows, at half resolution.
    pub fn contact(&self) -> &TextureView {
        &self.t.out_cs
    }

    fn attachments<'a>(views: &[&'a TextureView], clear: Color) -> Vec<Option<RenderPassColorAttachment<'a>>> {
        views.iter().map(|v| Some(RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(clear), store: StoreOp::Store } })).collect()
    }

    /// Switched off: no indirect light, nothing occluded, no contact shadows.
    pub fn clear(&self, enc: &mut CommandEncoder) {
        for (view, clear) in [(&self.t.out, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), (&self.t.out_cs, Color::WHITE)] {
            enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssilvb off"),
                color_attachments: &Self::attachments(&[view], clear),
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
    }

    /// `timestamp` is written by the last of the three passes.
    pub fn record(&self, enc: &mut CommandEncoder, global: &BindGroup, mut timestamp: Option<RenderPassTimestampWrites>) {
        let t = &self.t;
        let passes: [(Vec<&TextureView>, &RenderPipeline, &BindGroup); 3] =
            [(vec![&t.prep], &self.prep_pipe, &t.depth_bg), (vec![&t.raw, &t.raw_cs], &self.gi_pipe, &t.gi_bg), (vec![&t.out, &t.out_cs], &self.blur_pipe, &t.blur_bg)];
        let n = passes.len();
        for (k, (views, pipe, bg)) in passes.into_iter().enumerate() {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssilvb"),
                color_attachments: &Self::attachments(&views, Color::BLACK),
                depth_stencil_attachment: None,
                timestamp_writes: if k + 1 == n { timestamp.take() } else { None },
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
