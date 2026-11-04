//! HDR bloom: a mip chain from half resolution down, filtered down and then added back up
//! (`shaders/bloom.wgsl`). The post pass mixes the top level into the image.

use crate::util::*;
use wgpu::*;

const LEVELS: u32 = 6;

pub struct Bloom {
    bgl: BindGroupLayout,
    sampler: Sampler,
    down_first: RenderPipeline,
    down: RenderPipeline,
    up: RenderPipeline,
    mips: Vec<TextureView>,
    /// down_bgs[i] reads the input of level i; up_bgs[i] reads level i + 1.
    down_bgs: Vec<BindGroup>,
    up_bgs: Vec<BindGroup>,
}

impl Bloom {
    pub fn new(device: &Device, format: TextureFormat, hdr: &TextureView, width: u32, height: u32) -> Bloom {
        let bgl = bgl(
            device,
            "bloom",
            &[
                texture_entry(0, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
                sampler_entry(1, ShaderStages::FRAGMENT, SamplerBindingType::Filtering),
            ],
        );
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("bloom"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let module = shader(device, "bloom", &[include_str!("shaders/bloom.wgsl")]);
        let layout = pipeline_layout(device, &[&bgl]);
        let pipe = |entry: &str, blend: Option<BlendState>| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState { format, blend, write_mask: ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let add = BlendState {
            color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            alpha: BlendComponent::REPLACE,
        };
        let mut b = Bloom {
            down_first: pipe("fs_down_first", None),
            down: pipe("fs_down", None),
            up: pipe("fs_up", Some(add)),
            bgl,
            sampler,
            mips: Vec::new(),
            down_bgs: Vec::new(),
            up_bgs: Vec::new(),
        };
        b.resize(device, format, hdr, width, height);
        b
    }

    pub fn resize(&mut self, device: &Device, format: TextureFormat, hdr: &TextureView, width: u32, height: u32) {
        let (w, h) = ((width / 2).max(1), (height / 2).max(1));
        let levels = LEVELS.min(32 - w.min(h).leading_zeros());
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("bloom"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.mips = (0..levels)
            .map(|l| tex.create_view(&TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() }))
            .collect();
        let bg = |v: &TextureView| bind_group(device, "bloom", &self.bgl, &[BindingResource::TextureView(v), BindingResource::Sampler(&self.sampler)]);
        self.down_bgs = (0..levels as usize).map(|i| bg(if i == 0 { hdr } else { &self.mips[i - 1] })).collect();
        self.up_bgs = (0..levels as usize - 1).map(|i| bg(&self.mips[i + 1])).collect();
    }

    /// The finished bloom (half resolution), the sum of all levels.
    pub fn output(&self) -> &TextureView {
        &self.mips[0]
    }

    pub fn levels(&self) -> u32 {
        self.mips.len() as u32
    }

    /// `timestamp` is written by the last pass.
    pub fn record(&self, enc: &mut CommandEncoder, mut timestamp: Option<RenderPassTimestampWrites>) {
        let n = self.mips.len();
        let down = (0..n).map(|i| (i, LoadOp::Clear(Color::BLACK), if i == 0 { &self.down_first } else { &self.down }, &self.down_bgs[i]));
        let up = (0..n - 1).rev().map(|i| (i, LoadOp::Load, &self.up, &self.up_bgs[i]));
        let passes: Vec<_> = down.chain(up).collect();
        let count = passes.len();
        for (k, (i, load, pipe, bg)) in passes.into_iter().enumerate() {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("bloom"),
                color_attachments: &[Some(RenderPassColorAttachment { view: &self.mips[i], depth_slice: None, resolve_target: None, ops: Operations { load, store: StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: if k + 1 == count { timestamp.take() } else { None },
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, bg, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
