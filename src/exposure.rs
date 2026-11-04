//! Auto exposure metering (`shaders/exposure.wgsl`): the log-average luminance of the frame,
//! eased over time. The post pass opens up from it when the view is darker than open ground.

use crate::util::*;
use wgpu::*;

/// Seconds to adapt about 63% of the way to a new light level.
const TAU: f32 = 0.45;

pub struct Exposure {
    pipe: ComputePipeline,
    bgl: BindGroupLayout,
    bg: BindGroup,
    params: Buffer,
    /// x = adapted log luminance (read by the post pass)
    pub state: Buffer,
}

impl Exposure {
    pub fn new(device: &Device, hdr: &TextureView) -> Exposure {
        let bgl = bgl(
            device,
            "exposure",
            &[
                texture_entry(0, ShaderStages::COMPUTE, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
                uniform_entry(1, ShaderStages::COMPUTE),
                storage_entry(2, ShaderStages::COMPUTE, false),
            ],
        );
        let module = shader(device, "exposure", &[include_str!("shaders/exposure.wgsl")]);
        let pipe = compute_pipeline(device, "exposure", &pipeline_layout(device, &[&bgl]), &module, "main");
        let params = buffer(device, "exposure params", 16, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let state = buffer(device, "exposure state", 16, BufferUsages::STORAGE);
        let bg = Self::bind(device, &bgl, hdr, &params, &state);
        Exposure { pipe, bgl, bg, params, state }
    }

    fn bind(device: &Device, bgl: &BindGroupLayout, hdr: &TextureView, params: &Buffer, state: &Buffer) -> BindGroup {
        bind_group(device, "exposure", bgl, &[BindingResource::TextureView(hdr), params.as_entire_binding(), state.as_entire_binding()])
    }

    /// Call when the HDR target is recreated.
    pub fn set_target(&mut self, device: &Device, hdr: &TextureView) {
        self.bg = Self::bind(device, &self.bgl, hdr, &self.params, &self.state);
    }

    pub fn record(&self, queue: &Queue, enc: &mut CommandEncoder, time: f32) {
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[time, TAU, 0.0, 0.0]));
        let mut cp = enc.begin_compute_pass(&ComputePassDescriptor { label: Some("exposure"), timestamp_writes: None });
        cp.set_pipeline(&self.pipe);
        cp.set_bind_group(0, &self.bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
}
