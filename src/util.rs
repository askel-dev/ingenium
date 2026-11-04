//! Small helpers that keep the wgpu boilerplate readable.

use wgpu::util::DeviceExt;
use wgpu::*;

pub const NOISE_WGSL: &str = include_str!("shaders/noise.wgsl");
pub const ATMO_WGSL: &str = include_str!("shaders/atmo_common.wgsl");
/// The per-frame globals; they read the atmosphere tables, so they bring the atmosphere code along.
pub const GLOBALS_WGSL: &str = concat!(include_str!("shaders/atmo_common.wgsl"), include_str!("shaders/globals.wgsl"));

pub fn uniform_entry(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn storage_entry(binding: u32, visibility: ShaderStages, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn texture_entry(
    binding: u32,
    visibility: ShaderStages,
    sample_type: TextureSampleType,
    view_dimension: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Texture { sample_type, view_dimension, multisampled: false },
        count: None,
    }
}

pub fn storage_texture_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32, visibility: ShaderStages, ty: SamplerBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry { binding, visibility, ty: BindingType::Sampler(ty), count: None }
}

pub fn bgl(device: &Device, label: &str, entries: &[BindGroupLayoutEntry]) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some(label), entries })
}

pub fn bind_group(device: &Device, label: &str, layout: &BindGroupLayout, resources: &[BindingResource]) -> BindGroup {
    let entries: Vec<BindGroupEntry> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| BindGroupEntry { binding: i as u32, resource: r.clone() })
        .collect();
    device.create_bind_group(&BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

pub fn shader(device: &Device, label: &str, sources: &[&str]) -> ShaderModule {
    let src = sources.concat();
    device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(src.into()) })
}

pub fn pipeline_layout(device: &Device, layouts: &[&BindGroupLayout]) -> PipelineLayout {
    let v: Vec<Option<&BindGroupLayout>> = layouts.iter().map(|b| Some(*b)).collect();
    device.create_pipeline_layout(&PipelineLayoutDescriptor { label: None, bind_group_layouts: &v, immediate_size: 0 })
}

pub fn compute_pipeline(device: &Device, label: &str, layout: &PipelineLayout, module: &ShaderModule, entry: &str) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

pub fn buffer(device: &Device, label: &str, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

pub fn buffer_init(device: &Device, label: &str, contents: &[u8], usage: BufferUsages) -> Buffer {
    device.create_buffer_init(&util::BufferInitDescriptor { label: Some(label), contents, usage })
}

/// Blocking readback of a buffer's first `size` bytes (startup only).
pub fn read_buffer_blocking(device: &Device, buf: &Buffer, size: u64) -> Vec<u8> {
    let slice = buf.slice(0..size);
    slice.map_async(MapMode::Read, |r| r.expect("map failed"));
    device.poll(PollType::wait_indefinitely()).expect("poll failed");
    let out = slice.get_mapped_range().expect("mapped range").to_vec();
    buf.unmap();
    out
}

pub fn depth_state(format: TextureFormat, write: bool, compare: CompareFunction) -> DepthStencilState {
    DepthStencilState {
        format,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    }
}

pub fn shadow_depth_state() -> DepthStencilState {
    DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(CompareFunction::LessEqual),
        stencil: StencilState::default(),
        bias: DepthBiasState { constant: 2, slope_scale: 2.5, clamp: 0.0 },
    }
}

/// Six frustum planes (inside: dot(n, p) + d >= 0) from a view-projection matrix.
/// Degenerate planes (e.g. the far plane of an infinite projection) become "always pass".
pub fn frustum_planes(m: glam::Mat4) -> [glam::Vec4; 6] {
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    let r3 = m.row(3);
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
    raw.map(|p| {
        let len = p.truncate().length();
        if len < 1e-6 { glam::Vec4::new(0.0, 0.0, 0.0, 1.0) } else { p / len }
    })
}
