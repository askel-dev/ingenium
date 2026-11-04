//! The depth pyramid that the second culling pass tests distant trees against
//! (`shaders/hiz.wgsl`). A texel of level L holds the farthest depth of every sample in the
//! 2^(L+1) x 2^(L+1) pixels starting at its index times that, so a pixel's texel at any level
//! is just its coordinate shifted right (the last texel of a row or column may cover fewer).
//! Each depth sample is read once.

use crate::util::*;
use glam::Vec4;
use wgpu::*;

const FORMAT: TextureFormat = TextureFormat::R32Float;

pub struct Hiz {
    first_bgl: BindGroupLayout,
    reduce_bgl: BindGroupLayout,
    first_pipe: ComputePipeline,
    reduce_pipe: ComputePipeline,
    t: Targets,
}

struct Targets {
    /// Every level, for the culling pass.
    view: TextureView,
    /// The depth buffer's size.
    screen: (u32, u32),
    /// Each level's size.
    sizes: Vec<(u32, u32)>,
    first_bg: BindGroup,
    /// One per level above 0: reads the level below, writes this one.
    reduce_bgs: Vec<BindGroup>,
}

/// The size of each level for a depth buffer of `w` x `h`: halved and rounded up, down to 1 x 1.
fn level_sizes(w: u32, h: u32) -> Vec<(u32, u32)> {
    let mut sizes = vec![(w.div_ceil(2).max(1), h.div_ceil(2).max(1))];
    while *sizes.last().unwrap() != (1, 1) {
        let (w, h) = *sizes.last().unwrap();
        sizes.push((w.div_ceil(2), h.div_ceil(2)));
    }
    sizes
}

/// The texture that holds them: mip L of a power of two halves rounding down, so level 0 is the
/// power of two whose mips stay at least as big as every level (`level_sizes`).
fn storage_size(w: u32, h: u32) -> (u32, u32) {
    ((w.next_power_of_two() / 2).max(1), (h.next_power_of_two() / 2).max(1))
}

impl Hiz {
    pub fn new(device: &Device, depth: &TextureView, msaa: u32, width: u32, height: u32) -> Hiz {
        let first_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("hiz first"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: msaa > 1 },
                    count: None,
                },
                storage_texture_entry(1, FORMAT),
            ],
        });
        let reduce_bgl = bgl(
            device,
            "hiz reduce",
            &[
                storage_texture_entry(1, FORMAT),
                texture_entry(2, ShaderStages::COMPUTE, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2),
                uniform_entry(3, ShaderStages::COMPUTE),
            ],
        );
        let depth_decl = if msaa > 1 {
            format!("const SAMPLES: u32 = {msaa}u;\n@group(0) @binding(0) var t_depth: texture_depth_multisampled_2d;\n")
        } else {
            "const SAMPLES: u32 = 1u;\n@group(0) @binding(0) var t_depth: texture_depth_2d;\n".to_string()
        };
        let module = shader(device, "hiz", &[&depth_decl, include_str!("shaders/hiz.wgsl")]);
        let first_pipe = compute_pipeline(device, "hiz first", &pipeline_layout(device, &[&first_bgl]), &module, "first");
        let reduce_pipe = compute_pipeline(device, "hiz reduce", &pipeline_layout(device, &[&reduce_bgl]), &module, "reduce");
        let t = Self::make(device, &first_bgl, &reduce_bgl, depth, width, height);
        Hiz { first_bgl, reduce_bgl, first_pipe, reduce_pipe, t }
    }

    pub fn resize(&mut self, device: &Device, depth: &TextureView, width: u32, height: u32) {
        self.t = Self::make(device, &self.first_bgl, &self.reduce_bgl, depth, width, height);
    }

    fn make(device: &Device, first_bgl: &BindGroupLayout, reduce_bgl: &BindGroupLayout, depth: &TextureView, width: u32, height: u32) -> Targets {
        let sizes = level_sizes(width, height);
        let levels = sizes.len() as u32;
        let storage = storage_size(width, height);
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("hiz"),
            size: Extent3d { width: storage.0, height: storage.1, depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: FORMAT,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let level = |l: u32| tex.create_view(&TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() });
        let first_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("hiz first"),
            layout: first_bgl,
            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(depth) }, BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&level(0)) }],
        });
        let reduce_bgs = (1..levels as usize)
            .map(|l| {
                let (below, here) = (sizes[l - 1], sizes[l]);
                let buf = buffer_init(device, "hiz sizes", bytemuck::cast_slice(&[below.0, below.1, here.0, here.1]), BufferUsages::UNIFORM);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("hiz reduce"),
                    layout: reduce_bgl,
                    entries: &[
                        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&level(l as u32)) },
                        BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&level(l as u32 - 1)) },
                        BindGroupEntry { binding: 3, resource: buf.as_entire_binding() },
                    ],
                })
            })
            .collect();
        Targets { view: tex.create_view(&Default::default()), screen: (width, height), sizes, first_bg, reduce_bgs }
    }

    /// Every level of the pyramid.
    pub fn view(&self) -> &TextureView {
        &self.t.view
    }

    /// The depth buffer's width and height, the number of levels, then 0.
    pub fn info(&self) -> Vec4 {
        Vec4::new(self.t.screen.0 as f32, self.t.screen.1 as f32, self.t.sizes.len() as f32, 0.0)
    }

    /// Rebuild the pyramid from the depth buffer.
    pub fn build(&self, cp: &mut ComputePass<'_>) {
        cp.set_pipeline(&self.first_pipe);
        cp.set_bind_group(0, &self.t.first_bg, &[]);
        let (w, h) = self.t.sizes[0];
        cp.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        cp.set_pipeline(&self.reduce_pipe);
        for (bg, &(w, h)) in self.t.reduce_bgs.iter().zip(&self.t.sizes[1..]) {
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{level_sizes, storage_size};

    #[test]
    fn every_pixel_lands_in_a_texel_of_every_level() {
        for (w, h) in [(1, 1), (2, 1), (1600, 900), (801, 451), (3, 1000)] {
            let sizes = level_sizes(w, h);
            assert_eq!(*sizes.last().unwrap(), (1, 1));
            for (l, &(lw, lh)) in sizes.iter().enumerate() {
                // the last pixel's texel is the last texel: none is left over or missing
                assert_eq!(((w - 1) >> (l + 1), (h - 1) >> (l + 1)), (lw - 1, lh - 1), "{w}x{h} level {l}");
                // and the texture's mip is big enough for it
                let (sw, sh) = storage_size(w, h);
                assert!((sw >> l).max(1) >= lw && (sh >> l).max(1) >= lh, "{w}x{h} level {l}");
            }
            let (sw, sh) = storage_size(w, h);
            assert!(32 - sw.max(sh).leading_zeros() >= sizes.len() as u32, "{w}x{h}: too few mips");
        }
    }
}
