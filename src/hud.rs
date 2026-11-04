//! The stats overlay in the window's top-right corner: "60.0 fps   cpu 0.84 ms   gpu 7.10 ms".
//! A built-in 5x9 pixel font, one quad, drawn over the finished frame (`shaders/hud.wgsl`).

use crate::util::*;
use wgpu::*;

const MAX_CHARS: usize = 48;

/// The glyphs the overlay can show, in index order; anything else is drawn as a space.
const CHARS: &str = " 0123456789.fpscugmx";

/// 9 rows of 5 pixels per glyph (7 above the baseline, 2 for descenders), bit 4 the leftmost
/// pixel, same order as `CHARS`.
#[rustfmt::skip]
const FONT: [[u8; 9]; 20] = [
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // ' '
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110, 0b00000, 0b00000], // 0
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000], // 1
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111, 0b00000, 0b00000], // 2
    [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110, 0b00000, 0b00000], // 3
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010, 0b00000, 0b00000], // 4
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110, 0b00000, 0b00000], // 5
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // 6
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00000, 0b00000], // 7
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110, 0b00000, 0b00000], // 8
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100, 0b00000, 0b00000], // 9
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100, 0b00000, 0b00000], // .
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000, 0b00000, 0b00000], // f
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000], // p
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110, 0b00000, 0b00000], // s
    [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110, 0b00000, 0b00000], // c
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101, 0b00000, 0b00000], // u
    [0b00000, 0b00000, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // g
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001, 0b00000, 0b00000], // m
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000, 0b00000], // x
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HudUniform {
    screen: [f32; 4],
    rect: [f32; 4],
    origin: [f32; 4],
    text: [[u32; 4]; MAX_CHARS / 4],
    font: [[u32; 4]; 15],
}

pub struct Hud {
    pipeline: RenderPipeline,
    buf: Buffer,
    bg: BindGroup,
    uni: HudUniform,
    pub visible: bool,
}

impl Hud {
    pub fn new(device: &Device, format: TextureFormat) -> Hud {
        let bgl = bgl(device, "hud", &[uniform_entry(0, ShaderStages::VERTEX_FRAGMENT)]);
        let module = shader(device, "hud", &[include_str!("shaders/hud.wgsl")]);
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hud"),
            layout: Some(&pipeline_layout(device, &[&bgl])),
            vertex: VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format, blend: Some(BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mut uni: HudUniform = bytemuck::Zeroable::zeroed();
        for (i, rows) in FONT.iter().enumerate() {
            for (r, &bits) in rows.iter().enumerate() {
                let k = i * 3 + r / 4;
                uni.font[k / 4][k % 4] |= (bits as u32) << ((r % 4) * 8);
            }
        }
        let buf = buffer(device, "hud", size_of::<HudUniform>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let bg = bind_group(device, "hud", &bgl, &[buf.as_entire_binding()]);
        Hud { pipeline, buf, bg, uni, visible: true }
    }

    pub fn set_text(&mut self, text: &str) {
        self.uni.text = [[0; 4]; MAX_CHARS / 4];
        let n = text.chars().count().min(MAX_CHARS);
        for (i, ch) in text.chars().take(n).enumerate() {
            self.uni.text[i / 4][i % 4] = CHARS.find(ch).unwrap_or(0) as u32;
        }
        self.uni.origin[3] = n as f32;
    }

    /// Draws the bar over `view` (the finished frame) in its own pass. `scale_factor` is the
    /// window's, so the text keeps its size on a Retina display.
    pub fn draw(&mut self, device: &Device, queue: &Queue, view: &TextureView, width: u32, height: u32, scale_factor: f64) {
        let n = self.uni.origin[3];
        if !self.visible || n == 0.0 {
            return;
        }
        let s = (1.5 * scale_factor).round().max(1.0) as f32;
        let (pad_x, pad_y) = (5.0 * s, 3.0 * s);
        let (w, h) = (width as f32, height as f32);
        let bar_w = (n * 6.0 - 1.0) * s + 2.0 * pad_x;
        let bar_h = 9.0 * s + 2.0 * pad_y;
        self.uni.screen = [w, h, 0.0, 0.0];
        self.uni.rect = [w - bar_w, 0.0, w, bar_h];
        self.uni.origin = [w - bar_w + pad_x, pad_y, s, n];
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&self.uni));
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("hud") });
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("hud"),
                color_attachments: &[Some(RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Load, store: StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bg, &[]);
            rp.draw(0..4, 0..1);
        }
        queue.submit([enc.finish()]);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn font_covers_every_char() {
        assert_eq!(super::CHARS.chars().count(), super::FONT.len());
    }
}
