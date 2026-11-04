//! Impostors for the distant trees. At startup every tree shape's full-detail mesh (LOD0) is
//! rendered from 8 × 8 directions over the upper hemisphere into one layer of a texture
//! atlas: colour and coverage in one texture, and in the other the normal, the depth and the
//! ambient occlusion. Beyond the LOD1 range a tree is a quad facing the camera that blends the
//! four views nearest to the direction it is seen from (`fs_impostor` in tree.wgsl). The
//! depth puts each texel back where its leaf is, for the shadow lookup, the crown's
//! self-shadowing and the haze, so the impostor is lit like the mesh it was baked from.
//!
//! Everything in the atlas is premultiplied by coverage, so a plain average is the right
//! filter for the mip levels and nothing from around the tree bleeds into its edge.
//!
//! The views lie on a hemi-octahedral grid whose rings are pulled towards the horizon, where
//! trees are almost always seen from: the outer ring is level, the next ones ~14°, ~39° and
//! ~71° up. The same mapping is in tree_common.wgsl (`imp_decode`, `imp_encode`).

use crate::treegen::{Mesh, TreeVertex};
use crate::util::*;
use glam::{Vec2, Vec3};
use wgpu::*;

/// Views per side of the grid.
pub const FRAMES: u32 = 8;
/// Atlas texels per view, per side.
pub const CELL: u32 = 112;
pub const ATLAS: u32 = FRAMES * CELL;
/// Mip levels, down to 7 texels per view.
pub const MIPS: u32 = 5;
/// The views are rendered at this many samples per atlas texel each way and averaged down
/// (impostor_pack.wgsl assumes 2).
const SUPERSAMPLE: u32 = 2;
/// How far the grid's rings are pulled towards the horizon (1: plain hemi-octahedral).
pub const WARP: f32 = 1.5;
pub const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
pub const DATA_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const BAKE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Grid coordinates in [-1, 1]² to a direction on the upper hemisphere.
pub fn decode(e: Vec2) -> Vec3 {
    let x = (e.x + e.y) * 0.5;
    let z = (e.x - e.y) * 0.5;
    let r = x.abs() + z.abs();
    let el = (1.0 - r).clamp(0.0, 1.0).powf(WARP) * std::f32::consts::FRAC_PI_2;
    let h = Vec2::new(x, z).try_normalize().unwrap_or(Vec2::X);
    Vec3::new(h.x * el.cos(), el.sin(), h.y * el.cos())
}

/// A direction to grid coordinates; directions below the horizon are taken as level. (Only
/// the shaders use it, as `imp_encode`; this one is for the tests.)
#[cfg_attr(not(test), allow(dead_code))]
pub fn encode(d: Vec3) -> Vec2 {
    let el = d.y.clamp(0.0, 1.0).asin();
    let r = 1.0 - (el / std::f32::consts::FRAC_PI_2).powf(1.0 / WARP);
    let l1 = d.x.abs() + d.z.abs();
    let xz = if l1 > 1e-6 { Vec2::new(d.x, d.z) / l1 * r } else { Vec2::ZERO };
    Vec2::new(xz.x + xz.y, xz.x - xz.y)
}

/// The direction view (i, j) of the grid looks at the tree from.
pub fn frame_dir(i: u32, j: u32) -> Vec3 {
    decode(Vec2::new(i as f32, j as f32) / (FRAMES - 1) as f32 * 2.0 - 1.0)
}

/// The image axes (right, up) of a view from direction `d`.
pub fn basis(d: Vec3) -> (Vec3, Vec3) {
    let r = Vec3::Y.cross(d).try_normalize().unwrap_or(Vec3::X);
    (r, d.cross(r))
}

/// Where a shape sits in its views: the height of their centre above the tree's base, the half
/// size of the square that holds the tree in every view, and the depth range (the radius of a
/// sphere around the centre that holds it).
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub center_y: f32,
    pub half: f32,
    pub depth: f32,
}

pub fn fit(mesh: &Mesh) -> Fit {
    let (lo, hi) = mesh.verts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v.pos[1]), hi.max(v.pos[1])));
    let c = Vec3::new(0.0, (lo + hi) * 0.5, 0.0);
    let qs: Vec<Vec3> = mesh.verts.iter().map(|v| Vec3::from(v.pos) - c).collect();
    let mut half: f32 = 0.0;
    for j in 0..FRAMES {
        for i in 0..FRAMES {
            let (r, u) = basis(frame_dir(i, j));
            for q in &qs {
                half = half.max(q.dot(r).abs()).max(q.dot(u).abs());
            }
        }
    }
    let depth = qs.iter().map(|q| q.length()).fold(0.0, f32::max);
    // a few texels of margin, so the coarsest mips don't reach into the next view
    Fit { center_y: c.y, half: half * 1.05 + 0.1, depth: depth + 0.1 }
}

/// The atlas: one array layer per tree shape.
pub struct Atlas {
    color: Texture,
    data: Texture,
    pub color_view: TextureView,
    pub data_view: TextureView,
}

fn layer_view(tex: &Texture, layer: u32, mip: u32) -> TextureView {
    tex.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: layer,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

impl Atlas {
    pub fn new(device: &Device, layers: u32) -> Atlas {
        let tex = |label: &str, format: TextureFormat| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: layers },
                mip_level_count: MIPS,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let color = tex("impostor colour", COLOR_FORMAT);
        let data = tex("impostor data", DATA_FORMAT);
        let array = TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() };
        let (color_view, data_view) = (color.create_view(&array), data.create_view(&array));
        let bytes = (ATLAS as u64 * ATLAS as u64 * 8 * layers as u64) * 4 / 3;
        println!("impostors: {layers} shapes, {FRAMES}x{FRAMES} views of {CELL} px, {:.0} MB", bytes as f64 / 1e6);
        Atlas { color, data, color_view, data_view }
    }

    /// Writes the atlas at half resolution as two PNGs (debugging aid): `path` the colour over
    /// grey, `path` + ".data.png" the normals (in each view's frame) and the depth, both still
    /// premultiplied by coverage. One tile per shape, five to a row.
    pub fn debug_png(&self, device: &Device, queue: &Queue, path: &str) {
        let layers = self.color.size().depth_or_array_layers;
        let w = ATLAS / 2;
        let row = (w * 4).next_multiple_of(256);
        let cols = 5;
        let rows = layers.div_ceil(cols);
        let (iw, ih) = ((w * cols) as usize, (w * rows) as usize);
        let mut out = [vec![0u8; iw * ih * 4], vec![0u8; iw * ih * 4]];
        for layer in 0..layers {
            for (k, tex) in [&self.color, &self.data].into_iter().enumerate() {
                let buf = buffer(device, "impostor rb", (row * w) as u64, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_texture_to_buffer(
                    TexelCopyTextureInfo { texture: tex, mip_level: 1, origin: Origin3d { x: 0, y: 0, z: layer }, aspect: TextureAspect::All },
                    TexelCopyBufferInfo { buffer: &buf, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(w) } },
                    Extent3d { width: w, height: w, depth_or_array_layers: 1 },
                );
                queue.submit([enc.finish()]);
                let data = read_buffer_blocking(device, &buf, (row * w) as u64);
                let (tx, ty) = ((layer % cols * w) as usize, (layer / cols * w) as usize);
                for y in 0..w as usize {
                    for x in 0..w as usize {
                        let p = &data[y * row as usize + x * 4..][..4];
                        let o = ((ty + y) * iw + tx + x) * 4;
                        let a = p[3] as f32 / 255.0;
                        for c in 0..3 {
                            // colour: premultiplied sRGB over grey (roughly)
                            out[k][o + c] = if k == 0 { (p[c] as f32 + 140.0 * (1.0 - a)) as u8 } else { p[c] };
                        }
                        out[k][o + 3] = 255;
                    }
                }
            }
        }
        crate::write_png(path, iw as u32, ih as u32, &out[0]);
        crate::write_png(&format!("{path}.data.png"), iw as u32, ih as u32, &out[1]);
    }
}

fn clear(view: &TextureView) -> RenderPassColorAttachment<'_> {
    RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: StoreOp::Store } }
}

/// One shape to bake: its LOD0 in the shared tree buffers, and the atlas layer it goes to.
pub struct BakeItem {
    pub layer: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
}

/// Renders every item's views and packs them into its atlas layer, then builds the mips.
/// `sources` is the tree shader code the bake entry points (`vs_bake`, `fs_bake`) live in;
/// it reads the shape table (`shapes`, group 1 binding 1) and the leaf textures (group 0).
#[allow(clippy::too_many_arguments)]
pub fn bake(
    device: &Device,
    queue: &Queue,
    global_bgl: &BindGroupLayout,
    global_bg: &BindGroup,
    sources: &[&str],
    shapes: &Buffer,
    vbuf: &Buffer,
    ibuf: &Buffer,
    items: &[BakeItem],
    atlas: &Atlas,
) {
    let t0 = std::time::Instant::now();
    let size = ATLAS * SUPERSAMPLE;
    let target = |label: &str, format: TextureFormat| {
        device
            .create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let bake_color = target("impostor bake colour", BAKE_FORMAT);
    let bake_data = target("impostor bake data", BAKE_FORMAT);
    let bake_depth = target("impostor bake depth", TextureFormat::Depth32Float);

    // the views of one shape: instance i is view i
    let shapes_bgl = bgl(
        device,
        "impostor bake",
        &[BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    );
    let shapes_bg = device.create_bind_group(&BindGroupDescriptor {
        label: Some("impostor bake"),
        layout: &shapes_bgl,
        entries: &[BindGroupEntry { binding: 1, resource: shapes.as_entire_binding() }],
    });
    let module = shader(device, "impostor bake", sources);
    let vattrs = [
        VertexAttribute { format: VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
        VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
        VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 3 },
    ];
    let bake_targets = [Some(ColorTargetState::from(BAKE_FORMAT)), Some(ColorTargetState::from(BAKE_FORMAT))];
    let bake_pipe = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("impostor bake"),
        layout: Some(&pipeline_layout(device, &[global_bgl, &shapes_bgl])),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_bake"),
            compilation_options: Default::default(),
            buffers: &[Some(VertexBufferLayout { array_stride: std::mem::size_of::<TreeVertex>() as u64, step_mode: VertexStepMode::Vertex, attributes: &vattrs })],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: Some(depth_state(TextureFormat::Depth32Float, true, CompareFunction::Less)),
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState { module: &module, entry_point: Some("fs_bake"), compilation_options: Default::default(), targets: &bake_targets }),
        multiview_mask: None,
        cache: None,
    });

    // packing into the atlas, and the mips
    let pack_bgl = bgl(
        device,
        "impostor pack",
        &[
            texture_entry(0, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
            texture_entry(1, ShaderStages::FRAGMENT, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2),
        ],
    );
    let pack_module = shader(device, "impostor pack", &[include_str!("shaders/impostor_pack.wgsl")]);
    let pack_layout = pipeline_layout(device, &[&pack_bgl]);
    let atlas_targets = [Some(ColorTargetState::from(COLOR_FORMAT)), Some(ColorTargetState::from(DATA_FORMAT))];
    let pack_pipe = |entry: &str| {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&pack_layout),
            vertex: VertexState { module: &pack_module, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState { module: &pack_module, entry_point: Some(entry), compilation_options: Default::default(), targets: &atlas_targets }),
            multiview_mask: None,
            cache: None,
        })
    };
    let (pipe_pack, pipe_mip) = (pack_pipe("fs_pack"), pack_pipe("fs_mip"));
    let pack_bg = bind_group(device, "impostor pack", &pack_bgl, &[BindingResource::TextureView(&bake_color), BindingResource::TextureView(&bake_data)]);

    let pass = |enc: &mut CommandEncoder, label: &str, color: &TextureView, data: &TextureView, pipe: &RenderPipeline, bg: &BindGroup| {
        let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(clear(color)), Some(clear(data))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipe);
        rp.set_bind_group(0, bg, &[]);
        rp.draw(0..3, 0..1);
    };

    let mut enc = device.create_command_encoder(&Default::default());
    for item in items {
        {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("impostor views"),
                color_attachments: &[Some(clear(&bake_color)), Some(clear(&bake_data))],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &bake_depth,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&bake_pipe);
            rp.set_bind_group(0, global_bg, &[]);
            rp.set_bind_group(1, &shapes_bg, &[]);
            rp.set_vertex_buffer(0, vbuf.slice(..));
            rp.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
            rp.draw_indexed(item.first_index..item.first_index + item.index_count, item.base_vertex, 0..FRAMES * FRAMES);
        }
        let (color, data) = (layer_view(&atlas.color, item.layer, 0), layer_view(&atlas.data, item.layer, 0));
        pass(&mut enc, "impostor pack", &color, &data, &pipe_pack, &pack_bg);
    }
    for mip in 1..MIPS {
        for item in items {
            let src = [layer_view(&atlas.color, item.layer, mip - 1), layer_view(&atlas.data, item.layer, mip - 1)];
            let bg = bind_group(device, "impostor mip", &pack_bgl, &[BindingResource::TextureView(&src[0]), BindingResource::TextureView(&src[1])]);
            let (color, data) = (layer_view(&atlas.color, item.layer, mip), layer_view(&atlas.data, item.layer, mip));
            pass(&mut enc, "impostor mip", &color, &data, &pipe_mip, &bg);
        }
    }
    queue.submit([enc.finish()]);
    device.poll(PollType::wait_indefinitely()).expect("poll failed");
    println!("impostors baked in {:.2}s", t0.elapsed().as_secs_f32());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_inverts_decode() {
        for j in 0..=20 {
            for i in 0..=20 {
                let e = Vec2::new(i as f32, j as f32) / 10.0 - 1.0;
                let d = decode(e);
                assert!((d.length() - 1.0).abs() < 1e-5 && d.y >= 0.0);
                if d.y < 0.999 {
                    assert!(encode(d).distance(e) < 1e-4, "{e} -> {d} -> {}", encode(d));
                }
            }
        }
    }

    #[test]
    fn rings_crowd_towards_the_horizon() {
        let el = |i: u32| frame_dir(i, i).y.asin().to_degrees();
        assert!(el(0).abs() < 1e-3); // the outer ring is level
        let second = el(1);
        assert!(second > 8.0 && second < 18.0, "{second}");
        // neighbouring views along the horizon are less than 15° apart
        let a = frame_dir(0, 0).angle_between(frame_dir(1, 0)).to_degrees();
        assert!(a < 15.0, "{a}");
    }

    #[test]
    fn view_axes_are_orthonormal_and_upright() {
        for j in 0..FRAMES {
            for i in 0..FRAMES {
                let d = frame_dir(i, j);
                let (r, u) = basis(d);
                assert!(r.dot(u).abs() < 1e-5 && r.dot(d).abs() < 1e-5 && u.dot(d).abs() < 1e-5);
                assert!(u.y > 0.0, "the image's up points up");
                assert!((r.cross(u) - d).length() < 1e-4, "right-handed, looking along -d");
            }
        }
    }
}
