// Impostor atlas building (impostor.rs). `fs_pack` turns the views rendered for one tree
// shape (2 × 2 samples per atlas texel) into its atlas layer; `fs_mip` averages a level down
// into the next. Everything is premultiplied by coverage (the colour texture's alpha), so a
// plain average is the right filter and the empty space around a tree adds nothing to its
// edge. The colour target is sRGB: it is stored encoded and read back linear.

@group(0) @binding(0) var src_color: texture_2d<f32>;
@group(0) @binding(1) var src_data: texture_2d<f32>;

struct Out {
    @location(0) color: vec4f,
    @location(1) data: vec4f,
}

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
}

// The rendered views: colour (rgb, 1 where covered), data (normal, depth, ao).
@fragment
fn fs_pack(@builtin(position) pos: vec4f) -> Out {
    let base = vec2i(pos.xy) * 2;
    var c = vec4f(0.0);
    var d = vec4f(0.0);
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let p = base + vec2i(x, y);
            let a = textureLoad(src_color, p, 0);
            c += vec4f(a.rgb * a.a, a.a);
            d += textureLoad(src_data, p, 0) * a.a;
        }
    }
    return Out(c * 0.25, d * 0.25);
}

@fragment
fn fs_mip(@builtin(position) pos: vec4f) -> Out {
    let base = vec2i(pos.xy) * 2;
    var c = vec4f(0.0);
    var d = vec4f(0.0);
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            c += textureLoad(src_color, base + vec2i(x, y), 0);
            d += textureLoad(src_data, base + vec2i(x, y), 0);
        }
    }
    return Out(c * 0.25, d * 0.25);
}
