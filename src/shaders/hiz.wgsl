// The depth pyramid for occlusion culling (see hiz.rs). Depth is reversed (1 near, 0 far), so
// the farthest depth under a texel is the smallest value.
//
// (t_depth and SAMPLES are declared by the Rust side: multisampled or not, to match the depth
// buffer.)

// The texture is a power of two big enough for every level, so the levels' own sizes (rounded
// up halves of the depth buffer) are smaller than its mips: nothing reads beyond them.
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var<uniform> sizes: vec4u;   // the level below, then this level

// Level 0 from the depth buffer: each texel takes every sample of the 2x2 pixels under it.
@compute @workgroup_size(8, 8)
fn first(@builtin(global_invocation_id) id: vec3u) {
    let depth_size = textureDimensions(t_depth);
    if (any(id.xy >= (depth_size + 1u) / 2u)) {
        return;
    }
    let last = depth_size - 1u;
    let p = id.xy * 2u;
    var far = 1.0;
    for (var s = 0u; s < SAMPLES; s++) {
        far = min(far, textureLoad(t_depth, min(p, last), s));
        far = min(far, textureLoad(t_depth, min(p + vec2u(1u, 0u), last), s));
        far = min(far, textureLoad(t_depth, min(p + vec2u(0u, 1u), last), s));
        far = min(far, textureLoad(t_depth, min(p + vec2u(1u, 1u), last), s));
    }
    textureStore(dst, id.xy, vec4f(far));
}

// Every other level: the farthest of the 2x2 texels below (at an odd edge, of the ones there).
@compute @workgroup_size(8, 8)
fn reduce(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= sizes.zw)) {
        return;
    }
    let last = sizes.xy - 1u;
    let p = id.xy * 2u;
    let a = textureLoad(src, min(p, last), 0).r;
    let b = textureLoad(src, min(p + vec2u(1u, 0u), last), 0).r;
    let c = textureLoad(src, min(p + vec2u(0u, 1u), last), 0).r;
    let d = textureLoad(src, min(p + vec2u(1u, 1u), last), 0).r;
    textureStore(dst, id.xy, vec4f(min(min(a, b), min(c, d))));
}
