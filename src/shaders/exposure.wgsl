// Metering for the auto exposure: the log-average luminance of a 48x48 grid of the HDR
// frame, eased towards over a fraction of a second, like a camera's.

struct Params {
    time: f32,
    tau: f32, // seconds to cover ~63% of a change
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> E: Params;
// x = adapted log luminance, y = time of the last update, z = 1 once set
@group(0) @binding(2) var<storage, read_write> state: array<f32, 4>;

var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3u) {
    let size = vec2f(textureDimensions(t_hdr));
    var s = 0.0;
    for (var j = 0u; j < 3u; j++) {
        for (var i = 0u; i < 3u; i++) {
            let uv = (vec2f(lid.xy * 3u + vec2u(i, j)) + 0.5) / 48.0;
            let c = textureLoad(t_hdr, vec2i(uv * size), 0).rgb;
            s += log(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-4));
        }
    }
    sums[li] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (li < k) {
            sums[li] += sums[li + k];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let now = sums[0] / 2304.0;
        let dt = clamp(E.time - state[1], 0.0, 0.5);
        var v = now;
        if (state[2] > 0.5) {
            v = mix(state[0], now, 1.0 - exp(-dt / E.tau));
        }
        state[0] = v;
        state[1] = E.time;
        state[2] = 1.0;
    }
}
