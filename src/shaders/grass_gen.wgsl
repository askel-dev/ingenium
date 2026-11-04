// Per-frame grass generation around the camera. The grid is anchored to world cells
// so blades never swim as the camera moves; density falls off with distance.

struct GrassParams {
    planes: array<vec4f, 6>,
    origin: vec4f,   // x,z grid origin (world), cell size, radius
    dims: vec4u,     // grid dim, capacity
}

@group(1) @binding(0) var<uniform> G: GrassParams;
@group(1) @binding(1) var<storage, read_write> blades: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> args: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dim = G.dims.x;
    if (id.x >= dim || id.y >= dim) {
        return;
    }
    let cs = G.origin.z;
    let radius = G.origin.w;
    let cell = vec2i(floor(G.origin.xy / cs)) + vec2i(id.xy);
    for (var k = 0u; k < 2u; k++) {
        let h0 = pcg(hash_i2(cell) + k * 0x68e31da4u);
        let h1 = pcg(h0);
        let h2 = pcg(h1);
        let h3 = pcg(h2);
        let p = (vec2f(cell) + vec2f(u01(h0), u01(h1))) * cs;
        let d2 = distance(p, F.cam_pos.xz);
        if (d2 > radius) {
            continue;
        }
        let keep = 1.0 - smoothstep(radius * 0.2, radius, d2);
        if (u01(h2) > keep) {
            continue;
        }
        let dat = data_at(p);
        let amount = (1.0 - smoothstep(0.35, 0.75, dat.x)) * (1.0 - smoothstep(0.3, 0.6, dat.y));
        if (u01(h3) > amount) {
            continue;
        }
        let y = ground_h(p);
        let wp = vec3f(p.x, y, p.y);
        let d3 = distance(wp, F.cam_pos.xyz);
        if (d3 > radius) {
            continue;
        }
        var vis = true;
        for (var i = 0; i < 6; i++) {
            let pl = G.planes[i];
            if (dot(pl.xyz, wp + vec3f(0.0, 0.4, 0.0)) + pl.w < -0.8) {
                vis = false;
            }
        }
        if (!vis) {
            continue;
        }
        let h4 = pcg(h3);
        let h5 = pcg(h4);
        let h6 = pcg(h5);
        let tall = smoothstep(-0.25, 0.5, fbm(p * 0.06, 3));
        let r4 = u01(h4);
        var height = mix(0.18, 0.55, r4 * r4) * mix(0.5, 1.7, tall) * (1.0 + d3 / radius * 0.4);
        if (r4 > 0.93) {
            height *= 1.6;
        }
        let width = 0.032 * (1.0 + d3 / radius * 3.0) * mix(0.8, 1.2, u01(h0));
        let angle = u01(h5) * TAU;
        var flower = 0.0;
        if (u01(h6) < dat.z * flower_drift(p) * 0.22) {
            flower = 1.0 + f32(h6 & 1u);
        }
        let slot = atomicAdd(&args[1], 1u);
        if (slot >= G.dims.y) {
            atomicSub(&args[1], 1u);
            return;
        }
        blades[slot * 2u] = vec4f(wp, angle);
        blades[slot * 2u + 1u] = vec4f(height, width, flower, u01(h3));
    }
}
