// Screen-space ambient occlusion: the contact shadows the baked sky visibility is too coarse
// for (where a trunk meets the ground, under a low branch, between grass clumps and rocks).
//
// The renderer is forward, so the lighting is already in the HDR buffer and there is no
// separate ambient term to darken. Instead this pass works out, from the depth buffer, how
// much of each pixel's light was ambient: it re-evaluates the sun shadow, the cloud shadow
// and n.L at the reconstructed position, and only that share is occluded. A sunlit patch of
// ground next to a trunk keeps its sunlight; the shaded side of the trunk goes darker.
//
// fs_prep takes a half-resolution copy of the depth buffer; fs_ao runs on it with a 4x4
// rotating pattern of 10 hemisphere samples;
// fs_blur takes a depth-aware 4x4 average that cancels the pattern. The post pass
// multiplies the HDR colour by the result.

// (t_depth is declared by the Rust side: multisampled or not, to match the depth buffer.)
@group(1) @binding(1) var t_ao: texture_2d<f32>;
@group(1) @binding(2) var t_hd: texture_2d<f32>; // half-resolution copy of the depth buffer

const SAMPLES: u32 = 10u;
const RADIUS: f32 = 0.9;      // metres, near the camera
const FADE_START: f32 = 50.0; // AO fades out between these distances (m)
const FADE_END: f32 = 110.0;

struct VOut {
    @builtin(position) clip: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    var o: VOut;
    o.clip = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// One sample of each 2x2 block of the (multisampled) depth buffer, so the AO pass reads a
// small, cache-friendly texture instead of scattering over the full one.
@fragment
fn fs_prep(in: VOut) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy) * 2;
    return vec4f(textureLoad(t_depth, min(px, vec2i(F.screen.xy) - 1), 0));
}

fn half_dims() -> vec2i {
    return vec2i(textureDimensions(t_hd));
}

// position relative to the camera (world axes) of the surface at half-res pixel hp, which
// stands for the full-res pixel 2 * hp
fn rel_pos(hp: vec2i, d: f32) -> vec3f {
    let uv = (vec2f(hp * 2) + 0.5) * F.screen.zw;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = F.inv_view_proj_rot * vec4f(ndc, max(d, 1e-9), 1.0);
    return h.xyz / h.w;
}

fn hd_at(hp: vec2i) -> f32 {
    return textureLoad(t_hd, clamp(hp, vec2i(0), half_dims() - 1), 0).r;
}

fn rel_at(hp: vec2i) -> vec3f {
    return rel_pos(hp, hd_at(hp));
}

// pick the neighbour on the side with the smaller depth step, so the normal of a surface
// next to a silhouette doesn't bend towards the background
fn closest_delta(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let da = a - p;
    let db = p - b;
    return select(db, da, dot(da, da) < dot(db, db));
}

@fragment
fn fs_ao(in: VOut) -> @location(0) vec4f {
    let hp = vec2i(in.clip.xy);
    let d = hd_at(hp);
    if (d <= 0.0) {
        return vec4f(1.0, 1e6, 0.0, 0.0);
    }
    let P = rel_pos(hp, d);
    let dist = length(P);
    if (dist > FADE_END) {
        return vec4f(1.0, dist, 0.0, 0.0);
    }
    let dx = closest_delta(P, rel_at(hp + vec2i(1, 0)), rel_at(hp - vec2i(1, 0)));
    let dy = closest_delta(P, rel_at(hp + vec2i(0, 1)), rel_at(hp - vec2i(0, 1)));
    var n = normalize(cross(dy, dx));
    if (dot(n, P) > 0.0) {
        n = -n;
    }

    // hemisphere around n, rotated per pixel in a 4x4 pattern
    let k4 = vec2u(hp) & vec2u(3u);
    let rot = (f32(k4.x * 4u + k4.y) * 0.618034 + 0.31) % 1.0;
    let rot2 = f32((k4.x * 3u + k4.y * 5u) % 16u) / 16.0;
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    let radius = RADIUS * (1.0 + dist * 0.01);
    let bias = 0.02 + dist * 0.0015;
    let cam = F.cam_pos.xyz;
    let hdims = vec2f(half_dims());
    var occ = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let u = (f32(i) + 0.5) / f32(SAMPLES);
        let sin_t = sqrt(u);
        let cos_t = sqrt(1.0 - u);
        let phi = f32(i) * 2.399963 + rot * TAU;
        let dir = (t * cos(phi) + b * sin(phi)) * sin_t + n * cos_t;
        var sc = (f32(i) * 0.618034 + rot2) % 1.0;
        sc = mix(0.12, 1.0, sc * sc);
        let s = P + dir * radius * sc;
        let clip = F.view_proj * vec4f(cam + s, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let ndc = clip.xy / clip.w;
        let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(suv < vec2f(0.0)) || any(suv >= vec2f(1.0))) {
            continue;
        }
        let shp = vec2i(suv * hdims);
        let scene = length(rel_at(shp));
        let sdist = length(s);
        // range check: something far in front of P doesn't occlude it
        let range = smoothstep(0.0, 1.0, radius / max(abs(dist - scene), 1e-3));
        occ += select(0.0, 1.0, scene < sdist - bias) * range;
    }
    let ao = 1.0 - occ / f32(SAMPLES);

    // The share of this pixel's light that was ambient (albedo cancels out). Rougher than
    // the lighting itself (one shadow tap, no per-material wrap), which is plenty for this.
    let wp = cam + P;
    let L = F.sun_dir.xyz;
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    // foliage and grass wrap the light round, so meet them halfway
    let ndl = saturate((dot(n, L) + 0.25) / 1.25);
    let lp = F.shadow_vp[1] * vec4f(wp + n * F.cascade.w * 1.5, 1.0);
    let suv = lp.xy * vec2f(0.5, -0.5) + 0.5;
    let sh = textureSampleCompareLevel(t_shadow, s_shadow, suv, 1, lp.z);
    let cs = cloud_shadow(wp);
    let direct = dot(SKY.sun.rgb, lum) * cs.x * ndl * sh;
    let vis = sky_vis(wp.xz);
    let amb = dot(SKY.amb_up.rgb, lum) * F.sun_color.w * vis + dot(SKY.sun.rgb, lum) * max(L.y, 0.0) * cs.y * vis;
    let amb_frac = amb / max(amb + direct, 1e-5);

    let fade = 1.0 - smoothstep(FADE_START, FADE_END, dist);
    let k = (1.0 - ao) * fade * mix(0.15, 1.0, amb_frac);
    return vec4f(1.0 - saturate(k), dist, 0.0, 0.0);
}

// depth-aware 4x4 average over the rotation pattern
@fragment
fn fs_blur(in: VOut) -> @location(0) vec4f {
    let hp = vec2i(in.clip.xy);
    let dims = vec2i(textureDimensions(t_ao));
    let c = textureLoad(t_ao, hp, 0);
    let d0 = c.y;
    if (d0 > FADE_END) {
        return vec4f(1.0);
    }
    var sum = 0.0;
    var wsum = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            let s = textureLoad(t_ao, clamp(hp + vec2i(x, y), vec2i(0), dims - 1), 0);
            let w = 1.0 / (1.0 + abs(s.y - d0) / (0.03 * d0 + 0.05) * 4.0);
            sum += s.x * w;
            wsum += w;
        }
    }
    return vec4f(sum / max(wsum, 1e-5));
}
