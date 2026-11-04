// Screen-space indirect light with a visibility bitmask (SSILVB; Therrien, Levesque and Gilet
// 2023), and screen-space contact shadows from the sun.
//
// The renderer is forward, so the lighting is already in the HDR buffer. Alongside it the main
// pass wrote each pixel's diffuse albedo and the share of its light that came straight from the
// sun (SceneOut in globals.wgsl). From the depth buffer, at half resolution, this pass works out
//  - how much of each pixel's hemisphere the surfaces around it hide, whose share of the ambient
//    light is taken away, and the light those surfaces send it instead, which is added (times the
//    pixel's albedo): sunlit ground lights the trunks and the undersides of the leaves next to
//    it, a sunlit crown the shaded leaves around it, and the forest floor goes dark between the
//    roots and under the ferns;
//  - whether anything on screen stands between it and the sun within a metre or so, which the
//    shadow maps are too coarse to see: grass blades on each other and on the ground, leaves on
//    leaves, the undergrowth and the trunks on the forest floor. That takes the direct share away.
//
// The occlusion is GTAO-like: slices through the view vector, marched both ways in screen space.
// But instead of keeping the highest horizon, each sample marks the sectors of the slice it
// covers in a 32-bit mask, as a slab from its front face to `thickness` behind it. So light gets
// past thin things (grass, leaves, twigs) from behind them, and a sample adds its light only
// through the sectors no nearer sample already covers. The sectors are spaced by the sine of the
// angle from the normal, so each carries the same cosine-weighted share of the light.
//
// fs_prep takes a half-resolution copy of the depth buffer with normals; fs_gi runs on it with a
// 4x4 rotating pattern; fs_blur takes a depth-aware 4x4 average that cancels the pattern. The
// post pass composites the result.

// (t_depth, binding 0, is declared by the Rust side: multisampled or not, to match the depth buffer.)
@group(1) @binding(1) var t_prep: texture_2d<f32>;   // half res: xyz normal (world axes), w depth
@group(1) @binding(2) var t_hdr: texture_2d<f32>;    // the lit scene
@group(1) @binding(3) var t_raw: texture_2d<f32>;    // fs_gi: indirect light (rgb), visibility (a)
@group(1) @binding(4) var t_raw_cs: texture_2d<f32>; // fs_gi: contact shadow

const SLICES: u32 = 2u;
const STEPS: u32 = 8u;        // samples per side of a slice
const RADIUS: f32 = 2.5;      // metres the samples reach near the camera (further out, wider)
const MAX_PX: f32 = 160.0;    // and at most this many half-res pixels
const THICKNESS: f32 = 0.35;  // metres assumed behind each sample's front face
const FADE_START: f32 = 60.0; // the indirect light and occlusion fade out between these (m)
const FADE_END: f32 = 120.0;
const CONTACT_STEPS: u32 = 12u;
const CONTACT_LEN: f32 = 0.7;    // metres marched towards the sun near the camera
const CONTACT_THICK: f32 = 0.3;  // how far behind a surface the ray still counts as blocked
const CONTACT_FADE: f32 = 70.0;

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

// Position relative to the camera (world axes) of the surface at depth d behind the full-res
// pixel position px.
fn rel_full(px: vec2f, d: f32) -> vec3f {
    let uv = px * F.screen.zw;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = F.inv_view_proj_rot * vec4f(ndc, max(d, 1e-9), 1.0);
    return h.xyz / h.w;
}

fn depth_rel(px: vec2i) -> vec3f {
    let q = clamp(px, vec2i(0), vec2i(F.screen.xy) - 1);
    return rel_full(vec2f(q) + 0.5, textureLoad(t_depth, q, 0));
}

// pick the neighbour on the side with the smaller depth step, so the normal of a surface next
// to a silhouette doesn't bend towards the background
fn closest_delta(p: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let da = a - p;
    let db = p - b;
    return select(db, da, dot(da, da) < dot(db, db));
}

// One sample of each 2x2 block of the (multisampled) depth buffer, and the normal there from
// the full-resolution depth around it.
@fragment
fn fs_prep(in: VOut) -> @location(0) vec4f {
    let px = vec2i(in.clip.xy) * 2;
    let q = min(px, vec2i(F.screen.xy) - 1);
    let d = textureLoad(t_depth, q, 0);
    if (d <= 0.0) {
        return vec4f(0.0, 1.0, 0.0, 0.0);
    }
    let P = rel_full(vec2f(q) + 0.5, d);
    let dx = closest_delta(P, depth_rel(q + vec2i(1, 0)), depth_rel(q - vec2i(1, 0)));
    let dy = closest_delta(P, depth_rel(q + vec2i(0, 1)), depth_rel(q - vec2i(0, 1)));
    var n = normalize(cross(dy, dx));
    if (dot(n, P) > 0.0) {
        n = -n;
    }
    return vec4f(n, d);
}

fn half_dims() -> vec2i {
    return vec2i(textureDimensions(t_prep));
}

fn prep_at(hp: vec2i) -> vec4f {
    return textureLoad(t_prep, clamp(hp, vec2i(0), half_dims() - 1), 0);
}

// the surface at half-res pixel hp, which stands for the full-res pixel 2 * hp
fn rel_half(hp: vec2i, d: f32) -> vec3f {
    return rel_full(vec2f(hp * 2) + 0.5, d);
}

// A slice angle from the normal (-pi/2 .. pi/2 is the hemisphere) as a position 0..1 across its
// sectors: by the sine, so each sector carries the same cosine-weighted share.
fn sector_pos(a: f32) -> f32 {
    return sin(clamp(a, -0.5 * PI, 0.5 * PI)) * 0.5 + 0.5;
}

// The sectors from lo to hi (0..1) of the 32.
fn sector_bits(lo: f32, hi: f32) -> u32 {
    let a = u32(clamp(floor(lo * 32.0), 0.0, 32.0));
    let b = u32(clamp(ceil(hi * 32.0), 0.0, 32.0));
    if (b <= a) {
        return 0u;
    }
    let n = b - a;
    return select((1u << n) - 1u, 0xffffffffu, n >= 32u) << a;
}

// 4x4 ordered dither, 0..1
fn bayer(i: u32) -> f32 {
    var b = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return (b[i & 15u] + 0.5) / 16.0;
}

struct GiOut {
    @location(0) gi: vec4f,      // indirect light (rgb, to multiply by the albedo), visibility (a)
    @location(1) contact: vec4f, // share of the direct sunlight left after contact shadows
}

@fragment
fn fs_gi(in: VOut) -> GiOut {
    let hp = vec2i(in.clip.xy);
    var o: GiOut;
    o.gi = vec4f(0.0, 0.0, 0.0, 1.0);
    o.contact = vec4f(1.0);
    let c = prep_at(hp);
    if (c.w <= 0.0) {
        return o;
    }
    let P = rel_half(hp, c.w);
    let dist = length(P);
    if (dist > FADE_END) {
        return o;
    }
    let N = normalize(c.xyz);
    let V = -P / dist;
    let hdims = vec2f(half_dims());
    // metres across a half-res pixel at this depth
    let px_m = length(rel_full(vec2f(hp * 2) + 2.5, c.w) - rel_full(vec2f(hp * 2) + vec2f(0.5, 2.5), c.w));

    // the 4x4 pattern: slice rotation and step jitter
    let k4 = vec2u(hp) & vec2u(3u);
    let rot = (f32(k4.x * 4u + k4.y) * 0.618034 + 0.31) % 1.0;
    let jit = bayer(k4.y * 4u + k4.x);

    let radius = RADIUS * (1.0 + dist * 0.02);
    let thick = THICKNESS * (1.0 + dist * 0.01);
    let r_px = clamp(radius / max(px_m, 1e-5), 2.0, MAX_PX);
    var vis = 0.0;
    var light = vec3f(0.0);
    for (var sl = 0u; sl < SLICES; sl++) {
        let phi = (f32(sl) + rot) * PI / f32(SLICES);
        let w = vec2f(cos(phi), sin(phi)); // in half-res pixels, y down
        // the slice: the plane through the camera, this pixel and the next along w
        let dw = rel_full(vec2f(hp * 2) + 0.5 + w * 2.0, c.w) - P;
        let ov = normalize(dw - V * dot(dw, V));
        let axis = cross(V, ov);
        let np = N - axis * dot(N, axis);
        let n_ang = atan2(dot(np, ov), dot(np, V));
        var occ = 0u;
        var sl_light = vec3f(0.0);
        for (var side = 0u; side < 2u; side++) {
            let dir_px = select(-w, w, side == 0u);
            for (var k = 0u; k < STEPS; k++) {
                let t = (f32(k) + jit) / f32(STEPS);
                let sp = vec2f(hp) + 0.5 + dir_px * (1.0 + (r_px - 1.0) * t * t);
                if (any(sp < vec2f(0.0)) || any(sp >= hdims)) {
                    break;
                }
                let si = vec2i(sp);
                let sc = textureLoad(t_prep, si, 0);
                if (sc.w <= 0.0) {
                    continue; // the sky: its light is in the ambient term already
                }
                let S = rel_half(si, sc.w);
                let df = S - P;
                let len = length(df);
                if (len < 1e-4 || len > radius * 1.5) {
                    continue;
                }
                let db = df + normalize(S) * thick;
                let uf = sector_pos(atan2(dot(df, ov), dot(df, V)) - n_ang);
                let ub = sector_pos(atan2(dot(db, ov), dot(db, V)) - n_ang);
                let bits = sector_bits(min(uf, ub), max(uf, ub));
                let fresh = bits & ~occ;
                if (fresh != 0u) {
                    // the light the sample's surface sends back towards the camera, taken as what
                    // it sends this way (diffuse), if it faces this way at all
                    let facing = saturate(dot(normalize(sc.xyz), -df / len));
                    let radiance = textureSampleLevel(t_hdr, s_clamp, (vec2f(si * 2) + 1.0) * F.screen.zw, 0.0).rgb;
                    sl_light += radiance * (f32(countOneBits(fresh)) / 32.0 * facing);
                }
                occ |= bits;
            }
        }
        vis += 1.0 - f32(countOneBits(occ)) / 32.0;
        light += sl_light;
    }
    let fade = 1.0 - smoothstep(FADE_START, FADE_END, dist);
    o.gi = vec4f(light / f32(SLICES) * fade, mix(1.0, vis / f32(SLICES), fade));

    // contact shadow: march towards the sun and look for something on screen in front of the ray.
    // Only where the surface faces the sun: a leaf lit from behind gets its light through itself.
    let L = F.sun_dir.xyz;
    if (dist < CONTACT_FADE && L.y > 0.0 && dot(N, L) > 0.0) {
        let len = CONTACT_LEN * (1.0 + dist * 0.03);
        let bias = px_m * 1.5 + 0.01;
        let ct = CONTACT_THICK * (1.0 + dist * 0.02);
        let start = P + N * px_m;
        let jit2 = bayer(k4.x * 4u + k4.y + 5u);
        var shade = 0.0;
        for (var i = 0u; i < CONTACT_STEPS; i++) {
            let t = (f32(i) + jit2) / f32(CONTACT_STEPS);
            let X = start + L * (len * t);
            let clip = F.view_proj * vec4f(F.cam_pos.xyz + X, 1.0);
            if (clip.w <= 0.0) {
                break;
            }
            let ndc = clip.xy / clip.w;
            let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (any(suv < vec2f(0.0)) || any(suv >= vec2f(1.0))) {
                break;
            }
            let si = vec2i(suv * hdims);
            let sd = textureLoad(t_prep, si, 0).w;
            if (sd <= 0.0) {
                continue;
            }
            // how far behind the surface seen there the ray point lies
            let behind = length(X) - length(rel_half(si, sd));
            if (behind > bias && behind < ct) {
                // softer the further along the ray the blocker is
                shade = max(shade, 1.0 - t * 0.6);
            }
        }
        o.contact = vec4f(1.0 - shade * (1.0 - smoothstep(CONTACT_FADE * 0.6, CONTACT_FADE, dist)));
    }
    return o;
}

// Depth-aware 4x4 average over the rotation pattern. Weighted by how close the depths are,
// relative to the depth (reversed-Z: view depth is proportional to 1 / d).
@fragment
fn fs_blur(in: VOut) -> GiOut {
    let hp = vec2i(in.clip.xy);
    let dims = half_dims();
    var o: GiOut;
    o.gi = vec4f(0.0, 0.0, 0.0, 1.0);
    o.contact = vec4f(1.0);
    let d0 = prep_at(hp).w;
    if (d0 <= 0.0) {
        return o;
    }
    var gi = vec4f(0.0);
    var cs = 0.0;
    var wsum = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            let q = clamp(hp + vec2i(x, y), vec2i(0), dims - 1);
            let dq = textureLoad(t_prep, q, 0).w;
            let rel = abs(d0 / max(dq, 1e-12) - 1.0);
            let w = 1.0 / (1.0 + rel * 130.0);
            gi += textureLoad(t_raw, q, 0) * w;
            cs += textureLoad(t_raw_cs, q, 0).r * w;
            wsum += w;
        }
    }
    o.gi = gi / max(wsum, 1e-5);
    o.contact = vec4f(cs / max(wsum, 1e-5));
    return o;
}
