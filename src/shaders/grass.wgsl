// The meadow (grass_gen.wgsl places it): grass blades as tapered strips of 5 triangles,
// textured with scanned blades, and cards of scanned grass tufts, flower heads and seed heads.
// Everything is lit per vertex (sun and cloud shadow, sun shadow map, sky, aerial perspective),
// so a fragment is one texture read: a meadow is many small triangles, most a few pixels big.
// The blades are opaque, which keeps the GPU's hidden surface removal: the ground under a dense
// sward isn't shaded. The cards are cut out by alpha-to-coverage.

@group(1) @binding(0) var<storage, read> blades: array<vec4f>;
@group(1) @binding(1) var<storage, read> cards: array<vec4f>;
@group(2) @binding(0) var<uniform> P: Pass;

// Layer 4 of the foliage atlas (foliage.rs): 16 blade strips across u 0..0.5; flower heads in
// 0.125 cells along v 0..0.25 (buttercup x3, daisy x3, clover x2); seed heads 0.0625 x 0.25 in
// v 0.25..0.5; grass tufts 0.25 x 0.25 below.
const MEADOW_LAYER: i32 = 4;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    // colour = texture * mul + add (lighting, tint and aerial perspective folded in)
    @location(1) mul: vec3f,
    @location(2) add: vec3f,
    // scales the coverage: tuft cards fade out when seen from above
    @location(3) fade: f32,
}

// The sun shadow map with one hardware-filtered tap per cascade (a blade is a few pixels).
fn shadow_tap(ci: i32, wp: vec3f) -> f32 {
    let lp = F.shadow_vp[ci] * vec4f(wp, 1.0);
    let uv = lp.xy * vec2f(0.5, -0.5) + 0.5;
    if (uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0 || lp.z >= 1.0 || lp.z <= 0.0) {
        return 1.0;
    }
    return textureSampleCompareLevel(t_shadow, s_shadow, uv, ci, lp.z);
}

fn shadow_fast(wp: vec3f) -> f32 {
    let d = distance(wp, F.cam_pos.xyz);
    if (d < F.cascade.x) {
        return mix(shadow_tap(0, wp), shadow_tap(1, wp), smoothstep(F.cascade.x * 0.8, F.cascade.x, d));
    }
    if (d < F.cascade.y) {
        return mix(shadow_tap(1, wp), 1.0, smoothstep(F.cascade.y * 0.75, F.cascade.y, d));
    }
    return 1.0;
}

// Light at a point of a blade (or card) with normal n, `t` of the way up from its base, for an
// albedo the texture times `tint` will give: (multiplier, addend).
fn light(wp: vec3f, n: vec3f, t: f32, tint: vec3f, gloss: f32) -> array<vec3f, 2> {
    let L = F.sun_dir.xyz;
    let V = normalize(F.cam_pos.xyz - wp);
    let sh = shadow_fast(wp + vec3f(0.0, 0.05, 0.0));
    let sun = sun_light(wp) * sh;
    // blades are thin: lit from behind they glow, more towards the tip
    let wrap = max((dot(n, L) + 0.4) / 1.4, 0.0);
    let back = pow(max(dot(-V, L), 0.0), 4.0) * 0.5 * t + max(dot(-n, L), 0.0) * 0.3;
    // down in the sward, the blades shade each other from the sky and the sun
    let ao = mix(0.5, 1.0, smoothstep(0.0, 0.8, t));
    let c = canopy_sample(wp, n);
    let lit = sun * (wrap + back * vec3f(0.95, 1.0, 0.6)) * mix(0.7, 1.0, ao) + ambient_c(n, wp, c) * ao;
    // glossy: a sun highlight and a sheen of sky at grazing angles (as the ground has it)
    let ndv = saturate(dot(n, V));
    let sheen = SKY.amb_up.rgb * F.sun_color.w * canopy_sky(c, n) * 0.08 * pow(1.0 - ndv, 3.0);
    let spec = (sun * sun_specular(n, V, L, 0.42) + sheen) * gloss * ao;
    // aerial perspective, as apply_fog
    let clip = F.view_proj * vec4f(wp, 1.0);
    let ndc = clip.xy / clip.w;
    let w = sqrt(saturate(distance(wp, F.cam_pos.xyz) / AP_DIST));
    let ap = vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, w);
    let scat = textureSampleLevel(t_ap_scat, s_clamp, ap, 0.0).rgb;
    let trans = textureSampleLevel(t_ap_trans, s_clamp, ap, 0.0).rgb;
    if (F.debug.y > 1.5) {
        return array<vec3f, 2>(vec3f(0.0), canopy_debug(vec3f(0.0), c, n));
    }
    return array<vec3f, 2>(lit * tint * trans, spec * trans + scat);
}

// The wind: a gust field moving across the meadow, and a flutter per blade.
fn bend_of(pos: vec3f, facing: vec3f, rnd: f32) -> vec3f {
    let time = F.cam_pos.w;
    let gust = gnoise(pos.xz * 0.06 - F.wind.xy * time * 0.4) * 0.5 + 0.5;
    let lean = 0.15 + rnd * 0.3;
    let flutter = sin(time * 2.7 + rnd * 20.0 + pos.x * 0.4) * 0.06 * F.wind.z;
    return facing * (lean + flutter) + vec3f(F.wind.x, 0.0, F.wind.y) * gust * F.wind.z * 0.7;
}

// A point of blade `i`, `t` of the way up, `side` across (-1..1).
fn blade_point(i: u32, t: f32, side: f32) -> vec3f {
    let b0 = blades[i * 3u];
    let b1 = blades[i * 3u + 1u];
    let facing = vec3f(cos(b0.w), 0.0, sin(b0.w));
    let right = vec3f(-facing.z, 0.0, facing.x);
    let height = b1.x;
    let bend = bend_of(b0.xyz, facing, b1.w);
    let w = b1.y * (1.0 - t * 0.85);
    let y = height * t * (1.0 - 0.3 * length(bend) * t);
    return b0.xyz + right * side * w * 0.5 + vec3f(0.0, y, 0.0) + bend * height * t * t;
}

@vertex
fn vs_blade(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
    // a strip: three pairs up the blade, then the tip
    let ts = array<f32, 7>(0.0, 0.0, 0.4, 0.4, 0.75, 0.75, 1.0);
    let t = ts[vid];
    let side = select(select(1.0, -1.0, (vid & 1u) == 0u), 0.0, vid == 6u);
    let b0 = blades[iid * 3u];
    let b2 = blades[iid * 3u + 2u];
    let p = blade_point(iid, t, side);
    let facing = vec3f(cos(b0.w), 0.0, sin(b0.w));
    let right = vec3f(-facing.z, 0.0, facing.x);
    // rounded across, facing out of the tuft and up
    let n = normalize(facing * 0.55 + right * side * 0.3 + vec3f(0.0, 0.8, 0.0));
    var o: VOut;
    o.clip = P.view_proj * vec4f(p, 1.0);
    let strip = b2.w;
    o.uv = vec2f((strip + 0.5 + side * 0.4) / 32.0, 1.0 - t);
    let l = light(p, n, t, b2.rgb, 0.6);
    o.mul = l[0];
    o.add = l[1];
    o.fade = 1.0;
    if (F.debug.x > 0.5) {
        o.mul *= 0.4;
        o.add = vec3f(0.0, 0.0, 0.03);
    }
    return o;
}

@fragment
fn fs_blade(in: VOut) -> @location(0) vec4f {
    let tex = textureSample(t_leaf, s_repeat, in.uv, MEADOW_LAYER);
    return vec4f(tex.rgb * in.mul + in.add, 1.0);
}

// The atlas rect of a card: code = cell + 16 * kind (0 tuft, 1 flower, 2 seed head).
fn card_rect(code: u32) -> vec4f {
    let kind = code / 16u;
    let c = f32(code % 16u);
    if (kind == 1u) {
        let x = 0.5 + f32(code % 4u) * 0.125;
        let y = f32((code % 16u) / 4u) * 0.125;
        return vec4f(x, y, x + 0.125, y + 0.125);
    }
    if (kind == 2u) {
        let x = 0.5 + c * 0.0625;
        return vec4f(x, 0.25, x + 0.0625, 0.5);
    }
    let x = 0.5 + f32(code % 2u) * 0.25;
    let y = 0.5 + f32((code % 4u) / 2u) * 0.25;
    return vec4f(x, y, x + 0.25, y + 0.25);
}

@vertex
fn vs_card(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
    let c0 = cards[iid * 2u];
    let c1 = cards[iid * 2u + 1u];
    let code = u32(c1.w + 0.5);
    let kind = code / 16u;
    let size = c0.w;
    let corner = vec2f(f32(vid & 1u), f32(vid >> 1u)); // x across, y up
    let rect = card_rect(code);
    var p: vec3f;
    var n: vec3f;
    var t = corner.y;
    var fade = 1.0;
    if (kind == 0u) {
        // a tuft: upright, turned to face the camera about the vertical, swaying at the top
        let base = c0.xyz;
        let to_cam = F.cam_pos.xz - base.xz;
        let f = normalize(select(vec2f(0.0, 1.0), to_cam, dot(to_cam, to_cam) > 1e-6));
        let right = vec3f(-f.y, 0.0, f.x);
        let bend = bend_of(base, vec3f(f.x, 0.0, f.y), fract(c1.x * 7.3)) * 0.5;
        p = base + right * (corner.x - 0.5) * size + vec3f(0.0, corner.y * size, 0.0) + bend * size * corner.y * corner.y;
        n = normalize(vec3f(f.x, 0.0, f.y) * 0.35 + vec3f(0.0, 1.0, 0.0));
        t = corner.y * 0.85;
        // a card is a picture of a tuft from the side: looking down on it, it would show as
        // a line, so it fades out and leaves the ground (and the blades) to show the sward
        let el = (F.cam_pos.y - base.y) / max(distance(F.cam_pos.xyz, base), 1e-3);
        fade = 1.0 - smoothstep(0.3, 0.65, el);
    } else {
        let bi = u32(c0.x + 0.5);
        let tip = blade_point(bi, 1.0, 0.0);
        let to_cam = F.cam_pos.xyz - tip;
        let fh = normalize(select(vec2f(0.0, 1.0), to_cam.xz, dot(to_cam.xz, to_cam.xz) > 1e-6));
        let right = vec3f(-fh.y, 0.0, fh.x);
        if (kind == 1u) {
            // a flower head: facing up to the sky, tipped a little towards the camera
            let fwd = normalize(vec3f(fh.x, 0.0, fh.y) * 0.6 + vec3f(0.0, 0.8, 0.0));
            p = tip + right * (corner.x - 0.5) * size + fwd * (0.5 - corner.y) * size + vec3f(0.0, size * 0.1, 0.0);
            n = normalize(vec3f(fh.x, 0.0, fh.y) * 0.3 + vec3f(0.0, 1.0, 0.0));
            t = 1.0;
        } else {
            // a seed head hanging off the top of its stalk, a quarter as wide as it is tall
            let below = blade_point(bi, 0.85, 0.0);
            let up = normalize(tip - below);
            p = tip - up * size * 0.25 + right * (corner.x - 0.5) * size * 0.25 + up * corner.y * size;
            n = normalize(vec3f(fh.x, 0.0, fh.y) * 0.5 + vec3f(0.0, 0.8, 0.0));
            t = 1.0;
        }
    }
    var o: VOut;
    o.clip = P.view_proj * vec4f(p, 1.0);
    o.uv = vec2f(mix(rect.x, rect.z, corner.x), mix(rect.w, rect.y, corner.y));
    let l = light(p, n, t, c1.rgb, select(0.5, 0.25, kind == 0u));
    o.mul = l[0];
    o.add = l[1];
    o.fade = fade;
    if (F.debug.x > 0.5) {
        o.mul *= 0.4;
        o.add = vec3f(0.0, 0.0, 0.03);
    }
    return o;
}

fn card_tex(in: VOut) -> vec4f {
    let tex = textureSample(t_leaf, s_repeat, in.uv, MEADOW_LAYER);
    let aw = max(fwidth(tex.a), 1e-4);
    return vec4f(tex.rgb * in.mul + in.add, clamp((tex.a * in.fade - 0.5) / aw + 0.5, 0.0, 1.0));
}

// With MSAA: alpha-to-coverage.
@fragment
fn fs_card(in: VOut) -> @location(0) vec4f {
    return card_tex(in);
}

// Without MSAA: an alpha test.
@fragment
fn fs_card_test(in: VOut) -> @location(0) vec4f {
    let c = card_tex(in);
    if (c.a < 0.5) {
        discard;
    }
    return vec4f(c.rgb, 1.0);
}
