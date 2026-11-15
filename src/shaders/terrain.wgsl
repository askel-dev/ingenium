// CDLOD terrain: one 33x33 grid mesh instanced per quadtree patch, morphed
// towards the next-coarser grid as it approaches its LOD range so there are no cracks.

@group(1) @binding(0) var<uniform> P: Pass;

struct VOut {
    // invariant: the occluder pass lays down this depth, and the main pass must match it
    @builtin(position) @invariant clip: vec4f,
    @location(0) wp: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @location(0) quad: vec4f) -> VOut {
    var g = vec2f(f32(vid % 33u), f32(vid / 33u));
    let origin = quad.xy;
    let size = quad.z;
    let level = u32(quad.w);
    let p0 = origin + g * (size / 32.0);
    let d = distance(p0, P.ref_cam.xz);
    let m = F.morph[level];
    let k = clamp((d - m.x) / (m.y - m.x), 0.0, 1.0);
    g = g - fract(g * 0.5) * 2.0 * k;
    let p = origin + g * (size / 32.0);
    // the detail relief is gone by 140 m: skip its noise there (most of the vertices)
    let detail = 1.0 - smoothstep(80.0, 140.0, d);
    var h = height_at(p);
    if (detail > 0.0) {
        h += detail_h(p) * detail;
    }
    let wp = vec3f(p.x, h, p.y);
    var o: VOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    return o;
}

// Flower heads painted on the ground: one chance of a dot per 0.45 m cell. When a pixel
// covers more than a few cells the dots are replaced by their average cover, so a drift
// of flowers still tints the meadow from far away without sparkling.
fn flower_dots(p: vec2f, amount: f32, fp: f32, dist: f32) -> f32 {
    let cell = 0.45;
    // near the camera the flowers stand in the grass as real heads (grass_gen.wgsl, which
    // thins them out over the same distances)
    let density = amount * flower_drift(p) * 0.7 * smoothstep(GRASS_NEAR, GRASS_FAR, dist);
    let avg = density * 0.12;
    let far = smoothstep(0.04, 0.2, fp);
    if (far >= 1.0 || density < 0.003) {
        return avg;
    }
    let q = p / cell;
    let ci = vec2i(floor(q));
    let h = hash_i2(ci + vec2i(9137, 2251));
    let present = select(0.0, 1.0, u01(h) < density);
    let centre = vec2f(ci) + 0.2 + 0.6 * vec2f(u01(pcg(h)), u01(pcg(h + 1u)));
    let r = 0.07 + 0.05 * u01(pcg(h + 2u));
    let d = length(q - centre) * cell;
    let w = max(fp, 0.005);
    let cov = present * (1.0 - smoothstep(r - w, r + w, d));
    return mix(cov, avg, far);
}

// Fine relief of the sward for the sun to catch: clumps and tussocks at 1.8 m, 0.6 m,
// 0.23 m and 9 cm. Each octave fades out once a pixel covers a good part of its period. Returns
// (height in -1..1, slope x, slope z) in metres per metre.
fn sward(p: vec2f, fp: f32) -> vec3f {
    var acc = vec3f(0.0);
    let freqs = array<f32, 4>(0.55, 1.7, 4.3, 11.0);
    let amps = array<f32, 4>(0.12, 0.07, 0.03, 0.012);
    for (var i = 0; i < 4; i++) {
        let f = freqs[i];
        let w = 1.0 - smoothstep(0.2, 0.55, fp * f);
        if (w <= 0.0) {
            break;
        }
        let o = vec2f(f32(i) * 17.3, f32(i) * -5.1);
        let nd = gnoise_d(p * f + o);
        acc += vec3f(nd.x * 0.4 * w, nd.yz * amps[i] * f * w);
    }
    return acc;
}

// What each ground scan covers (m), from the Poly Haven metadata.
const TILE_LEAVES: f32 = 1.5;
const TILE_MOSS: f32 = 1.3;
const TILE_GRASS: f32 = 2.0;
const TILE_SOIL: f32 = 3.15;
const TILE_ROCK: f32 = 2.7;

// One ground scan at p, mapped from above. Up close it is hex-tiled (no repeats); once a pixel
// spans a good part of a tile the detail has nothing left to show but its tiling and shimmer,
// so it fades to the scan's average, which the coarsest mip holds: the far ground keeps the
// near ground's average colour by construction.
fn ground_mat(layer: i32, tile: f32, p: vec2f, dpx: vec2f, dpy: vec2f, fp: f32, seed: i32) -> MatSample {
    let far = smoothstep(tile * 0.08, tile * 0.3, fp);
    if (far >= 1.0) {
        return mat_mean(layer);
    }
    let m = mat_hex(layer, p / tile, dpx / tile, dpy / tile, seed);
    if (far <= 0.0) {
        return m;
    }
    return mat_mix(m, mat_mean(layer), far);
}

// The occluder pass: depth only (the colour target is there for the trees' alpha-to-coverage).
@fragment
fn fs_depth() {}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let p = in.wp.xz;
    let dat = data_at(p);
    let dist = distance(in.wp, F.cam_pos.xyz);
    let V = (F.cam_pos.xyz - in.wp) / max(dist, 1e-3);
    // metres covered by this pixel: between the area-based size and the long axis, since at
    // a grazing angle the long axis alone would blur the detail out far too early
    let dpx = dpdx(p);
    let dpy = dpdy(p);
    let ax = length(dpx);
    let ay = length(dpy);
    let fp = max(mix(sqrt(ax * ay), max(ax, ay), 0.25), 1e-4);
    // the share of the sky the ground sees, terrain and forest, for the meadow's colour
    let vis = sky_vis(p);

    // macro normal from the baked map + procedural micro detail up close; beyond 140 m the
    // detail weighs nothing, so its eight noise lookups are skipped
    let nm = normal_at(p);
    var n = nm;
    let df = 1.0 - smoothstep(60.0, 140.0, dist);
    if (df > 0.0) {
        let e = 0.25;
        let hx = detail_h(p + vec2f(e, 0.0)) - detail_h(p - vec2f(e, 0.0));
        let hz = detail_h(p + vec2f(0.0, e)) - detail_h(p - vec2f(0.0, e));
        n = normalize(n + vec3f(-hx, 0.0, -hz) / (2.0 * e) * df);
    }
    let slope = 1.0 - nm.y;

    let forest = dat.x;
    let rock = dat.y;
    let flowers = dat.z;
    let nz = fbm(p * 0.08, 3) * 0.5 + 0.5;

    // --- where each ground goes (the same masks as before the scans)
    let sw = sward(p, fp);
    // bare soil: a band on the steeper slopes before the rock, and small worn patches
    let ragged = fbm(p * 0.35, 3);
    let worn = smoothstep(0.72, 0.95, gnoise(p * 0.06 + vec2f(7.0, -3.0)) + (nz - 0.5) * 0.5 + ragged * 0.35) * 0.75;
    var soil_amt = clamp(smoothstep(0.14, 0.26, slope + (nz - 0.5) * 0.12 + ragged * 0.05) + worn, 0.0, 1.0);
    // tufts of grass break through a bare patch
    soil_amt = smoothstep(0.3, 0.8, soil_amt + (sw.x + ragged) * 0.3 * (1.0 - soil_amt)) * 0.9;
    let wood = smoothstep(0.35, 0.75, forest);
    // moss where the floor is damper
    let moss_amt = smoothstep(0.5, 0.8, nz + ragged * 0.15) * 0.75;
    let rock_amt = smoothstep(0.4, 0.7, rock + (nz - 0.5) * 0.3);
    var w = array<f32, 5>(
        wood * (1.0 - moss_amt) * (1.0 - rock_amt), // leaves
        wood * moss_amt * (1.0 - rock_amt),         // moss
        (1.0 - wood) * (1.0 - soil_amt) * (1.0 - rock_amt), // grass
        (1.0 - wood) * soil_amt * (1.0 - rock_amt), // soil
        rock_amt,
    );
    let layers = array<i32, 5>(MAT_LEAVES, MAT_MOSS, MAT_GRASS, MAT_SOIL, MAT_ROCK);
    let tiles = array<f32, 5>(TILE_LEAVES, TILE_MOSS, TILE_GRASS, TILE_SOIL, TILE_ROCK);

    // --- the scans, each under the large-scale colour changes the ground had before them
    var grass_tint = meadow_tint(p, dat, nm, vis) / max(mat_mean(MAT_GRASS).a.rgb, vec3f(1e-3));
    // seen from above, the gaps between blades are darker than the blades
    grass_tint *= 1.0 + 0.75 * sw.x;
    let tints = array<vec3f, 5>(
        vec3f(0.8 + 0.4 * nz),
        vec3f(0.8 + 0.4 * nz),
        grass_tint,
        vec3f((0.85 + 0.25 * nz) * (1.0 + 0.2 * sw.x) / 0.975),
        vec3f((0.75 + 0.35 * nz) / 0.925),
    );
    var s: array<MatSample, 5>;
    // Height blend: where two grounds meet, whichever stands higher there shows (leaves lie on
    // the moss, grass tufts poke through soil, rock juts out).
    var top = -1.0;
    for (var i = 0; i < 5; i++) {
        if (w[i] > 0.02) {
            s[i] = ground_mat(layers[i], tiles[i], p, dpx, dpy, fp, i);
            s[i].a = vec4f(s[i].a.rgb * tints[i], s[i].a.w);
            top = max(top, w[i] + s[i].a.w * 0.5);
        }
    }
    var sum = 0.0;
    for (var i = 0; i < 5; i++) {
        if (w[i] > 0.02) {
            w[i] = max(w[i] + s[i].a.w * 0.5 - (top - 0.15), 0.0);
            sum += w[i];
        } else {
            w[i] = 0.0;
        }
    }
    var mat: MatSample;
    mat.a = vec4f(0.0);
    mat.b = vec4f(0.0);
    for (var i = 0; i < 5; i++) {
        if (w[i] > 0.0) {
            let k = w[i] / sum;
            mat.a += s[i].a * k;
            mat.b += s[i].b * k;
        }
    }
    var albedo = mat.a.rgb;
    // flower heads painted on the meadow
    albedo = mix(albedo, flower_color(p), flower_dots(p, flowers, fp, dist) * (w[2] / sum));

    // grass cover decides how much the sward relief and the sward's look apply
    let grassy = w[2] / sum;
    n = normalize(n + vec3f(-sw.y, 0.0, -sw.z) * mix(0.35, 1.0, grassy));
    // the scans' normal maps (u along +x, v along +z)
    n = normalize(n + vec3f(mat.b.x, 0.0, mat.b.y));
    let rough = mat.b.z;
    let tex_ao = mat.b.w;

    let L = F.sun_dir.xyz;
    let ndl = max(dot(n, L), 0.0);
    let c = canopy_sample(in.wp, nm);
    let sh = shadow_at(in.wp, nm, canopy_sun(c));
    // beyond the shadow map the canopy volume lets the sun through the leaves; beyond that,
    // dense forest floor is darkened instead
    let canopy = mix(1.0, 0.35, smoothstep(0.45, 0.85, forest) * (1.0 - c.w) * smoothstep(F.cascade.y * 0.6, F.cascade.y, dist));

    // A grass sward is not a flat surface: looking down into it you see the dark gaps
    // between blades, at a grazing angle only lit blade sides; with the sun behind you
    // the blades hide their own shadows (the hot spot), and against the sun they glow.
    let ndv = saturate(dot(nm, V));
    let grazing = mix(0.8, 1.12, pow(1.0 - ndv, 2.0));
    let vl = dot(V, L);
    let hot = exp(-(1.0 - vl) * 18.0) * 0.35;
    let back = pow(saturate(-vl), 6.0) * 0.4 * smoothstep(0.0, 0.2, L.y);
    let sward_look = mix(1.0, grazing, grassy);
    let sun = sun_light(in.wp) * sh * canopy * micro_shadow(tex_ao, ndl);
    let direct = sun * (ndl * (1.0 + hot * grassy) + back * grassy);
    let amb = ambient_c(n, in.wp, c) * tex_ao;
    // blades are a little glossy: at a grazing angle the sward picks up a sheen of sky
    let sheen = SKY.amb_up.rgb * F.sun_color.w * canopy_sky(c, n) * 0.05 * pow(1.0 - ndv, 4.0) * grassy;
    var col = albedo * sward_look * (direct + amb) + sheen;
    // wet leaves and bare rock catch the sun (the sward's own gloss is the sheen above)
    col += sun * sun_specular(n, V, L, rough) * (1.0 - grassy);
    col = canopy_debug(apply_fog(col, in.wp), c, n);
    return vec4f(col, 1.0);
}

// ---------------------------------------------------------------------------
// Far-forest canopy shell: the terrain grid lifted by forest density and shaded
// like treetops. It stands in for individual trees beyond ~1.3 km, so a million
// distant trees cost nothing per instance (the "proxy without instance data").

struct CanopyOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) ground: f32,
}

// At most 18 * (0.85 + 0.25 * 0.75) = 18.7 m; SHELL_PAD in terrain.rs culls with that.
fn shell_height(p: vec2f) -> f32 {
    let forest = data_at(p).x;
    let crowns = gnoise(p * 0.19) * 0.5 + gnoise(p * 0.43) * 0.25;
    return 18.0 * smoothstep(0.2, 0.7, forest) * (0.85 + 0.25 * crowns);
}

@vertex
fn vs_canopy(@builtin(vertex_index) vid: u32, @location(0) quad: vec4f) -> CanopyOut {
    var g = vec2f(f32(vid % 33u), f32(vid / 33u));
    let origin = quad.xy;
    let size = quad.z;
    let level = u32(quad.w);
    let p0 = origin + g * (size / 32.0);
    let d = distance(p0, P.ref_cam.xz);
    let m = F.morph[level];
    let k = clamp((d - m.x) / (m.y - m.x), 0.0, 1.0);
    g = g - fract(g * 0.5) * 2.0 * k;
    let p = origin + g * (size / 32.0);
    let ground = height_at(p);
    let wp = vec3f(p.x, ground + shell_height(p), p.y);
    var o: CanopyOut;
    o.clip = P.view_proj * vec4f(wp, 1.0);
    o.wp = wp;
    o.ground = ground;
    return o;
}

@fragment
fn fs_canopy(in: CanopyOut) -> @location(0) vec4f {
    let p = in.wp.xz;
    let dat = data_at(p);
    let dist = distance(in.wp, F.cam_pos.xyz);
    let lift = in.wp.y - in.ground;
    // fade in where individual trees are thinning out, and only where there is forest
    let alpha = smoothstep(0.3, 0.45, dat.x) * smoothstep(F.lod.w * 0.62, F.lod.w * 0.8, dist) * smoothstep(1.0, 4.0, lift);

    // crown-scale bumps for shading
    let e = 0.6;
    let c0 = gnoise(p * 0.19) * 0.5 + gnoise(p * 0.43) * 0.25;
    let cx = gnoise((p + vec2f(e, 0.0)) * 0.19) * 0.5 + gnoise((p + vec2f(e, 0.0)) * 0.43) * 0.25;
    let cz = gnoise((p + vec2f(0.0, e)) * 0.19) * 0.5 + gnoise((p + vec2f(0.0, e)) * 0.43) * 0.25;
    let tn = normal_at(p);
    let n = normalize(tn + vec3f(-(cx - c0), 0.0, -(cz - c0)) / e * 6.0);

    let h = in.ground;
    let spruce = smoothstep(250.0, 380.0, h + fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3) * 140.0);
    var albedo = mix(vec3f(0.043, 0.08, 0.014), vec3f(0.022, 0.042, 0.02), spruce);
    let birchy = smoothstep(0.12, 0.32, fbm(p * 0.0022 + vec2f(-41.0, 17.0), 3)) * (1.0 - spruce);
    albedo = mix(albedo, vec3f(0.065, 0.10, 0.02), birchy * 0.6);
    let tint = gnoise(p * 0.07) * 0.5 + 0.5;
    albedo *= mix(0.75, 1.1, tint);
    // gaps between crowns are darker
    let gap = smoothstep(-0.1, 0.35, c0);

    let L = F.sun_dir.xyz;
    let wrap = max((dot(n, L) + 0.45) / 1.45, 0.0);
    // the canopy top sees about the sky the bare terrain would (the bake counts the canopy
    // itself as blocking, which is right only for the floor beneath it)
    let vis = mix(sky_vis(p), 1.0, 0.75);
    var col = albedo * (sun_light(in.wp) * wrap * mix(0.45, 1.0, gap) + ambient_vis(n, in.wp, vis) * mix(0.5, 1.0, gap));
    col = apply_fog(col, in.wp);
    if (F.debug.y > 1.5) {
        // the `K` views (beyond the canopy volume: the 2D bake's guess)
        var c = canopy_sample(in.wp, n);
        c.vis_fb = mix(c.vis_fb, 1.0, 0.75);
        col = canopy_debug(col, c, n);
    }
    return vec4f(col, alpha);
}
