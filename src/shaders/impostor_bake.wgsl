// The impostor bake (impostor.rs): a tree variant's LOD0, unlit, from every frame's direction
// into its cell of the atlas, one instance per frame, orthographic, centred on the frame
// centre. Out: albedo + whether leaf (a: 0 nothing, 0.5 bark, 1 leaf), and the surface normal
// in tree space + ambient occlusion.

struct Bake {
    center_y: f32,
    half: f32,
    species: u32,
    pad: u32,
}

@group(1) @binding(0) var<uniform> BK: Bake;

struct BIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) attr: vec4f,   // kind, sway, ao, girth (bark) or foliage layer (leaves)
}

struct BOut {
    @builtin(position) clip: vec4f,
    @location(0) lp: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) info: vec2u,   // kind, foliage layer
    @location(4) ao: f32,
    @location(5) bark: vec3f,
    @location(6) girth: f32,
}

@vertex
fn vs_bake(v: BIn, @builtin(instance_index) frame: u32) -> BOut {
    let g = vec2f(f32(frame % IMP_GRID), f32(frame / IMP_GRID));
    let d = frame_dir(g);
    let r = frame_right(d);
    let u = cross(d, r);
    let q = v.pos - vec3f(0.0, BK.center_y, 0.0);
    let x = dot(q, r) / BK.half;
    let y = dot(q, u) / BK.half;
    let z = dot(q, d) / BK.half; // towards the camera
    let a = (g + vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5)) / f32(IMP_GRID);
    var o: BOut;
    o.clip = vec4f(a.x * 2.0 - 1.0, 1.0 - a.y * 2.0, z * 0.5 + 0.5, 1.0);
    o.lp = v.pos;
    o.nrm = v.nrm;
    o.uv = v.uv;
    let kind = u32(v.attr.x + 0.5);
    o.info = vec2u(kind, select(0u, u32(v.attr.w + 0.5), kind == 1u));
    o.ao = v.attr.z;
    o.bark = bark_uv(BK.species, v.uv, v.attr.w, 0u);
    o.girth = v.attr.w;
    return o;
}

struct BakeOut {
    @location(0) a: vec4f,
    @location(1) b: vec4f,
}

@fragment
fn fs_bake(in: BOut) -> BakeOut {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.y));
    let texn = textureSample(t_leaf_n, s_repeat, in.uv, i32(in.info.y));
    let dwx = dpdx(in.lp);
    let dwy = dpdy(in.lp);
    let dbx = dpdx(in.bark.xy);
    let dby = dpdy(in.bark.xy);
    let dux = dpdx(in.uv);
    let duy = dpdy(in.uv);
    let n = normalize(in.nrm);
    var out: BakeOut;
    if (in.info.x == 1u) {
        if (tex.a < 0.5) {
            discard;
        }
        let ns = perturb(n, texn.xy * 2.0 - 1.0, dwx, dwy, dux, duy);
        out.a = vec4f(tex.rgb, 1.0);
        out.b = vec4f(ns * 0.5 + 0.5, in.ao);
    } else if (in.info.x == 0u) {
        let m = mat_fetch(i32(min(BK.species, 2u)), in.bark.xy, dbx, dby);
        var albedo = m.a.rgb;
        if (BK.species == 1u) {
            albedo = birch_bark(albedo, in.bark, max(length(dbx), length(dby)), in.girth).rgb;
        }
        let ns = perturb(n, m.b.xy, dwx, dwy, dbx, dby);
        out.a = vec4f(albedo, 0.5);
        out.b = vec4f(ns * 0.5 + 0.5, in.ao * m.b.w);
    } else {
        out.a = vec4f(canopy_color(BK.species), 1.0);
        out.b = vec4f(n * 0.5 + 0.5, in.ao);
    }
    return out;
}
