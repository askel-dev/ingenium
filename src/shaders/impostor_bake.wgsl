// Impostor bake (impostor.rs): one tree shape's LOD0 drawn into all of its views at once,
// instance i being view i, each into its own cell of the atlas (the shape's fit keeps every
// view inside its cell). Writes the unlit colour with full coverage, and the normal (in the
// view's frame, octahedral), the depth towards the viewer and the ambient occlusion. Leaves
// are cut out at the same alpha the mesh uses.

struct BakeIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) attr: vec4f, // kind, sway, ao, shape slot
}

struct BakeOut {
    @builtin(position) clip: vec4f,
    @location(0) nrm: vec3f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) info: vec4u, // species, kind, view
    @location(3) misc: vec2f,                    // ao, depth (-1..1, positive towards the viewer)
}

@vertex
fn vs_bake(v: BakeIn, @builtin(instance_index) view: u32) -> BakeOut {
    let s = SHAPES[u32(v.attr.w + 0.5)];
    let cell = vec2u(view % IMP_FRAMES, view / IMP_FRAMES);
    let d = imp_frame_dir(cell);
    let ax = imp_axes(d);
    let q = v.pos - vec3f(0.0, s.imp.x, 0.0);
    let local = vec2f(dot(q, ax.r), -dot(q, ax.u)) / (2.0 * s.imp.y) + 0.5;
    let auv = (vec2f(cell) + local) / f32(IMP_FRAMES);
    let depth = dot(q, d) / s.imp.z;
    var o: BakeOut;
    o.clip = vec4f(auv.x * 2.0 - 1.0, 1.0 - auv.y * 2.0, 0.5 - 0.5 * depth, 1.0);
    o.nrm = v.nrm;
    o.uv = v.uv;
    o.info = vec4u(u32(s.crown_r.w), u32(v.attr.x + 0.5), view, 0u);
    o.misc = vec2f(v.attr.z, depth);
    return o;
}

struct BakeTargets {
    @location(0) color: vec4f,
    @location(1) data: vec4f,
}

@fragment
fn fs_bake(in: BakeOut) -> BakeTargets {
    let tex = textureSample(t_leaf, s_repeat, in.uv, i32(in.info.x));
    let kind = in.info.y;
    if (kind == 1u && tex.a < 0.5) {
        discard;
    }
    var albedo = tex.rgb;
    if (kind == 0u) {
        albedo = bark(in.info.x, in.uv);
    } else if (kind == 2u) {
        albedo = canopy_color(in.info.x);
    }
    let d = imp_frame_dir(vec2u(in.info.z % IMP_FRAMES, in.info.z / IMP_FRAMES));
    let ax = imp_axes(d);
    let n = normalize(in.nrm);
    let nv = vec3f(dot(n, ax.r), dot(n, ax.u), dot(n, d));
    return BakeTargets(vec4f(albedo, 1.0), vec4f(oct_encode(nv) * 0.5 + 0.5, in.misc.y * 0.5 + 0.5, in.misc.x));
}
