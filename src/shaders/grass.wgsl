// Grass blades: 5 triangles each, built from vertex_index, bent by a moving wind field.

@group(1) @binding(0) var<storage, read> blades: array<vec4f>;
@group(2) @binding(0) var<uniform> P: Pass;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) wp: vec3f,
    @location(1) nrm: vec3f,
    @location(2) misc: vec4f,   // t along blade, flower, random, 0
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VOut {
    var sides = array<f32, 15>(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0);
    var ts = array<f32, 15>(0.0, 0.0, 0.4, 0.0, 0.4, 0.4, 0.4, 0.4, 0.75, 0.4, 0.75, 0.75, 0.75, 0.75, 1.0);
    let b0 = blades[iid * 2u];
    let b1 = blades[iid * 2u + 1u];
    let side = sides[vid];
    let t = ts[vid];
    let pos = b0.xyz;
    let angle = b0.w;
    let height = b1.x;
    let width = b1.y;
    let flower = b1.z;
    let rnd = b1.w;

    let facing = vec3f(cos(angle), 0.0, sin(angle));
    let right = vec3f(-facing.z, 0.0, facing.x);
    let time = F.cam_pos.w;
    let wdir = vec3f(F.wind.x, 0.0, F.wind.y);
    let gust = gnoise(pos.xz * 0.06 - F.wind.xy * time * 0.4) * 0.5 + 0.5;
    let lean = 0.2 + rnd * 0.35;
    let flutter = sin(time * 2.7 + rnd * 20.0 + pos.x * 0.4) * 0.06 * F.wind.z;
    var w = width * (1.0 - t * 0.8);
    if (flower > 0.5 && t > 0.9) {
        w = 0.055;
    }
    let y = height * t * (1.0 - 0.25 * (lean + gust * F.wind.z) * t);
    let bend = facing * (lean + flutter) + wdir * gust * F.wind.z * 0.7;
    let p = pos + right * side * w * 0.5 + vec3f(0.0, y, 0.0) + bend * height * t * t;

    var o: VOut;
    o.clip = P.view_proj * vec4f(p, 1.0);
    o.wp = p;
    o.nrm = normalize(facing * 0.5 + vec3f(0.0, 1.0, 0.0));
    o.misc = vec4f(t, flower, rnd, 0.0);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let t = in.misc.x;
    let flower = in.misc.y;
    let rnd = in.misc.z;
    let p = in.wp.xz;
    let dat = data_at(p);
    // the same tint as the ground the blades fade into further out, a bit brighter since
    // the ground's colour averages in the gaps between blades
    var tip = meadow_tint(p, dat, normal_at(p), sky_vis(p)) * (1.1 + 0.55 * rnd);
    tip = mix(tip, vec3f(0.17, 0.15, 0.06), smoothstep(0.82, 1.0, rnd) * 0.6);
    var albedo = mix(vec3f(0.02, 0.038, 0.009), tip, smoothstep(0.0, 0.85, t));
    if (flower > 0.5 && t > 0.85) {
        albedo = flower_color(p) * 1.1;
    }
    let n = normalize(in.nrm);
    let L = F.sun_dir.xyz;
    let V = normalize(F.cam_pos.xyz - in.wp);
    let sh = shadow_at(in.wp, vec3f(0.0, 1.0, 0.0));
    let wrap = max((dot(n, L) + 0.4) / 1.4, 0.0);
    let trans = pow(max(dot(-V, L), 0.0), 4.0) * 0.5 * t;
    let ao = mix(0.2, 1.0, t * t);
    var col = albedo * (sun_light(in.wp) * (wrap + trans) * sh * ao + ambient(n, in.wp) * ao);
    col = apply_fog(col, in.wp);
    return vec4f(col, 1.0);
}
