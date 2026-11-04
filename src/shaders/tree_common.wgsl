// Shared by the tree shaders (tree.wgsl) and the impostor bake (impostor_bake.wgsl): the
// per-shape table, the layout of the impostor views, the crown's self-shadowing and the bark.
// forest.rs puts SLOTS, IMP_FRAMES and IMP_WARP in front.

// One tree shape (treegen.rs). A tree's slot is its species * 8 + the low 3 bits of its seed.
struct Shape {
    crown_c: vec4f, // crown ellipsoid centre (object space, m), w = sunlight extinction in it (1/m)
    crown_r: vec4f, // crown ellipsoid radii, w = species
    imp: vec4f,     // impostor: x = centre height, y = half size of a view, z = depth range, w = atlas layer
}

@group(1) @binding(1) var<uniform> SHAPES: array<Shape, SLOTS>;

fn shape_slot(packed: u32) -> u32 {
    return (packed & 3u) * 8u + ((packed >> 18u) & 7u);
}

// Rotation about y by the angle with this cosine and sine, and back.
fn rot_y(v: vec3f, cs: vec2f) -> vec3f {
    return vec3f(cs.x * v.x - cs.y * v.z, v.y, cs.y * v.x + cs.x * v.z);
}

fn unrot_y(v: vec3f, cs: vec2f) -> vec3f {
    return vec3f(cs.x * v.x + cs.y * v.z, v.y, -cs.y * v.x + cs.x * v.z);
}

// Share of the sunlight (towards l, object space) that reaches p through the tree's own crown,
// from the length of the sun ray inside the crown's ellipsoid. The inside of a crown and its
// side away from the sun come out dark and its sunlit clumps bright, at any distance and
// without a shadow map.
fn crown_light(p: vec3f, l: vec3f, s: Shape, scale: f32) -> f32 {
    let r = s.crown_r.xyz;
    let o = (p - s.crown_c.xyz) / r;
    let d = l / r;
    let a = dot(d, d);
    let b = dot(o, d);
    let c = dot(o, o) - 1.0;
    let disc = b * b - a * c;
    if (disc <= 0.0) {
        return 1.0;
    }
    let q = sqrt(disc);
    let len = max((-b + q) / a - max((-b - q) / a, 0.0), 0.0) * scale;
    // some light always finds its way through the gaps
    return mix(0.12, 1.0, exp(-s.crown_c.w * len));
}

// ---------------------------------------------------------------------------
// Impostor views (impostor.rs): an IMP_FRAMES² grid over the upper hemisphere, hemi-octahedral
// with its rings pulled towards the horizon.

fn imp_decode(e: vec2f) -> vec3f {
    let x = (e.x + e.y) * 0.5;
    let z = (e.x - e.y) * 0.5;
    let r = abs(x) + abs(z);
    let el = pow(saturate(1.0 - r), IMP_WARP) * (PI * 0.5);
    let hl = length(vec2f(x, z));
    let h = select(vec2f(1.0, 0.0), vec2f(x, z) / hl, hl > 1e-6);
    return vec3f(h.x * cos(el), sin(el), h.y * cos(el));
}

// Below the horizon counts as level.
fn imp_encode(d: vec3f) -> vec2f {
    let el = asin(clamp(d.y, 0.0, 1.0));
    let r = 1.0 - pow(el / (PI * 0.5), 1.0 / IMP_WARP);
    let l1 = abs(d.x) + abs(d.z);
    let xz = select(vec2f(0.0), d.xz / l1 * r, l1 > 1e-6);
    return vec2f(xz.x + xz.y, xz.x - xz.y);
}

fn imp_frame_dir(cell: vec2u) -> vec3f {
    return imp_decode(vec2f(cell) / f32(IMP_FRAMES - 1u) * 2.0 - 1.0);
}

// A view's image axes (it looks along -d).
struct Axes {
    r: vec3f,
    u: vec3f,
}

fn imp_axes(d: vec3f) -> Axes {
    let c = cross(vec3f(0.0, 1.0, 0.0), d);
    let r = select(vec3f(1.0, 0.0, 0.0), normalize(c), dot(c, c) > 1e-10);
    return Axes(r, cross(d, r));
}

// Octahedral encoding of a unit vector in [-1, 1]².
fn oct_encode(n: vec3f) -> vec2f {
    let m = n / (abs(n.x) + abs(n.y) + abs(n.z));
    var p = m.xy;
    if (m.z < 0.0) {
        p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    }
    return p;
}

fn oct_decode(e: vec2f) -> vec3f {
    var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// ---------------------------------------------------------------------------

fn bark(species: u32, uv: vec2f) -> vec3f {
    let n1 = gnoise(vec2f(uv.x * 7.0, uv.y * 1.3));
    let n2 = gnoise(vec2f(uv.x * 23.0, uv.y * 5.0));
    if (species == 1u) {
        let marks = smoothstep(0.3, 0.6, gnoise(vec2f(uv.x * 3.0, uv.y * 7.0)));
        let c = vec3f(0.62, 0.60, 0.56) * (0.88 + 0.12 * n2);
        return mix(c, vec3f(0.05, 0.045, 0.04), marks * 0.85);
    }
    let ridges = abs(n1);
    var base = vec3f(0.16, 0.14, 0.12);
    if (species == 2u) {
        base = vec3f(0.20, 0.12, 0.08);
    }
    return base * (0.5 + 0.7 * ridges + 0.2 * n2);
}

fn canopy_color(species: u32) -> vec3f {
    switch species {
        case 1u: { return vec3f(0.075, 0.12, 0.028); }
        case 2u: { return vec3f(0.03, 0.055, 0.026); }
        default: { return vec3f(0.055, 0.10, 0.02); }
    }
}
