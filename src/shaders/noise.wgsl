// Hashing and gradient noise shared by every shader.

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash_i2(p: vec2i) -> u32 {
    return pcg(bitcast<u32>(p.x) ^ pcg(bitcast<u32>(p.y) + 0x9e3779b9u));
}

fn u01(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn grad2(p: vec2i) -> vec2f {
    let a = u01(hash_i2(p)) * TAU;
    return vec2f(cos(a), sin(a));
}

// Perlin-style gradient noise, roughly in [-1, 1].
fn gnoise(p: vec2f) -> f32 {
    let fl = floor(p);
    let i = vec2i(fl);
    let f = p - fl;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(grad2(i), f);
    let b = dot(grad2(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
    let c = dot(grad2(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
    let d = dot(grad2(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 1.4;
}

fn octave_step(q: vec2f) -> vec2f {
    // rotate ~37 degrees and scale by 2 so octaves don't line up
    return vec2f(q.x * 1.6 - q.y * 1.2, q.x * 1.2 + q.y * 1.6) + vec2f(19.1, 7.7);
}

fn fbm(p: vec2f, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i++) {
        sum += amp * gnoise(q);
        q = octave_step(q);
        amp *= 0.5;
    }
    return sum;
}

fn ridged(p: vec2f, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    var prev = 1.0;
    for (var i = 0; i < octaves; i++) {
        var n = 1.0 - abs(gnoise(q));
        n = n * n;
        sum += n * amp * prev;
        prev = n;
        q = octave_step(q);
        amp *= 0.5;
    }
    return sum;
}
