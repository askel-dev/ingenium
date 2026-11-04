// The sun's view of the terrain and the forest canopy, for the shadows beyond the shadow
// cascades and for long mountain shadows inside them (`sun_height_vis` in globals.wgsl).
// Rebuilt only when the sun moves.
//
// The map is laid out along the sun: each column runs from the side of the world that faces
// the sun to the far side, so each texel follows from the one before it in its column,
//   H(p) = max(occluder(p), H(p + step towards the sun) - step * tan(sun elevation)),
// the height at which the sun's rays over p clear everything between p and the sun. A point
// at height y over p is in sunlight if y > H(p). One thread runs down each column, so the whole
// map costs a few texture reads per texel however long the shadows are.
//
// Per texel: x = H with the forest canopy counted as a solid block, y = how far away (m) the
// occluder that sets it is (for the penumbra), z and w the same for the bare terrain. Heights
// are stored relative to the ground at the texel, so the half floats keep centimetres where
// it matters, on sunlit ground (where H is the ground itself).

struct Params {
    world: f32,
    res: f32,       // heightmap texels per side
    extent: f32,    // metres the map covers along each axis
    tan_el: f32,    // tangent of the sun's elevation
    sun: vec2f,     // horizontal direction towards the sun
    pad: vec2f,
}

@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var t_h: texture_2d<f32>;
@group(0) @binding(2) var t_data: texture_2d<f32>;
@group(0) @binding(3) var o_map: texture_storage_2d<rgba16float, write>;

// Canopy height over closed forest (where scatter.wgsl fills in the trees).
const CANOPY_H: f32 = 17.0;
// Below anything: no occluder (outside the world).
const NONE: f32 = -1000.0;

// Bilinear ground height and forest density at p (m).
fn ground_forest(p: vec2f) -> vec2f {
    let res = i32(U.res);
    let x = p / (U.world / U.res) - 0.5;
    let fl = floor(x);
    let f = x - fl;
    let i = vec2i(fl);
    let hi = vec2i(res - 1);
    var v = array<vec2f, 4>();
    for (var k = 0; k < 4; k++) {
        let c = clamp(i + vec2i(k & 1, k >> 1), vec2i(0), hi);
        v[k] = vec2f(textureLoad(t_h, c, 0).r, textureLoad(t_data, c, 0).r);
    }
    return mix(mix(v[0], v[1], f.x), mix(v[2], v[3], f.x), f.y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let n = textureDimensions(o_map).x;
    if (id.x >= n) {
        return;
    }
    let a = U.sun;
    let b = vec2f(-a.y, a.x);
    let texel = U.extent / f32(n);
    let centre = vec2f(U.world * 0.5);
    let across = (f32(id.x) + 0.5) * texel - U.extent * 0.5;
    let drop = texel * U.tan_el;
    var hc = NONE; // shadow height, canopy counted
    var ht = NONE; // bare terrain
    var dc = 0.0;
    var dt = 0.0;
    for (var j = 0u; j < n; j++) {
        // row 0 lies furthest towards the sun
        let along = U.extent * 0.5 - (f32(j) + 0.5) * texel;
        let p = centre + a * along + b * across;
        let inside = all(p >= vec2f(0.0)) && all(p <= vec2f(U.world));
        let gf = ground_forest(p);
        let ground = gf.x;
        let occ_t = select(NONE, ground, inside);
        let occ_c = select(NONE, ground + CANOPY_H * smoothstep(0.3, 0.55, gf.y), inside);
        // the shadow from up-sun drops along the sun's rays; a higher occluder here takes over
        hc = max(hc - drop, NONE);
        ht = max(ht - drop, NONE);
        dc += texel;
        dt += texel;
        if (occ_c >= hc) {
            hc = occ_c;
            dc = 0.0;
        }
        if (occ_t >= ht) {
            ht = occ_t;
            dt = 0.0;
        }
        textureStore(o_map, vec2u(id.x, j), vec4f(hc - ground, min(dc, 60000.0), ht - ground, min(dt, 60000.0)));
    }
}
