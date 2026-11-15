// The canopy light volume's passes (canopy.rs). Each runs over a box of one level's voxels: the
// slabs the level scrolled over, a strip it refreshes, or all of it.
//
// density: extinction (1/m) of the trees' leaves and bark in each voxel, gathered from the trees
//          whose cells are around it, from each kind's baked density (canopy.rs, `bake_kinds`).
// sun:     how much sun reaches each voxel (marched towards the sun through ever coarser levels,
//          0 under a hill), and how much of the light from straight above.
// light:   16 cones over the sky from each voxel: what share of each the leaves let through,
//          projected onto SH-L1, and how sunlit and sky-lit the leaves that block it are on
//          average. It reads the sun pass, not itself, so it needs no frames to settle.
//
// A voxel under the ground takes the light half a metre above it, so a surface between the two
// isn't darkened by the filtering.

struct CanopyBox {
    min: vec2i,    // first voxel (world voxel coordinates of its level)
    size: vec2u,   // voxels
    level: u32,
    passes: u32,
}

// How far out a level's own voxels are used (in half-widths): all of it but the half voxel
// whose filter would reach round to the other side.
const C_VALID: f32 = 0.99;

struct CanopyStatic {
    kinds: array<vec4f, 12>,   // x centre height, y half side of the kind's box (tree space), z its voxel size
    dims: vec4u,               // x scatter cells per side, y tree count
    reach: vec4u,              // scatter cells to look through around a column, per level
    misc: vec4f,               // x scatter cell size (m), y highest terrain (m)
}

@group(1) @binding(0) var<uniform> BX: CanopyBox;
@group(1) @binding(1) var<uniform> CS: CanopyStatic;
@group(1) @binding(2) var<storage, read> trees: array<vec4u>;
@group(1) @binding(3) var<storage, read> cells: array<u32>;
@group(1) @binding(4) var t_kind: texture_3d<f32>;
@group(1) @binding(5) var s_kind: sampler;
@group(1) @binding(6) var o_density: texture_storage_3d<rgba16float, write>;
@group(1) @binding(7) var t_dens: texture_3d<f32>;
@group(1) @binding(9) var o_sun: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(10) var t_sunv: texture_3d<f32>;
@group(1) @binding(11) var o_light: texture_storage_3d<rgba16float, write>;

const KINDS: u32 = 12u;
const KIND_RES: u32 = 32u;
const KIND_MIPS: f32 = 6.0;
// cones over the sky, in rings of 4
const CONES: u32 = 16u;
// a cone's width over its length (16 cones share the hemisphere: half-angle ~0.36 rad)
const CONE_WIDTH: f32 = 0.74;

// level l's texel for world voxel (ix, iz), row k of the level
fn c_texel(l: u32, ix: i32, iz: i32, k: i32) -> vec3i {
    let n = i32(C_N);
    return vec3i(((ix % n) + n) % n, i32(F.canopy[l].w) + k, ((iz % n) + n) % n);
}

fn rows_of(l: u32) -> i32 {
    return i32(round((C_Y1 - C_Y0) / c_dy(l)));
}

fn c_density(l: u32, p: vec3f, h: f32) -> f32 {
    let c = c_coords(l, p, h);
    return textureSampleLevel(t_dens, s_canopy, vec3f(c.x, c.y / C_NY, c.z), 0.0).r;
}

// the level whose voxels are about `width` across, but no finer than `first`
fn level_for(width: f32, first: u32) -> u32 {
    return max(first, u32(clamp(ceil(log2(max(width, 1.0))), 0.0, f32(C_LEVELS - 1u))));
}

// One thread per voxel, in x, then z, then row order over the box: a slab one voxel thick
// keeps every lane busy too.
fn box_voxel(i: u32) -> vec3i {
    let n = BX.size.x * BX.size.y;
    return vec3i(BX.min.x + i32(i % BX.size.x), BX.min.y + i32((i / BX.size.x) % BX.size.y), i32(i / n));
}

fn box_count() -> u32 {
    return BX.size.x * BX.size.y * u32(rows_of(BX.level));
}

@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= box_count()) {
        return;
    }
    let l = BX.level;
    let vs = F.canopy[l].z;
    let dy = c_dy(l);
    let b = box_voxel(id.x);
    let xz = (vec2f(f32(b.x), f32(b.y)) + 0.5) * vs;
    let h = height_at(xz);
    let y = h + C_Y0 + (f32(b.z) + 0.5) * dy;
    // A voxel wider than it is tall takes 2 x 2 samples across, each of the kind's mip about
    // as fine as the voxel is tall (or as the samples are apart).
    let ns = i32(clamp(vs / dy, 1.0, 2.0));
    let foot = max(vs / f32(ns), dy);
    let wsub = 1.0 / f32(ns * ns);
    var acc = 0.0;
    let dim = i32(CS.dims.x);
    let c0 = vec2i(floor(xz / CS.misc.x));
    let r = i32(CS.reach[l]);
    for (var dz = -r; dz <= r; dz++) {
        for (var dx = -r; dx <= r; dx++) {
            let c = c0 + vec2i(dx, dz);
            if (any(c < vec2i(0)) || any(c >= vec2i(dim))) {
                continue;
            }
            let t = cells[c.y * dim + c.x];
            if (t == 0u) {
                continue;
            }
            let i = inst(trees[t - 1u]);
            let kind = min(i.species * 4u + i.variant, KINDS - 1u);
            let K = CS.kinds[kind];
            let size = inst_size(i);
            // the voxel's height in the kind's box (0..1 up it), give or take half a voxel
            let v = ((y - i.base.y) / size.y - K.x) / (2.0 * K.y) + 0.5;
            let my = 0.5 * dy / (2.0 * K.y * size.y);
            if (v < -my || v > 1.0 + my) {
                continue;
            }
            // Leaf and bark areas grow with the square of the tree's size and its volume with
            // the cube: the extinction goes as one over its size.
            let s_avg = pow(size.x * size.x * size.y, 1.0 / 3.0);
            let lod = clamp(log2(foot / (K.z * s_avg)), 0.0, KIND_MIPS - 1.0);
            // the kinds are stacked along z: keep the filter inside this one's
            let hw = 0.5 / f32(KIND_RES >> u32(ceil(lod)));
            let mxz = 0.5 * foot / (2.0 * K.y * size.x);
            for (var sz = 0; sz < ns; sz++) {
                for (var sx = 0; sx < ns; sx++) {
                    let q = xz + (vec2f(f32(sx), f32(sz)) + 0.5 - 0.5 * f32(ns)) * (vs / f32(ns));
                    let ql = inst_local(i, vec3f(q.x - i.base.x, 0.0, q.y - i.base.z));
                    let uv = ql.xz / (2.0 * K.y) + 0.5;
                    if (any(uv < vec2f(-mxz)) || any(uv > vec2f(1.0 + mxz))) {
                        continue;
                    }
                    let uvw = vec3f(saturate(uv.x), saturate(v), (f32(kind) + clamp(uv.y, hw, 1.0 - hw)) / f32(KINDS));
                    acc += textureSampleLevel(t_kind, s_kind, uvw, lod).r * (wsub / s_avg);
                }
            }
        }
    }
    textureStore(o_density, c_texel(l, b.x, b.y, b.z), vec4f(acc, 0.0, 0.0, 0.0));
}

struct Voxel {
    ix: i32,
    iz: i32,
    k: i32,
    p: vec3f,   // its centre, or half a metre over the ground if that is under it
}

fn voxel(i: u32) -> Voxel {
    let l = BX.level;
    let vs = F.canopy[l].z;
    let b = box_voxel(i);
    var v: Voxel;
    v.ix = b.x;
    v.iz = b.y;
    v.k = b.z;
    let xz = (vec2f(f32(v.ix), f32(v.iz)) + 0.5) * vs;
    let y = max(C_Y0 + (f32(v.k) + 0.5) * c_dy(l), 0.5);
    v.p = vec3f(xz.x, height_at(xz) + y, xz.y);
    return v;
}

@compute @workgroup_size(64)
fn sun(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= box_count()) {
        return;
    }
    let l = BX.level;
    let vs = F.canopy[l].z;
    let v = voxel(id.x);
    var L = F.sun_dir.xyz;
    L = normalize(vec3f(L.x, max(L.y, 0.02), L.z));
    var tau = 0.0;
    var t = 0.5 * vs;
    for (var s = 0; s < 44; s++) {
        let p = v.p + L * t;
        if (p.y > CS.misc.y) {
            break; // above every hill, and the trees on them
        }
        let h = height_fast(p.xz);
        let yr = p.y - h;
        if (yr < 0.0) {
            tau = 100.0; // a hill's shadow
            break;
        }
        // Steps that grow with the distance, through ever coarser levels: the shadow softens
        // the further the leaves casting it are. Beyond the volume only hills are left to hit,
        // as over the canopy.
        let lc = c_level_at(p.xz, l, C_VALID);
        var dt = max(t * 0.2, 0.7 * F.canopy[min(lc, C_LEVELS - 1u)].z);
        if (lc < C_LEVELS && yr < C_Y1) {
            tau += c_density(level_for(dt * 1.25, lc), p, h) * dt;
        } else {
            dt = max(dt, t * 0.35);
        }
        t += dt;
        if (tau > 7.0) {
            break;
        }
    }
    // straight up, through the voxel's own column
    let dy = c_dy(l);
    var tau_up = 0.5 * textureLoad(t_dens, c_texel(l, v.ix, v.iz, v.k), 0).r * dy;
    for (var k = v.k + 1; k < rows_of(l); k++) {
        tau_up += textureLoad(t_dens, c_texel(l, v.ix, v.iz, k), 0).r * dy;
    }
    textureStore(o_sun, c_texel(l, v.ix, v.iz, v.k), vec4f(exp(-tau), exp(-tau_up), 0.0, 1.0));
}

@compute @workgroup_size(64)
fn light(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= box_count()) {
        return;
    }
    let l = BX.level;
    let vs = F.canopy[l].z;
    let v = voxel(id.x);
    // the cones turned by a random angle per voxel, so neighbours miss different gaps
    let rot = u01(hash_i2(vec2i(v.ix * 31 + v.k * 7, v.iz * 17 + i32(l) * 1031))) * TAU;
    var sh = vec4f(0.0);
    var occ = vec3f(0.0); // light absorbed, and its share on sunlit and on sky-lit leaves
    for (var i = 0u; i < CONES; i++) {
        // 4 rings of equal solid angle (mean height 1/2, as for the whole hemisphere), 4 cones
        // a ring, each ring turned by 1/8 of a turn from the one above
        let ring = i / 4u;
        let y = 1.0 - (f32(ring) + 0.5) / 4.0;
        let az = rot + f32(i % 4u) * (0.5 * PI) + f32(ring) * (0.125 * PI);
        let rxz = sqrt(1.0 - y * y);
        let d = vec3f(rxz * cos(az), y, rxz * sin(az));
        var T = 1.0;
        var t = 0.5 * vs;
        for (var s = 0; s < 18; s++) {
            // the step from t to 1.6 t (about the cone's width), sampled at its middle at the
            // level as wide as the cone
            let tm = t * 1.3;
            let p = v.p + d * tm;
            let lc = c_level_at(p.xz, l, C_VALID);
            if (lc >= C_LEVELS) {
                break;
            }
            let h = height_fast(p.xz);
            let yr = p.y - h;
            // over the canopy there is nothing left; under the ground is the terrain's (sky_vis)
            if (yr > C_Y1 || yr < C_Y0) {
                break;
            }
            let lw = level_for(tm * CONE_WIDTH, lc);
            let c = c_coords(lw, p, h);
            let uvw = vec3f(c.x, c.y / C_NY, c.z);
            let sigma = textureSampleLevel(t_dens, s_canopy, uvw, 0.0).r;
            let a = (1.0 - exp(-sigma * t * 0.6)) * T;
            if (a > 1e-4) {
                let st = textureSampleLevel(t_sunv, s_canopy, uvw, 0.0).rg;
                occ += vec3f(a, a * st.r, a * st.g);
                T -= a;
                if (T < 0.01) {
                    T = 0.0;
                    break;
                }
            }
            t *= 1.6;
        }
        sh += T * vec4f(0.282095, 0.488603 * d);
    }
    sh *= TAU / f32(CONES);
    let lit = select(vec2f(1.0), occ.yz / max(occ.x, 1e-6), occ.x > 1e-3);
    let tc = c_texel(l, v.ix, v.iz, v.k);
    let t_sun = textureLoad(t_sunv, tc, 0).r;
    textureStore(o_light, tc, sh);
    textureStore(o_light, tc + vec3i(0, i32(C_NY), 0), vec4f(lit, t_sun, 1.0));
}
