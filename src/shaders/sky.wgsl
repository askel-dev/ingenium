// Sky (from the sky-view table) and the sun disk with the volumetric clouds composited over
// them, drawn behind everything (reversed-Z depth 0). The clouds come from the
// reduced-resolution buffer of clouds.wgsl.

@group(1) @binding(0) var t_clouds: texture_2d<f32>;

// The disk is drawn this much dimmer than the real sun (which would be ~10^4 times brighter
// than the sky and overflow half floats once bloom spreads it); bloom makes up the glow.
const SUN_DISK_SCALE: f32 = 0.04;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    let ndc = uv * 2.0 - 1.0;
    var o: VOut;
    o.clip = vec4f(ndc, 0.0, 1.0);
    o.ndc = ndc;
    return o;
}

// The sun's disk at its real size (0.53 degrees), darker towards the limb.
fn sun_disk(dir: vec3f) -> vec3f {
    let L = F.sun_dir.xyz;
    // the angle from the sun's centre; sin() of it from a cross product keeps the precision
    // that acos(dot) loses this close to 1
    let ang = asin(min(length(cross(dir, L)), 1.0));
    let aa = max(fwidth(ang), 1e-5);
    let cover = saturate((SUN_ANGULAR_R - ang) / aa + 0.5) * select(0.0, 1.0, dot(dir, L) > 0.0);
    if (cover <= 0.0) {
        return vec3f(0.0);
    }
    let x = min(ang / SUN_ANGULAR_R, 1.0);
    let mu = sqrt(1.0 - x * x);
    // limb darkening, u = 0.6
    let limb = 1.0 - 0.6 * (1.0 - mu);
    // radiance = illuminance / solid angle of the disk
    let radiance = SKY.sun.rgb / (PI * SUN_ANGULAR_R * SUN_ANGULAR_R);
    return radiance * limb * cover * SUN_DISK_SCALE;
}

@fragment
fn fs(in: VOut) -> SceneOut {
    let pn = F.inv_view_proj_rot * vec4f(in.ndc, 1.0, 1.0);
    let dir = normalize(pn.xyz / pn.w);
    let sky = sky_radiance(dir);
    var col = sky + sun_disk(dir);

    // rgb = light the clouds scatter towards us, a = how much of the sky shows through
    let suv = in.clip.xy * F.screen.zw;
    let cl = textureSampleLevel(t_clouds, s_clamp, suv, 0.0);
    col = col * cl.a + cl.rgb;
    if (dir.y > 0.0) {
        // the most distant clouds dissolve into the haze at the horizon
        col = mix(col, sky, (1.0 - smoothstep(0.0, 0.1, dir.y)) * 0.5);
    }
    // the haze near the camera that lies in shadow: the shafts of light under the trees, seen
    // against the sky (whose table has the air and the ground haze, not the haze among the trees)
    let shade = ap_shade_full(suv, NEAR_DIST);
    col = max(col - shade.rgb * shade.a, vec3f(0.0));
    // no surface here for the screen-space lighting to work on
    return SceneOut(vec4f(min(col, vec3f(60000.0)), 1.0), vec4f(0.0));
}
