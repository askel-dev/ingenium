// The stats overlay: one quad in the top-right corner, a translucent bar with a 5x9 pixel font.
// The text is a list of glyph indices; each glyph is 9 rows of 5 bits packed into three words
// (4 rows a word, 8 bits a row, bit 4 the leftmost pixel), glyph i at words 3i..3i+2.

struct Hud {
    screen: vec4f,  // width, height (px)
    rect: vec4f,    // bar x0, y0, x1, y1 (px)
    origin: vec4f,  // text x, y (px), pixel scale, glyph count
    text: array<vec4u, 12>,
    font: array<vec4u, 15>,
}

@group(0) @binding(0) var<uniform> hud: Hud;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let c = vec2f(f32(i & 1u), f32(i >> 1u));
    let p = mix(hud.rect.xy, hud.rect.zw, c);
    return vec4f(p.x / hud.screen.x * 2.0 - 1.0, 1.0 - p.y / hud.screen.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let local = (pos.xy - hud.origin.xy) / hud.origin.z;
    var lit = false;
    if local.x >= 0.0 && local.y >= 0.0 && local.y < 9.0 {
        let cell = u32(local.x) / 6u;
        let gx = u32(local.x) - cell * 6u;
        if cell < u32(hud.origin.w) && gx < 5u {
            let code = hud.text[cell / 4u][cell % 4u];
            let gy = u32(local.y);
            let k = code * 3u + gy / 4u;
            let word = hud.font[k / 4u][k % 4u];
            let row = (word >> ((gy % 4u) * 8u)) & 0x1fu;
            lit = ((row >> (4u - gx)) & 1u) == 1u;
        }
    }
    if lit {
        return vec4f(0.78, 0.78, 0.78, 1.0);
    }
    return vec4f(0.015, 0.015, 0.015, 0.62);
}
