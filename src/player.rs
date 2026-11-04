//! First-person player: walks on the terrain with gravity, jumps, sprints and crouches,
//! slides off slopes too steep to climb and bumps into tree trunks. With `fly` set it is
//! the free-fly (noclip) camera instead.

use crate::camera::Camera;
use crate::forest::SCATTER_CELL;
use crate::terrain::Terrain;
use crate::treegen::TRUNK_R;
use glam::{IVec2, Vec2, Vec3, Vec3Swizzles};

const RADIUS: f32 = 0.35;
const EYE_STAND: f32 = 1.65;
const EYE_CROUCH: f32 = 0.95;
const WALK_SPEED: f32 = 4.5;
const SPRINT_SPEED: f32 = 8.5;
const CROUCH_SPEED: f32 = 2.0;
const GROUND_ACCEL: f32 = 12.0; // 1/s: how quickly ground speed follows the input
const AIR_ACCEL: f32 = 2.0; // limited steering in the air
const GRAVITY: f32 = 20.0; // above 9.8 so jumps don't feel floaty
const JUMP_SPEED: f32 = 6.6; // ~1.1 m jump
const MAX_SLOPE_NY: f32 = 0.70; // ~45°: steeper ground can't be climbed or landed on
const SLIDE_NY: f32 = 0.66; // ~49°: standing on ground steeper than this slides (hysteresis)
const SNAP_DOWN: f32 = 0.3; // stay on the ground when walking downhill
const SLIDE_DRAG: f32 = 0.6;
const COYOTE: f32 = 0.12; // a jump still works this long after leaving the ground
const JUMP_BUFFER: f32 = 0.15; // a jump pressed this long before landing still happens
const SUBSTEP: f32 = 1.0 / 120.0;
const FLY_SPEED: f32 = 25.0;
const LOOK_SENSITIVITY: f32 = 0.0022;
const BOB_V: f32 = 0.04;
const BOB_H: f32 = 0.02;

/// What the keyboard asks for this frame.
#[derive(Default)]
pub struct Input {
    pub forward: f32,
    pub right: f32,
    /// Flying only: up minus down.
    pub up: f32,
    /// Sprint when walking, fast when flying.
    pub sprint: bool,
    /// Crouch when walking, slow when flying.
    pub crouch: bool,
}

pub struct Player {
    /// Feet position.
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fly: bool,
    grounded: bool,
    eye: f32,
    coyote: f32,
    jump_buffer: f32,
    bob_phase: f32,
    bob: f32,
    dip: f32,
    dip_vel: f32,
    fov_scale: f32,
}

impl Player {
    pub fn new() -> Player {
        Player {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly: false,
            grounded: false,
            eye: EYE_STAND,
            coyote: 0.0,
            jump_buffer: 0.0,
            bob_phase: 0.0,
            bob: 0.0,
            dip: 0.0,
            dip_vel: 0.0,
            fov_scale: 1.0,
        }
    }

    /// Take over from another camera: same view direction; flying starts at the camera,
    /// walking starts on the ground at the nearest open spot (out of thick forest, on
    /// walkable ground), clear of trunks.
    pub fn take_over(&mut self, cam: &Camera, terrain: &Terrain, trunks: &Trunks) {
        self.yaw = cam.yaw;
        self.pitch = cam.pitch.clamp(-1.5, 1.5);
        self.vel = Vec3::ZERO;
        self.pos = cam.pos - Vec3::Y * self.eye;
        if !self.fly {
            let open = find_open_spot(cam.pos.xz(), terrain, trunks);
            self.pos.x = open.x;
            self.pos.z = open.y;
            trunks.push_out(&mut self.pos, &mut self.vel, RADIUS);
            self.pos.y = terrain.ground_at(self.pos.x, self.pos.z);
            self.grounded = true;
        }
    }

    pub fn set_fly(&mut self, fly: bool) {
        // keep the eye where it is
        self.pos.y += self.eye - EYE_STAND;
        self.eye = EYE_STAND;
        self.fly = fly;
        self.vel = Vec3::ZERO;
        self.grounded = false;
        self.dip = 0.0;
        self.dip_vel = 0.0;
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - dy * LOOK_SENSITIVITY).clamp(-1.5, 1.5);
    }

    pub fn jump(&mut self) {
        self.jump_buffer = JUMP_BUFFER;
    }

    pub fn update(&mut self, input: &Input, dt: f32, terrain: &Terrain, trunks: &Trunks) {
        if self.fly {
            self.fly_step(input, dt, terrain);
        } else {
            let steps = (dt / SUBSTEP).ceil().max(1.0);
            for _ in 0..steps as u32 {
                self.step(input, dt / steps, terrain, trunks);
            }
        }
        self.animate(input, dt);
    }

    pub fn camera(&self, fov_y: f32) -> Camera {
        let right = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin());
        let sway = self.bob_phase.sin();
        let bob = Vec3::Y * (self.dip - BOB_V * self.bob * sway * sway) + right * (BOB_H * self.bob * sway);
        Camera { pos: self.pos + Vec3::Y * self.eye + bob, yaw: self.yaw, pitch: self.pitch, fov_y: fov_y * self.fov_scale }
    }

    fn fly_step(&mut self, input: &Input, dt: f32, terrain: &Terrain) {
        let mut speed = FLY_SPEED;
        if input.sprint {
            speed *= 6.0;
        }
        if input.crouch {
            speed *= 0.2;
        }
        let cam = self.camera(1.0);
        let mv = cam.forward() * input.forward + cam.right() * input.right + Vec3::Y * input.up;
        self.vel = mv.normalize_or_zero() * speed;
        self.pos += self.vel * dt;
        self.clamp_to_world(terrain);
        self.pos.y = self.pos.y.max(terrain.ground_at(self.pos.x, self.pos.z));
    }

    fn step(&mut self, input: &Input, h: f32, terrain: &Terrain, trunks: &Trunks) {
        // horizontal: follow the input on the ground, steer a little in the air
        let (s, c) = self.yaw.sin_cos();
        let wish = (Vec2::new(s, -c) * input.forward + Vec2::new(c, s) * input.right).clamp_length_max(1.0);
        let speed = if input.crouch {
            CROUCH_SPEED
        } else if input.sprint {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        let mut v = self.vel.xz();
        if self.grounded {
            v = v.lerp(wish * speed, 1.0 - (-GROUND_ACCEL * h).exp());
        } else if wish != Vec2::ZERO {
            let dir = wish.normalize();
            let add = (speed * wish.length() - v.dot(dir)).clamp(0.0, AIR_ACCEL * speed * h);
            v += dir * add;
        }
        // no walking up slopes that are too steep: drop the uphill part of the motion
        let ahead = self.pos.xz() + v * h + v.normalize_or_zero() * RADIUS;
        let n = terrain.normal_at(ahead.x, ahead.y);
        if n.y < MAX_SLOPE_NY {
            let downhill = n.xz().normalize_or_zero();
            let uphill_speed = -v.dot(downhill);
            if uphill_speed > 0.0 {
                v += downhill * uphill_speed;
            }
        }
        self.vel.x = v.x;
        self.vel.z = v.y;

        // vertical: jump, gravity
        self.coyote = if self.grounded { COYOTE } else { self.coyote - h };
        self.jump_buffer -= h;
        if self.jump_buffer > 0.0 && self.coyote > 0.0 {
            self.vel.y = JUMP_SPEED;
            self.grounded = false;
            self.jump_buffer = 0.0;
            self.coyote = 0.0;
        }
        if self.grounded {
            self.vel.y = 0.0;
        } else {
            self.vel.y -= GRAVITY * h;
        }

        self.pos += self.vel * h;
        trunks.push_out(&mut self.pos, &mut self.vel, RADIUS);
        self.clamp_to_world(terrain);

        // ground contact
        let g = terrain.ground_at(self.pos.x, self.pos.z);
        let n = terrain.normal_at(self.pos.x, self.pos.z);
        let walkable = n.y >= if self.grounded { SLIDE_NY } else { MAX_SLOPE_NY };
        if self.pos.y <= g || (self.grounded && walkable && self.pos.y - g < SNAP_DOWN) {
            self.pos.y = g;
            if walkable {
                if !self.grounded {
                    // landing: dip the camera, harder for bigger falls
                    self.dip_vel -= (-self.vel.y * 0.4).clamp(0.0, 7.0);
                }
                self.grounded = true;
                self.vel.y = 0.0;
            } else {
                // too steep to stand on: slide down along the slope
                self.grounded = false;
                let into = self.vel.dot(n);
                if into < 0.0 {
                    self.vel -= n * into;
                }
                self.vel *= (-SLIDE_DRAG * h).exp();
            }
        } else {
            self.grounded = false;
        }

        // landing dip: critically damped spring back to zero
        let w = 11.0;
        self.dip_vel += (-w * w * self.dip - 2.0 * w * self.dip_vel) * h;
        self.dip += self.dip_vel * h;
    }

    fn animate(&mut self, input: &Input, dt: f32) {
        let ease = |k: f32| 1.0 - (-k * dt).exp();
        let crouch = input.crouch && !self.fly;
        self.eye += (if crouch { EYE_CROUCH } else { EYE_STAND } - self.eye) * ease(12.0);

        let speed = if self.grounded { self.vel.xz().length() } else { 0.0 };
        // one bob cycle per step; steps get longer as you go faster
        self.bob_phase = (self.bob_phase + speed * dt / (1.2 + 0.12 * speed) * std::f32::consts::PI) % std::f32::consts::TAU;
        self.bob += ((speed / WALK_SPEED).min(1.3) - self.bob) * ease(10.0);

        let sprinting = input.sprint && !crouch && speed > WALK_SPEED + 0.5;
        self.fov_scale += (if sprinting { 1.06 } else { 1.0 } - self.fov_scale) * ease(6.0);
    }

    fn clamp_to_world(&mut self, terrain: &Terrain) {
        let w = terrain.world;
        self.pos.x = self.pos.x.clamp(1.0, w - 1.0);
        self.pos.z = self.pos.z.clamp(1.0, w - 1.0);
    }
}

const SPAWN_CLEAR_R: f32 = 16.0; // look this far around a spawn candidate for trunks
const SPAWN_MAX_TRUNKS: usize = 2; // ...and accept it with at most this many
const SPAWN_STEP: f32 = 10.0; // spacing of the search rings
const SPAWN_SEARCH_R: f32 = 2000.0;

/// Nearest point to `from` (searching outward in rings) that is out of the thick forest
/// and on walkable, in-world ground; `from` itself if nothing qualifies.
fn find_open_spot(from: Vec2, terrain: &Terrain, trunks: &Trunks) -> Vec2 {
    let margin = SPAWN_CLEAR_R + 8.0;
    let ok = |p: Vec2| {
        p.x > margin
            && p.y > margin
            && p.x < terrain.world - margin
            && p.y < terrain.world - margin
            && terrain.normal_at(p.x, p.y).y > 0.9
            && trunks.count_within(p, SPAWN_CLEAR_R) <= SPAWN_MAX_TRUNKS
    };
    if ok(from) {
        return from;
    }
    let rings = (SPAWN_SEARCH_R / SPAWN_STEP) as i32;
    for ring in 1..=rings {
        let r = ring as f32 * SPAWN_STEP;
        let n = (std::f32::consts::TAU * r / SPAWN_STEP).ceil() as i32;
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let p = from + Vec2::new(a.cos(), a.sin()) * r;
            if ok(p) {
                return p;
            }
        }
    }
    from
}

/// Tree trunks as vertical cylinders, on a grid with one cell per scatter cell (the
/// scatter puts at most one tree in each). A cell packs the trunk's offset inside the
/// cell (12 + 12 bits) and its radius (8 bits); 0 means no tree.
pub struct Trunks {
    dim: i32,
    cells: Vec<u32>,
}

const RADIUS_UNIT: f32 = 1.0 / 200.0;

impl Trunks {
    /// `trees` are the records written by scatter.wgsl.
    pub fn new(trees: &[[u32; 4]], world: f32) -> Trunks {
        let dim = (world / SCATTER_CELL) as i32;
        let mut cells = vec![0u32; (dim * dim) as usize];
        let quant = |f: f32| ((f * 4095.0).round() as u32).min(4095);
        for t in trees {
            let p = Vec2::new(f32::from_bits(t[0]), f32::from_bits(t[2])) / SCATTER_CELL;
            let c = p.floor().as_ivec2();
            if c.x < 0 || c.y < 0 || c.x >= dim || c.y >= dim {
                continue;
            }
            // bits 0-1 species, 10-17 scale (decoded as in tree.wgsl)
            let species = (t[3] & 3).min(2) as usize;
            let scale = 0.75 + ((t[3] >> 10) & 255) as f32 / 255.0 * 0.55;
            let r = ((TRUNK_R[species] * scale / RADIUS_UNIT).round() as u32).clamp(1, 255);
            let f = p - c.as_vec2();
            cells[(c.y * dim + c.x) as usize] = quant(f.x) | quant(f.y) << 12 | r << 24;
        }
        Trunks { dim, cells }
    }

    /// Number of trunks whose centre lies within `r` of `p`.
    fn count_within(&self, p: Vec2, r: f32) -> usize {
        let lo = ((p - r) / SCATTER_CELL).floor().as_ivec2().max(IVec2::ZERO);
        let hi = ((p + r) / SCATTER_CELL).floor().as_ivec2().min(IVec2::splat(self.dim - 1));
        let mut n = 0;
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let v = self.cells[(z * self.dim + x) as usize];
                if v == 0 {
                    continue;
                }
                let off = Vec2::new((v & 4095) as f32, (v >> 12 & 4095) as f32) / 4095.0;
                let c = (IVec2::new(x, z).as_vec2() + off) * SCATTER_CELL;
                if c.distance_squared(p) <= r * r {
                    n += 1;
                }
            }
        }
        n
    }

    /// (centre, radius) of the trunks in the 3x3 cells around `p`.
    fn near(&self, p: Vec2) -> impl Iterator<Item = (Vec2, f32)> + '_ {
        let c = (p / SCATTER_CELL).floor().as_ivec2();
        (-1..=1)
            .flat_map(move |dz| (-1..=1).map(move |dx| c + IVec2::new(dx, dz)))
            .filter(move |q| q.x >= 0 && q.y >= 0 && q.x < self.dim && q.y < self.dim)
            .filter_map(move |q| {
                let v = self.cells[(q.y * self.dim + q.x) as usize];
                let off = Vec2::new((v & 4095) as f32, (v >> 12 & 4095) as f32) / 4095.0;
                (v != 0).then(|| ((q.as_vec2() + off) * SCATTER_CELL, (v >> 24) as f32 * RADIUS_UNIT))
            })
    }

    /// Push a body of `radius` at `pos` out of any trunk and drop its velocity into it,
    /// so it slides around trees.
    fn push_out(&self, pos: &mut Vec3, vel: &mut Vec3, radius: f32) {
        for _ in 0..3 {
            let mut moved = false;
            for (c, r) in self.near(pos.xz()) {
                let d = pos.xz() - c;
                let min = r + radius;
                if d.length_squared() >= min * min {
                    continue;
                }
                let n = d.try_normalize().unwrap_or(Vec2::X);
                let p = c + n * min;
                pos.x = p.x;
                pos.z = p.y;
                let into = vel.xz().dot(n);
                if into < 0.0 {
                    vel.x -= n.x * into;
                    vel.z -= n.y * into;
                }
                moved = true;
            }
            if !moved {
                break;
            }
        }
    }
}
