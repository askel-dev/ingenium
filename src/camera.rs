//! Fly camera and the automatic cinematic flythrough.

use crate::terrain::Terrain;
use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.sin() * self.pitch.cos(), self.pitch.sin(), -self.yaw.cos() * self.pitch.cos())
    }

    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.pos, self.forward(), Vec3::Y)
    }

    /// Reversed-Z, infinite far plane.
    pub fn proj(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite_reverse(self.fov_y, aspect, 0.1)
    }

    pub fn look_dir(&mut self, dir: Vec3) {
        let d = dir.normalize();
        self.pitch = d.y.clamp(-0.99, 0.99).asin();
        self.yaw = d.x.atan2(-d.z);
    }
}

/// A closed Catmull-Rom loop around the valley: low over meadows, above the canopy over forest.
pub struct FlyPath {
    points: Vec<Vec3>,
    seg_len: Vec<f32>,
    total: f32,
}

impl FlyPath {
    pub fn new(terrain: &Terrain) -> FlyPath {
        let c = Vec2::splat(terrain.world * 0.5);
        let n = 72;
        let mut pts: Vec<Vec3> = Vec::with_capacity(n);
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let r = 1500.0 + 450.0 * (3.0 * a).sin() + 260.0 * (5.0 * a + 1.0).sin();
            let p = c + Vec2::new(a.cos(), a.sin()) * r;
            pts.push(Vec3::new(p.x, 0.0, p.y));
        }
        // altitude: sample the max ground + canopy clearance in a small neighbourhood
        let mut alts = vec![0.0f32; n];
        for i in 0..n {
            let p = pts[i];
            let mut need: f32 = 0.0;
            for dx in [-30.0, 0.0, 30.0] {
                for dz in [-30.0, 0.0, 30.0] {
                    let x = p.x + dx;
                    let z = p.z + dz;
                    let g = terrain.height_at(x, z);
                    let f = terrain.forest_at(x, z);
                    let clearance = if f > 0.2 { 38.0 } else { 3.0 };
                    need = need.max(g + clearance);
                }
            }
            alts[i] = need;
        }
        // smooth altitudes a little so the camera doesn't bob
        for _ in 0..2 {
            let prev = alts.clone();
            for i in 0..n {
                let a = prev[(i + n - 1) % n];
                let b = prev[i];
                let d = prev[(i + 1) % n];
                alts[i] = b.max((a + b * 2.0 + d) * 0.25);
            }
        }
        for i in 0..n {
            pts[i].y = alts[i];
        }
        let mut seg_len = Vec::with_capacity(n);
        let mut total = 0.0;
        for i in 0..n {
            let l = pts[i].distance(pts[(i + 1) % n]);
            seg_len.push(l);
            total += l;
        }
        FlyPath { points: pts, seg_len, total }
    }

    fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    /// Position at arc length `s` (metres, wraps).
    pub fn at(&self, s: f32) -> Vec3 {
        let n = self.points.len();
        let mut s = s.rem_euclid(self.total);
        let mut i = 0;
        while s > self.seg_len[i] {
            s -= self.seg_len[i];
            i = (i + 1) % n;
        }
        let t = s / self.seg_len[i];
        let p0 = self.points[(i + n - 1) % n];
        let p1 = self.points[i];
        let p2 = self.points[(i + 1) % n];
        let p3 = self.points[(i + 2) % n];
        Self::catmull(p0, p1, p2, p3, t)
    }

    pub fn camera(&self, time: f32, terrain: &Terrain, fov_y: f32) -> Camera {
        let speed = 14.0;
        let s = time * speed;
        let mut pos = self.at(s);
        let ahead = self.at(s + 40.0);
        let ground = terrain.height_at(pos.x, pos.z);
        pos.y = pos.y.max(ground + 2.0);
        let mut cam = Camera { pos, yaw: 0.0, pitch: 0.0, fov_y };
        let mut dir = ahead - pos;
        dir.y = dir.y * 0.4 - dir.length() * 0.06;
        cam.look_dir(dir);
        cam
    }
}
