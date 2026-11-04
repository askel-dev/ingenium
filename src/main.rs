//! ingenium: a small GPU-driven forest renderer.
//!
//!   cargo run --release                       window, cinematic flythrough
//!   cargo run --release -- --capture out.png  headless screenshot (+ --time, --size, --atmos, --cam, --no-ao)

mod bloom;
mod camera;
mod clouds;
mod exposure;
mod forest;
mod gpu;
mod grass;
mod hiz;
mod hud;
mod leaftex;
mod player;
mod renderer;
mod sky;
mod ssao;
mod terrain;
mod treegen;
mod undergrowth;
mod util;

use camera::{Camera, FlyPath};
use glam::Vec3;
use player::{Input, Player, Trunks};
use renderer::{GpuTiming, Renderer};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const FOV_Y: f32 = 0.95;

pub fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(rgba).expect("png data");
}

#[derive(Clone)]
struct Args {
    capture: Option<String>,
    time: f32,
    width: u32,
    height: u32,
    frames: u32,
    atmos: usize,
    sun_shift: f32,
    cam: Option<(Vec3, f32, f32)>,
    msaa: u32,
    vsync: bool,
    lod_tint: bool,
    ssao: bool,
    hud: bool,
    /// Render scale; None in a window means one scene pixel per point (half on Retina).
    scale: Option<f32>,
    prepass: bool,
    occlusion: bool,
    leaf_png: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        capture: None,
        time: 20.0,
        width: 1600,
        height: 900,
        frames: 30,
        atmos: 1,
        sun_shift: 0.0,
        cam: None,
        msaa: 4,
        vsync: true,
        lod_tint: false,
        ssao: true,
        hud: false,
        scale: None,
        prepass: true,
        occlusion: true,
        leaf_png: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let next = |i: usize| v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--capture" => { a.capture = Some(next(i)); i += 1; }
            "--time" => { a.time = next(i).parse().unwrap_or(a.time); i += 1; }
            "--frames" => { a.frames = next(i).parse().unwrap_or(a.frames); i += 1; }
            "--atmos" => { a.atmos = next(i).parse().unwrap_or(a.atmos); i += 1; }
            "--sun-shift" => { a.sun_shift = next(i).parse().unwrap_or(a.sun_shift); i += 1; }
            "--msaa" => { a.msaa = next(i).parse().unwrap_or(a.msaa); i += 1; }
            "--size" => {
                let s = next(i);
                if let Some((w, h)) = s.split_once('x') {
                    a.width = w.parse().unwrap_or(a.width);
                    a.height = h.parse().unwrap_or(a.height);
                }
                i += 1;
            }
            "--cam" => {
                let n: Vec<f32> = next(i).split(',').filter_map(|x| x.parse().ok()).collect();
                if n.len() == 5 {
                    a.cam = Some((Vec3::new(n[0], n[1], n[2]), n[3], n[4]));
                }
                i += 1;
            }
            "--novsync" => a.vsync = false,
            "--lod-tint" => a.lod_tint = true,
            "--no-ao" => a.ssao = false,
            "--hud" => a.hud = true,
            "--no-prepass" => a.prepass = false,
            "--no-occlusion" => a.occlusion = false,
            "--scale" => { a.scale = next(i).parse().ok(); i += 1; }
            "--leaf-png" => { a.leaf_png = Some(next(i)); i += 1; }
            other => eprintln!("unknown argument: {other}"),
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    if let Some(p) = &args.leaf_png {
        leaftex::debug_png(p);
        println!("wrote {p}");
        return;
    }
    if args.capture.is_some() {
        capture(&args);
        return;
    }
    println!(
        "controls: Tab cinematic/player · click to grab mouse, Esc to release · V walk/fly\n\
         \x20 walk:    WASD move, Space jump, Shift sprint, Ctrl/C crouch\n\
         \x20 fly:     WASD move, Space/E up, Q/C down, Shift fast, Ctrl slow\n\
         \x20 render:  1-4 time of day · [ ] move the sun · F freeze culling · L LOD colours · G grass · H stats overlay · -/= render scale · F11 fullscreen · P print camera"
    );
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { args, state: None };
    event_loop.run_app(&mut app).expect("run");
}

fn capture(args: &Args) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let gpu = pollster::block_on(gpu::Gpu::new(&instance, None));
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&gpu.device, &gpu.queue, gpu.timestamps, args.width, args.height, fmt, args.msaa);
    r.settings.atmos = args.atmos;
    r.settings.sun_shift = args.sun_shift;
    r.settings.lod_tint = args.lod_tint;
    r.settings.ssao = args.ssao;
    r.settings.prepass = args.prepass;
    r.settings.occlusion = args.occlusion;
    if let Some(s) = args.scale {
        r.set_scale(&gpu.device, s);
    }
    let path = FlyPath::new(&r.terrain);
    let cam = match args.cam {
        // a negative height means that far above the ground
        Some((mut p, yaw, pitch)) => {
            if p.y < 0.0 {
                p.y = r.terrain.ground_at(p.x, p.z) - p.y;
            }
            Camera { pos: p, yaw, pitch, fov_y: FOV_Y }
        }
        None => path.camera(args.time, &r.terrain, FOV_Y),
    };
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture"),
        size: wgpu::Extent3d { width: args.width, height: args.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    // the first half of the frames warms up (cloud reprojection, caches); the rest are timed
    let warmup = (args.frames / 2) as u64;
    let (mut cpu_ms, mut wall_ms) = (Vec::new(), Vec::new());
    for f in 0..args.frames {
        let t0 = Instant::now();
        r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + f as f32 / 60.0);
        let t1 = Instant::now();
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if f as u64 >= warmup {
            cpu_ms.push((t1 - t0).as_secs_f32() * 1000.0);
            wall_ms.push(t0.elapsed().as_secs_f32() * 1000.0);
        }
    }
    // two more frames so the stats and timestamps of the last frames land
    r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + args.frames as f32 / 60.0);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + args.frames as f32 / 60.0);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let timings: Vec<GpuTiming> = r.take_timings().into_iter().filter(|t| t.frame > warmup).collect();
    if args.hud {
        // a capture waits for the GPU after every frame, so its fps is unthrottled (no vsync)
        let mut hud = hud::Hud::new(&gpu.device, fmt);
        let fps = 1000.0 / median(wall_ms.into_iter()).unwrap_or(f32::INFINITY);
        let cpu = median(cpu_ms.into_iter()).unwrap_or(0.0);
        let gpu_ms = median(timings.iter().map(|t| t.total_ms)).unwrap_or(0.0);
        hud.set_text(&format!("{fps:.1} fps   cpu {cpu:.2} ms   gpu {gpu_ms:.2} ms"));
        hud.draw(&gpu.device, &gpu.queue, &view, args.width, args.height, 1.0);
    }

    let row = (args.width * 4).div_ceil(256) * 256;
    let buf = util::buffer(&gpu.device, "capture rb", (row * args.height) as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(args.height) },
        },
        wgpu::Extent3d { width: args.width, height: args.height, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    let data = util::read_buffer_blocking(&gpu.device, &buf, (row * args.height) as u64);
    let mut rgba = Vec::with_capacity((args.width * args.height * 4) as usize);
    for y in 0..args.height {
        let s = (y * row) as usize;
        rgba.extend_from_slice(&data[s..s + (args.width * 4) as usize]);
    }
    let out = args.capture.as_ref().unwrap();
    write_png(out, args.width, args.height, &rgba);

    let ms = |v: Option<f32>| v.map_or("n/a".to_string(), |x| format!("{x:.2}"));
    println!(
        "captured {out} ({}x{}) cam=({:.1},{:.1},{:.1}) yaw={:.3} pitch={:.3}\n  gpu {} ms (median of {} frames, whole frame) · {} ms through the main pass · {}",
        args.width,
        args.height,
        cam.pos.x,
        cam.pos.y,
        cam.pos.z,
        cam.yaw,
        cam.pitch,
        ms(median(timings.iter().map(|t| t.total_ms))),
        timings.len(),
        ms(median(timings.iter().filter_map(|t| t.through_main_ms))),
        stats_line(&r)
    );
    // Per pass: the median over the frames whose timestamps all came back in order (their
    // passes add up to the frame). Medians of the passes need not add up to the median frame.
    let ordered: Vec<&GpuTiming> = timings.iter().filter(|t| t.ordered).collect();
    let (split, note) = match ordered.len() {
        0 if !timings.is_empty() => (timings.iter().collect(), " (approximate: no frame had its timestamps in order)".to_string()),
        n if n < timings.len() => (ordered, format!(" ({} of {} frames left out: timestamps out of order)", timings.len() - n, timings.len())),
        _ => (ordered, String::new()),
    };
    let passes: Vec<String> = renderer::GPU_SECTIONS
        .iter()
        .enumerate()
        .map(|(i, name)| format!("{name} {}", ms(median(split.iter().filter_map(|t| t.sections[i])))))
        .collect();
    println!("  passes (median ms): {}{note} · ground light {:.2}", passes.join(" · "), r.stats.ground_light);
    let p = r.stats.patches;
    println!("  terrain patches: {} picked by LOD · drawn by the camera {} · near cascade {} · far cascade {}", p[0], p[1], p[2], p[3]);
}

/// The median (upper, for an even count), or None if there are no values.
fn median(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut v: Vec<f32> = values.collect();
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied()
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn stats_line(r: &Renderer) -> String {
    let s = &r.stats;
    let mut lod = [0u64; 4];
    for (d, n) in s.visible.iter().enumerate() {
        lod[d % 4] += *n as u64;
    }
    format!(
        "trees {} · drawn L0 {} L1 {} L2 {} L3 {} · grass {} · {} tris",
        fmt_count(r.forest.tree_count as u64),
        fmt_count(lod[0]),
        fmt_count(lod[1]),
        fmt_count(lod[2]),
        fmt_count(lod[3]),
        fmt_count(s.grass as u64),
        fmt_count(s.tris)
    )
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: gpu::Gpu,
    renderer: Renderer,
    hud: hud::Hud,
    path: FlyPath,
    cam: Camera,
    player: Player,
    trunks: Trunks,
    cinematic: bool,
    cine_time: f32,
    keys: HashSet<KeyCode>,
    grabbed: bool,
    start: Instant,
    last: Instant,
    title_timer: f32,
    frame_times: Vec<f32>,
    cpu_times: Vec<f32>,
    gpu_times: Vec<f32>,
}

struct App {
    args: Args,
    state: Option<State>,
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("ingenium")
            .with_inner_size(LogicalSize::new(self.args.width, self.args.height));
        let window = Arc::new(el.create_window(attrs).expect("window"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = pollster::block_on(gpu::Gpu::new(&instance, Some(&surface)));
        let size = window.inner_size();
        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let mut config = surface.get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1)).expect("surface config");
        config.format = format;
        config.present_mode = if self.args.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        config.desired_maximum_frame_latency = 2;
        surface.configure(&gpu.device, &config);
        let mut renderer = Renderer::new(&gpu.device, &gpu.queue, gpu.timestamps, config.width, config.height, format, self.args.msaa);
        renderer.settings.atmos = self.args.atmos;
        renderer.settings.sun_shift = self.args.sun_shift;
        renderer.settings.lod_tint = self.args.lod_tint;
        renderer.settings.ssao = self.args.ssao;
        renderer.settings.prepass = self.args.prepass;
        renderer.settings.occlusion = self.args.occlusion;
        renderer.set_scale(&gpu.device, self.args.scale.unwrap_or(1.0 / window.scale_factor() as f32));
        let hud = hud::Hud::new(&gpu.device, format);
        let path = FlyPath::new(&renderer.terrain);
        let cam = path.camera(self.args.time, &renderer.terrain, FOV_Y);
        let trunks = Trunks::new(&renderer.forest.read_trees(&gpu.device, &gpu.queue), renderer.terrain.world);
        let now = Instant::now();
        self.state = Some(State {
            window,
            surface,
            config,
            gpu,
            renderer,
            hud,
            path,
            cam,
            player: Player::new(),
            trunks,
            cinematic: true,
            cine_time: self.args.time,
            keys: HashSet::new(),
            grabbed: false,
            start: now,
            last: now,
            title_timer: 0.0,
            frame_times: Vec::new(),
            cpu_times: Vec::new(),
            gpu_times: Vec::new(),
        });
    }
}

impl State {
    fn set_grab(&mut self, grab: bool) {
        if grab {
            let ok = self.window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            if ok.is_ok() {
                self.window.set_cursor_visible(false);
                self.grabbed = true;
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
            self.grabbed = false;
        }
    }

    fn update(&mut self, dt: f32) {
        // hold [ / ] to move the sun down / up through the day, 8 degrees a second
        let sun = self.keys.contains(&KeyCode::BracketRight) as i32 - self.keys.contains(&KeyCode::BracketLeft) as i32;
        if sun != 0 {
            let s = &mut self.renderer.settings;
            s.sun_shift = (s.sun_shift + sun as f32 * 8.0 * dt).clamp(-60.0, 60.0);
        }
        if self.cinematic {
            self.cine_time += dt;
            self.cam = self.path.camera(self.cine_time, &self.renderer.terrain, FOV_Y);
            return;
        }
        let k = |c: KeyCode| self.keys.contains(&c);
        let axis = |pos: bool, neg: bool| pos as i32 as f32 - neg as i32 as f32;
        let input = Input {
            forward: axis(k(KeyCode::KeyW), k(KeyCode::KeyS)),
            right: axis(k(KeyCode::KeyD), k(KeyCode::KeyA)),
            up: axis(k(KeyCode::Space) || k(KeyCode::KeyE), k(KeyCode::KeyQ) || k(KeyCode::KeyC)),
            sprint: k(KeyCode::ShiftLeft) || k(KeyCode::ShiftRight),
            crouch: k(KeyCode::ControlLeft) || k(KeyCode::ControlRight) || (!self.player.fly && k(KeyCode::KeyC)),
        };
        self.player.update(&input, dt, &self.renderer.terrain, &self.trunks);
        self.cam = self.player.camera(FOV_Y);
    }

    fn leave_cinematic(&mut self) {
        self.cinematic = false;
        self.player.take_over(&self.cam, &self.renderer.terrain, &self.trunks);
        println!("{} (click to grab the mouse, V to switch)", self.mode_name());
    }

    fn mode_name(&self) -> &'static str {
        match (self.cinematic, self.player.fly) {
            (true, _) => "cinematic",
            (false, false) => "walk",
            (false, true) => "fly",
        }
    }

    fn key(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        let s = &mut self.renderer.settings;
        match code {
            KeyCode::Tab => {
                if self.cinematic {
                    self.leave_cinematic();
                } else {
                    self.cinematic = true;
                    println!("cinematic");
                }
            }
            KeyCode::KeyV if !self.cinematic => {
                self.player.set_fly(!self.player.fly);
                println!("{}", self.mode_name());
            }
            KeyCode::Space if !self.cinematic && !self.player.fly => self.player.jump(),
            KeyCode::Escape => {
                if self.grabbed {
                    self.set_grab(false);
                } else {
                    el.exit();
                }
            }
            KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4 => {
                s.atmos = match code {
                    KeyCode::Digit1 => 0,
                    KeyCode::Digit2 => 1,
                    KeyCode::Digit3 => 2,
                    _ => 3,
                };
                s.sun_shift = 0.0;
            }
            KeyCode::KeyF => {
                s.freeze = !s.freeze;
                println!("culling {}", if s.freeze { "FROZEN: fly away to see what the GPU kept" } else { "live" });
            }
            KeyCode::KeyL => s.lod_tint = !s.lod_tint,
            KeyCode::F11 => {
                let full = self.window.fullscreen().is_some();
                self.window.set_fullscreen(if full { None } else { Some(winit::window::Fullscreen::Borderless(None)) });
            }
            KeyCode::KeyG => s.grass = !s.grass,
            KeyCode::KeyO => s.ssao = !s.ssao,
            KeyCode::KeyH => self.hud.visible = !self.hud.visible,
            KeyCode::Minus | KeyCode::Equal => {
                const STEPS: [f32; 6] = [0.5, 0.6, 0.67, 0.75, 0.85, 1.0];
                let now = STEPS.iter().position(|&x| x >= self.renderer.scale - 0.01).unwrap_or(STEPS.len() - 1);
                let next = if code == KeyCode::Minus { now.saturating_sub(1) } else { (now + 1).min(STEPS.len() - 1) };
                self.renderer.set_scale(&self.gpu.device, STEPS[next]);
                println!("render scale {:.2}: {}x{}", self.renderer.scale, self.renderer.width, self.renderer.height);
            }
            KeyCode::KeyP => {
                let c = self.cam;
                println!("--cam {:.1},{:.1},{:.1},{:.3},{:.3}", c.pos.x, c.pos.y, c.pos.z, c.yaw, c.pitch);
            }
            _ => {}
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        self.update(dt);
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            _ => return,
        };
        // CPU cost of building the frame (excludes waiting on vsync for the swapchain image)
        let cpu0 = Instant::now();
        let view = frame.texture.create_view(&Default::default());
        let time = (now - self.start).as_secs_f32();
        self.renderer.render(&self.gpu.device, &self.gpu.queue, &view, &self.cam, time);
        let (w, h) = (self.config.width, self.config.height);
        self.hud.draw(&self.gpu.device, &self.gpu.queue, &view, w, h, self.window.scale_factor());
        let cpu_ms = cpu0.elapsed().as_secs_f32() * 1000.0;
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);

        self.frame_times.push(dt);
        self.cpu_times.push(cpu_ms);
        self.gpu_times.extend(self.renderer.take_timings().iter().map(|t| t.total_ms));
        self.title_timer += dt;
        if self.title_timer > 0.5 {
            let n = self.frame_times.len() as f32;
            let fps = n / self.frame_times.iter().sum::<f32>();
            let cpu = self.cpu_times.iter().sum::<f32>() / n;
            let gpu = median(self.gpu_times.drain(..)).unwrap_or(self.renderer.stats.gpu_ms);
            let title = format!(
                "ingenium · {} · {:.1} fps · cpu {:.2} ms · gpu {:.2} ms · {} · {}",
                self.mode_name(),
                fps,
                cpu,
                gpu,
                stats_line(&self.renderer),
                renderer::atmosphere(self.renderer.settings.atmos).name
            );
            self.window.set_title(&title);
            let (rw, rh) = (self.renderer.width, self.renderer.height);
            self.hud.set_text(&format!("{fps:.1} fps   cpu {cpu:.2} ms   gpu {gpu:.2} ms   {rw}x{rh}"));
            self.frame_times.clear();
            self.cpu_times.clear();
            self.title_timer = 0.0;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_none() {
            self.init(el);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                if width > 0 && height > 0 {
                    st.config.width = width;
                    st.config.height = height;
                    st.surface.configure(&st.gpu.device, &st.config);
                    st.renderer.resize(&st.gpu.device, width, height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if !event.repeat {
                                st.key(code, el);
                            }
                            st.keys.insert(code);
                        }
                        ElementState::Released => {
                            st.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !st.grabbed {
                    if st.cinematic {
                        st.leave_cinematic();
                    }
                    st.set_grab(true);
                }
            }
            WindowEvent::Focused(false) => {
                st.keys.clear();
                st.set_grab(false);
            }
            WindowEvent::RedrawRequested => st.frame(),
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        let Some(st) = self.state.as_mut() else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if st.grabbed && !st.cinematic {
                st.player.look(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(st) = &self.state {
            st.window.request_redraw();
        }
    }
}
