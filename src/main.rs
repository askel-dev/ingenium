//! ingenium: a small GPU-driven forest renderer.
//!
//!   cargo run --release                       window, cinematic flythrough
//!   cargo run --release -- --capture out.png  headless screenshot (+ --time, --size, --atmos, --cam)

mod camera;
mod forest;
mod gpu;
mod grass;
mod leaftex;
mod renderer;
mod terrain;
mod treegen;
mod util;

use camera::{Camera, FlyPath};
use glam::Vec3;
use renderer::Renderer;
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
    cam: Option<(Vec3, f32, f32)>,
    msaa: u32,
    vsync: bool,
    lod_tint: bool,
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
        cam: None,
        msaa: 4,
        vsync: true,
        lod_tint: false,
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
        "controls: Tab cinematic/fly · click to grab mouse, Esc to release · WASD move, Space/E up, Q/C down, Shift fast\n\
         \x20         1-4 time of day · F freeze culling · L LOD colours · G grass · F11 fullscreen · P print camera"
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
    r.settings.lod_tint = args.lod_tint;
    let path = FlyPath::new(&r.terrain);
    let cam = match args.cam {
        Some((p, yaw, pitch)) => Camera { pos: p, yaw, pitch, fov_y: FOV_Y },
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
    let mut gpu_ms = Vec::new();
    for f in 0..args.frames {
        r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + f as f32 / 60.0);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if f > args.frames / 2 && r.stats.gpu_ms > 0.0 {
            gpu_ms.push(r.stats.gpu_ms);
        }
    }
    // one more frame so the stats readback for the final image lands
    r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + args.frames as f32 / 60.0);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    r.render(&gpu.device, &gpu.queue, &view, &cam, args.time + args.frames as f32 / 60.0);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

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
    gpu_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = gpu_ms.get(gpu_ms.len() / 2).copied().unwrap_or(0.0);
    let s = &r.stats;
    println!(
        "captured {out} ({}x{}) cam=({:.1},{:.1},{:.1}) yaw={:.3} pitch={:.3}\n  gpu {:.2} ms (median) · {}",
        args.width, args.height, cam.pos.x, cam.pos.y, cam.pos.z, cam.yaw, cam.pitch, med, stats_line(&r)
    );
    println!("  passes: cull+grass {:.2} · shadows {:.2} · main {:.2} · post {:.2} ms", s.pass_ms[0], s.pass_ms[1], s.pass_ms[2], s.pass_ms[3]);
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
    path: FlyPath,
    cam: Camera,
    cinematic: bool,
    cine_time: f32,
    keys: HashSet<KeyCode>,
    grabbed: bool,
    start: Instant,
    last: Instant,
    title_timer: f32,
    frame_times: Vec<f32>,
    cpu_times: Vec<f32>,
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
        renderer.settings.lod_tint = self.args.lod_tint;
        let path = FlyPath::new(&renderer.terrain);
        let cam = path.camera(self.args.time, &renderer.terrain, FOV_Y);
        let now = Instant::now();
        self.state = Some(State {
            window,
            surface,
            config,
            gpu,
            renderer,
            path,
            cam,
            cinematic: true,
            cine_time: self.args.time,
            keys: HashSet::new(),
            grabbed: false,
            start: now,
            last: now,
            title_timer: 0.0,
            frame_times: Vec::new(),
            cpu_times: Vec::new(),
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
        if self.cinematic {
            self.cine_time += dt;
            self.cam = self.path.camera(self.cine_time, &self.renderer.terrain, FOV_Y);
            return;
        }
        let k = |c: KeyCode| self.keys.contains(&c);
        let mut speed = 25.0;
        if k(KeyCode::ShiftLeft) || k(KeyCode::ShiftRight) {
            speed *= 6.0;
        }
        if k(KeyCode::ControlLeft) {
            speed *= 0.2;
        }
        let f = self.cam.forward();
        let r = self.cam.right();
        let mut mv = Vec3::ZERO;
        if k(KeyCode::KeyW) { mv += f; }
        if k(KeyCode::KeyS) { mv -= f; }
        if k(KeyCode::KeyD) { mv += r; }
        if k(KeyCode::KeyA) { mv -= r; }
        if k(KeyCode::Space) || k(KeyCode::KeyE) { mv += Vec3::Y; }
        if k(KeyCode::KeyQ) || k(KeyCode::KeyC) { mv -= Vec3::Y; }
        if mv.length_squared() > 0.0 {
            self.cam.pos += mv.normalize() * speed * dt;
        }
        let w = self.renderer.terrain.world;
        self.cam.pos.x = self.cam.pos.x.clamp(1.0, w - 1.0);
        self.cam.pos.z = self.cam.pos.z.clamp(1.0, w - 1.0);
        let g = self.renderer.terrain.height_at(self.cam.pos.x, self.cam.pos.z);
        self.cam.pos.y = self.cam.pos.y.max(g + 1.7);
    }

    fn key(&mut self, code: KeyCode, el: &ActiveEventLoop) {
        let s = &mut self.renderer.settings;
        match code {
            KeyCode::Tab => {
                self.cinematic = !self.cinematic;
                println!("{}", if self.cinematic { "cinematic camera" } else { "free fly (click to grab the mouse)" });
            }
            KeyCode::Escape => {
                if self.grabbed {
                    self.set_grab(false);
                } else {
                    el.exit();
                }
            }
            KeyCode::Digit1 => s.atmos = 0,
            KeyCode::Digit2 => s.atmos = 1,
            KeyCode::Digit3 => s.atmos = 2,
            KeyCode::Digit4 => s.atmos = 3,
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
        let cpu_ms = cpu0.elapsed().as_secs_f32() * 1000.0;
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);

        self.frame_times.push(dt);
        self.cpu_times.push(cpu_ms);
        self.title_timer += dt;
        if self.title_timer > 0.5 {
            let n = self.frame_times.len() as f32;
            let fps = n / self.frame_times.iter().sum::<f32>();
            let cpu = self.cpu_times.iter().sum::<f32>() / n;
            let title = format!(
                "ingenium · {:.1} fps · cpu {:.2} ms · gpu {:.2} ms · {} · {}",
                fps,
                cpu,
                self.renderer.stats.gpu_ms,
                stats_line(&self.renderer),
                renderer::atmosphere(self.renderer.settings.atmos).name
            );
            self.window.set_title(&title);
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
                        st.cinematic = false;
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
                st.cam.yaw += delta.0 as f32 * 0.0022;
                st.cam.pitch = (st.cam.pitch - delta.1 as f32 * 0.0022).clamp(-1.5, 1.5);
            }
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(st) = &self.state {
            st.window.request_redraw();
        }
    }
}
