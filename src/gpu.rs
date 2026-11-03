//! Adapter / device setup.

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub timestamps: bool,
}

impl Gpu {
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Gpu {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
                apply_limit_buckets: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let info = adapter.get_info();
        println!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

        let available = adapter.features();
        let mut features = wgpu::Features::empty();
        let timestamps = available.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size.min(1 << 30),
            max_buffer_size: al.max_buffer_size.min(1 << 31),
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: features,
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        Gpu { adapter, device, queue, timestamps }
    }
}
