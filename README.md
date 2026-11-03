# ingenium

A small GPU-driven forest renderer in Rust + [wgpu](https://wgpu.rs): an 8 × 8 km procedural
valley with **~1.2 million trees**, grass, sun shadows, sky, clouds and fog. It runs on
Vulkan/DX12 (Windows, Linux) and Metal (macOS) from the same code.

![Golden hour over the valley](docs/golden_hour.jpg)

| Midday meadow | LOD debug view (`L`) |
| --- | --- |
| ![Midday meadow with birches](docs/midday.jpg) | ![Trees coloured by LOD](docs/lod_view.jpg) |

Inspired by Toni Macaroni's custom-engine forest showcase. This is the same family of ideas at
hobby scale: the CPU never touches individual trees, and far-away forest is a proxy with no
per-tree data.

## Run it

```bash
cargo run --release
```

On Windows, install Rust from <https://rustup.rs> first. Its installer also offers the Visual
Studio C++ build tools it needs. Then run the command above from this folder. The first build
takes a minute or two.

| Key | Action |
| --- | --- |
| `Tab` | toggle cinematic flythrough / free fly |
| click / `Esc` | grab / release the mouse |
| `W A S D`, `Space`/`E`, `Q`/`C` | move, up, down (`Shift` = fast, `Ctrl` = slow) |
| `1` `2` `3` `4` | morning · midday · golden hour · overcast |
| `F` | **freeze culling**, then fly away to see exactly what the GPU decided to draw |
| `L` | colour trees by LOD (red → yellow → green → blue = full detail → far hull) |
| `G` | toggle grass |
| `F11` | borderless fullscreen |
| `P` | print the camera as a `--cam` argument |

The window title shows fps, CPU and GPU time, trees drawn per LOD and triangles per frame.

Command-line options: `--novsync` (benchmark), `--msaa 1|4`, `--atmos 0-3`, `--size 1920x1080`,
and headless screenshots with `--capture out.png [--time 60] [--cam x,y,z,yaw,pitch] [--lod-tint]`.

## How it works

Every frame:

1. **Cull + LOD on the GPU** (`shaders/cull.wgsl`): one compute thread per tree (1.2M) does a
   frustum test, picks one of 4 LODs from distance, and appends the tree's index to that
   (species, LOD) bucket while incrementing the bucket's indirect-draw instance count. This runs
   three times: once for the camera and once per shadow cascade.
2. **Grass** (`shaders/grass_gen.wgsl`): a compute shader regenerates ~100k blades in a
   world-anchored grid around the camera, so blades don't swim.
3. **Shadows**: two cascades. The near one renders full leaf cards with alpha test, which
   gives the dappled light under the canopy. The far one only renders the cheap canopy hulls.
4. **Main pass** (HDR, 4× MSAA with alpha-to-coverage for leaves): CDLOD terrain, the far
   canopy shell, 12 indirect draws for all trees, grass, then the sky.
5. **Tonemap** (ACES) to the screen.

The CPU only writes a few uniforms and records about 40 draw calls, whatever the tree count.

Pieces worth reading:

| File | What |
| --- | --- |
| `src/treegen.rs` | procedural trees (oak-ish, birch, spruce). One branch skeleton per species, 4 LODs: full branches + ~1000 leaf cards → fewer cards → huge cards + hull → a 80-triangle "union of spheres" canopy hull |
| `src/leaftex.rs` | foliage textures painted in code; mipmaps keep alpha-test coverage so canopies don't thin out with distance |
| `src/terrain.rs`, `shaders/terrain.wgsl` | heightmap generated on the GPU; CDLOD quadtree patches that morph between levels (no cracks, no popping); the **canopy shell**: the terrain lifted by forest density and shaded as treetops, standing in for every tree beyond ~1.3 km |
| `src/forest.rs` | GPU scatter of the trees (density map → compact list), cull buffers, indirect draws |
| `src/renderer.rs` | frame orchestration, cascaded shadows, atmosphere presets, GPU timestamps |

## Where to take it next

- **Hi-Z occlusion culling**: build a depth pyramid and skip trees hidden behind hills and
  other trees. In a dense forest that removes most of the work.
- **Cluster LOD / virtual geometry**: split tree meshes into ~128-triangle clusters with a
  simplification hierarchy (see `meshoptimizer`), picked per cluster on the GPU instead of
  per tree.
- **Voxel far field**: voxelise each species into a small 3D texture and raymarch it for
  distant trees instead of drawing hulls.
- **GI**: screen-space GI plus a coarse voxel scene for off-screen bounce light.
