# ingenium

A small GPU-driven forest renderer in Rust + [wgpu](https://wgpu.rs): an 8 × 8 km procedural
valley with **~980,000 trees and shrubs** in 20 shapes, ferns and shrubs under them, grass, sun shadows, volumetric
clouds, sky and fog. It runs on
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
| `Tab` | toggle cinematic flythrough / player |
| click / `Esc` | grab / release the mouse |
| `V` | switch the player between walking and free fly |
| walking: `W A S D`, `Space`, `Shift`, `Ctrl`/`C` | move, jump, sprint, crouch |
| flying: `W A S D`, `Space`/`E`, `Q`/`C` | move, up, down (`Shift` = fast, `Ctrl` = slow) |
| `1` `2` `3` `4` | morning · midday · golden hour · overcast |
| hold `[` / `]` | move the sun down / up through the day |
| `F` | **freeze culling**, then fly away to see exactly what the GPU decided to draw |
| `L` | colour trees by LOD (red → yellow → green → blue = full detail → far impostor) |
| `G` | toggle grass and undergrowth |
| `O` | toggle screen-space ambient occlusion |
| `H` | toggle the stats overlay |
| `-` / `=` | lower / raise the render scale |
| `F11` | borderless fullscreen |
| `P` | print the camera as a `--cam` argument |

The overlay in the top-right corner shows fps, CPU time and the median GPU time of the whole frame,
updated twice a second; it is left out of captures. The window title shows fps, CPU time, the median GPU time of the whole frame, trees drawn per LOD
and triangles per frame.

Command-line options: `--novsync` (benchmark), `--msaa 1|4`, `--scale 0.5` (render the scene at that fraction of
the output size and upscale it; a window defaults to one scene pixel per point, so half size on Retina, a capture to
full size), `--no-prepass` (shade trees without the depth prepass, to compare), `--no-occlusion` (draw the distant trees without
testing them against the near ones, to compare), `--atmos 0-3`, `--sun-shift <degrees>`, `--size 1920x1080`,
and headless screenshots with `--capture out.png [--time 60] [--cam x,y,z,yaw,pitch] [--lod-tint] [--no-ao] [--hud] [--frames 30]`
(a negative camera height in `--cam` means that many metres above the ground; `--impostor-png atlas.png` also writes
the impostor atlas). A capture renders
`--frames` frames of the same view and prints the median GPU time over the second half of them:
the whole frame, the span through the main pass, each pass (cull, grass and occluders, sky + clouds,
shadows, main, SSAO, bloom, post) and how many terrain patches each view drew.

## How it works

Every frame:

1. **Cull + LOD on the GPU** (`shaders/cull.wgsl`): one compute thread per tree (980k) does a
   frustum test, picks one of 4 LODs from distance, and appends the tree's index to its bucket
   while incrementing the bucket's indirect-draw instance count: one bucket per tree shape for
   each of the two mesh LODs, and one per impostor LOD for all shapes. This runs three times:
   once for the camera and once per shadow cascade.
   **Occlusion culling** (`shaders/hiz.wgsl`): the camera's pass appends only the two near
   LODs. They and the terrain lay down their depth first, a depth pyramid is built from it, and
   a second pass appends the distant LODs whose screen box isn't entirely behind it. Under the
   canopy the near trees hide almost every distant one, yet those were most of the triangles:
   about 8 ms of a 26 ms frame. Nothing comes from the last frame, so a tree is never drawn late,
   and a tree is only culled if every sample it could cover is nearer. In open country it costs
   about half a millisecond.
2. **Grass and undergrowth** (`shaders/grass_gen.wgsl`, `shaders/undergrowth.wgsl`): compute
   shaders regenerate ~100k grass blades in the meadows, and ferns, shrubs and ground cover on
   the forest floor, in world-anchored grids around the camera, so nothing swims. The plants
   are written in the trees' record layout, so the tree pipeline draws them: one indirect
   draw per kind.
3. **Clouds** (`shaders/clouds.wgsl`): a cloud layer 1.2–4.4 km up, wrapped around a curved
   planet, is raymarched through tiling Perlin-Worley and Worley noise volumes and a weather map
   (coverage, cloud type), all baked on the GPU at startup. Light comes from a short march towards
   the sun, a cheap multiple-scattering approximation and occluded skylight. The buffer is
   ½–¼ of the screen resolution. Each frame marches one pixel per 2×2 block and reprojects the
   other three from the previous frame, so there is at most one ray per 16 screen pixels.
   The same density also builds a 512² **cloud shadow map** over 32 km: from points on the
   cloud base, a march towards the sun gives the direct sunlight that gets through, and the
   diffuse light the cloud scatters down. The map wraps around in world space, so it survives
   camera movement and only a quarter of it (in 8×8 tiles) is refreshed per frame.
   Before the clouds, the **atmosphere** (`shaders/atmosphere.wgsl`, Hillaire 2020) rebuilds a
   192×108 sky-view table and the sun and ambient light from small transmittance and
   multiple-scattering tables. A 64×64×32 **aerial perspective** volume over the screen
   (`shaders/aerial.wgsl`) then marches each view ray through the same air plus a ground haze
   layer, in the cloud shadows, and stores the haze in front of each depth. The sky, the sun's
   colour, the ambient light, the haze on distant ridges and the light on the clouds all come
   from it, so a ridge fades into exactly the sky behind it and a preset is mostly a sun angle
   and a haze level.
4. **Shadows**: two cascades. The near one renders full leaf cards with alpha test, which
   gives the dappled light under the canopy. The far one renders every tree as its impostor,
   cut out by its coverage, so trees out to 260 m cast tree-shaped, dappled shadows too.
   Each cascade draws only the terrain patches inside its box (see below), and only the trees
   that can shade the part of the view it covers: the outline of that part as the sun sees it,
   swept towards the sun, plus a plane past its far side. The box itself is a square around a
   sphere, so most trees in it would cast their shadows off screen or behind the camera; over
   the forest this leaves out about 80% of the trees in the near cascade and 70% in the far one.
5. **Main pass** (HDR, 4× MSAA with alpha-to-coverage for leaves and impostors): CDLOD
   terrain, the far canopy shell, 42 indirect draws for all trees, grass, then the sky with
   the clouds composited over it. Beyond ~140 m a tree is an **impostor**: a quad facing the
   camera that blends the four views of the tree baked at startup nearest to the direction it
   is seen from (see `src/impostor.rs`). Each texel's depth puts its leaf back in place, so an
   impostor is lit like the mesh: the same shadows, haze and light through the leaves.
   A **crown shades itself**: each leaf, of a mesh or an impostor, is darkened by how far the
   sun's rays travel through its tree's crown (an ellipsoid per shape) to reach it. The inside
   of a crown and its side away from the sun come out dark and its sunlit clumps bright, at
   any distance, without a shadow map. Within the near cascade the leaves' real shadows do it
   instead. Where trees block the sky, the forest floor gets green **light through
   the canopy** instead (a share of the sun and sky falling on the leaves above), and leaves
   let light through from their other side, so a canopy glows seen from below.
   Ambient light everywhere is scaled by a **sky visibility** map baked
   once at startup (`shaders/sky_vis.wgsl`): from every heightmap texel, 16 directions are
   marched ~2 km out to find the horizon, with the forest counted as a 15 m canopy that blocks
   part of the sky. Valleys, the foot of slopes, clearings and the forest floor get less sky
   light and more light bounced off the surrounding ground.
6. **SSAO** (`shaders/ssao.wgsl`, half resolution, ~0.4 ms at 1080p): contact shadows where
   trunks meet the ground, between grass blades and under low branches. Lighting is forward,
   so there is no separate ambient term to darken. Instead the pass reconstructs each pixel's
   position and normal from depth, re-evaluates the sun and cloud shadow there, and occludes
   only the share of the light that was ambient. Sunlit ground next to a trunk keeps its
   sunlight.
7. **Bloom** (`shaders/bloom.wgsl`): a 6-level mip chain filtered down and added back up, so
   the sun disk (drawn at its real 0.53°) and the bright cloud edges glow.
8. **Auto exposure and tonemap** (`shaders/exposure.wgsl`, ACES): the log-average brightness
   of a 48 × 48 grid of the frame, eased over about half a second, opens the exposure by up
   to ~1.6 stops when the view is darker than open ground, the way a camera does under trees.

The CPU only writes a few uniforms and records about 40 draw calls, whatever the tree count.

Pieces worth reading:

| File | What |
| --- | --- |
| `src/treegen.rs` | procedural trees (oak-ish, birch, spruce) and forest-floor plants (fern, shrub, ground cover). 20 shapes, each from its own branch skeleton: per species 4 forest-grown (bare trunks up to a crown that starts halfway up, crowns that are a shell of leaves around a mostly open inside, so sunflecks reach the floor), 2 open-grown for edges and meadows (limbs low on the trunk, full crowns), and 2 hazel-like edge shrubs. A tree's seed picks its shape. Two mesh LODs per shape: full branches + leaf cards → limbs + fewer, larger cards. Leaf normals lean out from the clump of leaves they grow in as well as from the crown, so a crown is lit as a heap of clumps rather than one ball |
| `src/impostor.rs`, `shaders/impostor_bake.wgsl` | the impostors: every shape's full-detail mesh rendered at startup from 8 × 8 directions (a hemi-octahedral grid with its rings pulled towards the horizon, where trees are seen from) into an atlas layer. Colour and coverage in one texture; normal, depth and ambient occlusion in the other; all premultiplied by coverage, so the mips are plain averages and nothing bleeds in from around the tree (~170 MB for the 20 shapes) |
| `src/undergrowth.rs`, `shaders/undergrowth.wgsl` | the forest-floor plants: placed around the camera each frame in drifts of fern, mats of ground cover and scattered shrubs, drawn with the tree pipeline |
| `src/exposure.rs`, `shaders/exposure.wgsl` | auto exposure metering |
| `shaders/tree.wgsl`, `shaders/tree_common.wgsl` | tree shading: per-tree brightness and hue (blue-green to yellow-green, the odd early-turning tree) plus a shift shared by neighbouring trees, translucency, a soft waxy highlight and sky sheen on the leaves, and the crown's self-shadowing (`crown_light`). The impostors: picking and blending the four nearest views, and a coverage that grows with distance as the gaps between the leaves close up |
| `src/leaftex.rs` | foliage textures painted in code (broadleaf, birch, spruce, fern frond); mipmaps keep alpha-test coverage so canopies don't thin out with distance |
| `src/terrain.rs`, `shaders/terrain.wgsl` | heightmap generated on the GPU; CDLOD quadtree patches that morph between levels (no cracks, no popping), picked once per frame by distance, then culled against the camera and each shadow cascade using min/max height bounds per quadtree node; the **canopy shell**: the terrain lifted by forest density and shaded as treetops, standing in for every tree beyond ~1.3 km |
| `shaders/terrain.wgsl`, `globals.wgsl` (`meadow_tint`) | the ground material: dry and lush stretches (drier on raised, open or steep ground, lusher in hollows), clumps, drifts of buttercups, daisies or clover painted as dots that fade to their average colour with distance, bare soil on slopes and worn patches, and sward relief at four scales that fades with the pixel's footprint. A grass sward isn't a flat surface, so the ground darkens when seen from above (gaps between blades), brightens at grazing angles, has a hot spot with the sun behind the viewer and glows against the sun. The grass blades take their colour from the same function, so they blend into the ground. |
| `src/clouds.rs`, `shaders/clouds.wgsl` | volumetric clouds: noise baking, the weather-driven density, lighting and the temporal reprojection; the cloud shadow map; each time-of-day preset sets the layer height, coverage and cloud types |
| `src/sky.rs`, `shaders/atmosphere.wgsl`, `shaders/aerial.wgsl` | the atmosphere tables, the sun and ambient light, and the aerial perspective volume |
| `src/forest.rs`, `shaders/scatter.wgsl` | GPU scatter of the trees into a compact list: forest-grown trees in stands of different ages (young stands, trees held back under the canopy, emergents above it) with gaps where an old tree fell; along forest edges, found by sampling for open ground on rings around each tree, open-grown trees and dense shrubs that step down to the meadow; in the open, rare groups ringed by shrubs and the odd lone tree. Cull buffers and indirect draws |
| `shaders/terrain.wgsl` (`litter`) | the forest floor: three layers of fallen leaves, twigs and moss, with the leaves curling for the sunflecks to catch |
| `src/renderer.rs` | frame orchestration, cascaded shadows, atmosphere presets, GPU timestamps |
| `src/hiz.rs`, `shaders/cull.wgsl` (`occluded`) | the depth pyramid, and the test of a tree's projected bounding sphere against it |
| `src/player.rs` | first-person controller: stands on exactly the ground that is drawn (a CPU copy of the terrain's detail noise), can't climb slopes over ~45° and slides off them, and walks around trunks (and through shrubs). The tree list is read back from the GPU once into a one-tree-per-cell grid, the only place the CPU looks at individual trees |

## Where to take it next

- **Occlusion culling for the near trees**: they are the occluders, so they are never tested,
  and under the canopy they are now most of the frame. Testing them against last frame's depth
  pyramid first, and only the rest against this frame's, would cull the near trees hidden behind
  trunks and hills too.
- **Cluster LOD / virtual geometry**: split tree meshes into ~128-triangle clusters with a
  simplification hierarchy (see `meshoptimizer`), picked per cluster on the GPU instead of
  per tree.
- **Voxel far field**: voxelise each tree shape into a small 3D texture and raymarch it for
  distant trees instead of drawing impostors.
- **GI**: screen-space GI plus a coarse voxel scene for off-screen bounce light.
