# Scanned foliage

Texture atlases of plant models from [Poly Haven](https://polyhaven.com), all **CC0** (public
domain, <https://polyhaven.com/license>). Only the 2K JPEG texture maps were downloaded (no
meshes), from `https://api.polyhaven.com/files/<id>`. Each folder holds `diff.jpg` (albedo),
`alpha.jpg` (coverage), `nor_gl.jpg` (OpenGL normal map) and `arm.jpg` (AO / roughness / metal);
the periwinkle also has `translucency.jpg`. `src/foliage.rs` cuts single leaves, fronds, sprays,
blades and flower heads out of them at startup and composes the atlases the renderer uses.

| Asset id | Used for | Source |
| --- | --- | --- |
| `shrub_01` | serrated leaves: broadleaf and birch twigs, shrubs | <https://polyhaven.com/a/shrub_01> |
| `shrub_03` | heart-shaped (lime-like) leaves: broadleaf twigs | <https://polyhaven.com/a/shrub_03> |
| `fir_tree_01` | conifer sprays (`twig_*` maps): spruce branches | <https://polyhaven.com/a/fir_tree_01> |
| `fern_02` | fern fronds | <https://polyhaven.com/a/fern_02> |
| `periwinkle_plant` | ground-cover leaves; flower recoloured to violet clover | <https://polyhaven.com/a/periwinkle_plant> |
| `celandine_01` | yellow flower heads (buttercups; recoloured to daisies) | <https://polyhaven.com/a/celandine_01> |
| `grass_medium_02` | single grass blades, green and dry | <https://polyhaven.com/a/grass_medium_02> |
| `grass_medium_01` | grass tufts and seed heads | <https://polyhaven.com/a/grass_medium_01> |

Poly Haven has no oak, birch or spruce leaf scans, and no daisy or clover: the nearest leaves
and flowers stand in for them (see `src/foliage.rs`).
