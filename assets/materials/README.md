# Scanned materials

PBR texture sets from [Poly Haven](https://polyhaven.com), all **CC0** (public domain,
<https://polyhaven.com/license>). Each folder holds the set's diffuse (`diff.jpg`) and OpenGL
normal map (`nor_gl.jpg`) at 2K, and its packed AO / roughness / metal (`arm.jpg`) and
displacement (`disp.jpg`) at 1K. `src/materials.rs` decodes them at startup.

| Layer | Used for | Asset id | Covers | Source |
| --- | --- | --- | --- | --- |
| 0 | broadleaf bark | `bark_willow_02` | 2.15 m | <https://polyhaven.com/a/bark_willow_02> |
| 1 | birch bark (marks painted on top) | `japanese_sycamore` | 1.1 × 2.2 m | <https://polyhaven.com/a/japanese_sycamore> |
| 2 | spruce bark | `pine_bark` | 2.0 m | <https://polyhaven.com/a/pine_bark> |
| 3 | forest floor: leaf litter | `forest_leaves_04` | 1.5 m | <https://polyhaven.com/a/forest_leaves_04> |
| 4 | forest floor: moss over soil | `brown_mud_leaves_01` | 1.3 m | <https://polyhaven.com/a/brown_mud_leaves_01> |
| 5 | meadow ground | `forrest_ground_01` | 2.0 m | <https://polyhaven.com/a/forrest_ground_01> |
| 6 | bare soil | `forest_ground_04` | 3.15 m | <https://polyhaven.com/a/forest_ground_04> |
| 7 | rock | `rock_face_03` | 2.7 m | <https://polyhaven.com/a/rock_face_03> |

Files were fetched from `https://api.polyhaven.com/files/<id>` (the `jpg` entries). The order of
the layers is fixed: the shaders index them by number.
