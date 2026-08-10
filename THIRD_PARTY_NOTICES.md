# Third-Party Notices

HIVE-3D is a non-commercial academic research release containing code from multiple upstream projects. The root `LICENSE` applies only to original HIVE-3D contributions and does not replace or narrow any third-party license.

## Bundled source code

| Component | Bundled paths | Governing terms |
| --- | --- | --- |
| Microsoft TRELLIS | `trellis/` and `extensions/vox2seq/`, except the separately identified files below | [MIT License](LICENSES/TRELLIS-MIT.txt) |
| HYPIR | `gradio_scripts/HYPIR/` | [HYPIR Software License Agreement](LICENSES/HYPIR-LICENSE.txt), including its non-commercial restrictions |
| Gaussian Splatting-derived code | `trellis/renderers/gaussian_render.py`, `trellis/representations/gaussian/general_utils.py`, and `extensions/vox2seq/setup.py` | [Gaussian-Splatting License](LICENSES/GAUSSIAN-SPLATTING-LICENSE.md), non-commercial research/evaluation only |
| FlexiCubes vendored snapshot | `trellis/representations/mesh/flexicubes/` | NVIDIA Apache License 2.0 files vendored from [`MaxtirError/FlexiCubes`](https://github.com/MaxtirError/FlexiCubes) at commit `815e075a2a400d06c48d94c347674344ed6ae5c5`; [local license](trellis/representations/mesh/flexicubes/LICENSE.txt) |
| PlenOctree spherical-harmonics helpers | `trellis/renderers/sh_utils.py` | BSD 2-Clause terms and The PlenOctree Authors' 2021 copyright notice retained at the top of the file |
| Transformers Tutorials Grounding DINO/SAM example | `gradio_scripts/grounding_sam.py` | Adapted from Niels Rogge's Transformers Tutorials; [MIT License](LICENSES/TRANSFORMERS-TUTORIALS-MIT.txt) |
| BasicSR download helper | `gradio_scripts/HYPIR/utils/common.py` (`load_file_from_url`) | BasicSR Authors, Apache License 2.0; upstream source URL retained in the file and the complete Apache text is included in the FlexiCubes license linked above |
| face-alignment download helper | `gradio_scripts/HYPIR/utils/common.py` (`load_file_from_url`) | Adrian Bulat, [BSD 3-Clause License](LICENSES/FACE-ALIGNMENT-BSD-3-CLAUSE.txt) |
| CCSR Gaussian tile weights | `gradio_scripts/HYPIR/utils/common.py` (`gaussian_weights`) | CCSR, Apache License 2.0; upstream source URL retained in the file and the complete Apache text is included in the FlexiCubes license linked above |
| Ultimate VAE Tile Optimization | `gradio_scripts/HYPIR/utils/tiled_vae/vaehook.py` | LI YI, MIT License; author, date, and license notice retained in the file; [MIT terms](LICENSES/ULTIMATE-VAE-TILE-MIT.txt) |

Copyright, patent, trademark, attribution, and modification notices present in individual source files remain in force.

## Models downloaded at runtime

The following models are referenced by repository identifiers and are not relicensed by HIVE-3D:

| Component | Source | License or terms |
| --- | --- | --- |
| TRELLIS image-to-3D model | `microsoft/TRELLIS-image-large` | MIT |
| HYPIR enhancement weights | `lxq007/HYPIR` | HYPIR Software License Agreement; the official GitHub terms are treated as controlling where metadata conflicts |
| Stable Diffusion 2.1 base | `sd2-community/stable-diffusion-2-1-base` | CreativeML Open RAIL++-M |
| SAM2 | `facebook/sam2-hiera-large` | Apache License 2.0 |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` | Apache License 2.0 |

Users are responsible for reviewing the current terms published by each model owner before use. Commercial use of the complete HIVE-3D workflow may require separate written permission from third-party copyright holders.

## Example assets

The HIVE-3D authors confirmed the following release provenance on 2026-08-10:

| Paths | Source/ownership | Release treatment |
| --- | --- | --- |
| `assets/pipeline.png` | HIVE-3D authors | Included with author-confirmed public redistribution permission |
| `assets/example/` | HIVE-3D authors, including the images and derived masks | Included with author-confirmed public redistribution permission |
| `assets/example_image/` | TRELLIS assets | Excluded by `.gitignore`; not part of this release |
| `assets/example_multi_image/` | TRELLIS assets | Excluded by `.gitignore`; not part of this release |

The root source-code license does not relicense assets owned by third parties.
