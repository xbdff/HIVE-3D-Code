---
license: other
license_name: CC BY-NC 4.0
license_link: https://creativecommons.org/licenses/by-nc/4.0/legalcode
base_model: microsoft/TRELLIS-image-large
pipeline_tag: image-to-3d
tags:
  - image-to-3d
  - 3d-generation
  - non-commercial
---

# HIVE-3D

HIVE-3D provides a denoiser and latent projector for hierarchical image-to-3D scene reconstruction on top of `microsoft/TRELLIS-image-large`.

## Files

```text
denoiser/
├── config.json
└── model.safetensors
latent_projector/
├── config.json
└── model.safetensors
```

## License

The HIVE-3D weights and their accompanying configuration files are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) for academic research and non-commercial use.

This license applies only to the HIVE-3D files in this model repository. It does not relicense the TRELLIS base model, HYPIR, Stable Diffusion 2.1, SAM2, Grounding DINO, or any other third-party dependency. Users must comply with the terms published by each upstream owner.

The reference Gradio workflow includes HYPIR and Gaussian Splatting-derived components with separate non-commercial terms. See the HIVE-3D code repository's `THIRD_PARTY_NOTICES.md` before deploying the complete workflow.

## Base model

- [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large), MIT License

## Usage

The reference implementation downloads this repository automatically. Set the model source with:

```bash
export HIVE3D_MODEL=mocun123/HIVE-3D
```
