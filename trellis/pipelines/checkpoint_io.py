from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Union

import torch.nn as nn
from safetensors.torch import load_file


@dataclass(frozen=True)
class HiveCheckpointPaths:
    denoiser_config: Path
    denoiser_weights: Path
    projector_config: Path
    projector_weights: Path


_HIVE_CHECKPOINT_FILES = {
    "denoiser_config": "denoiser/config.json",
    "denoiser_weights": "denoiser/model.safetensors",
    "projector_config": "latent_projector/config.json",
    "projector_weights": "latent_projector/model.safetensors",
}


def resolve_hive_checkpoint_paths(
    source: Union[str, PathLike[str]],
) -> HiveCheckpointPaths:
    """Resolve a local HIVE-3D directory or Hugging Face repository ID."""
    source_path = Path(source)
    if source_path.is_dir():
        resolved = {
            name: source_path / relative_path
            for name, relative_path in _HIVE_CHECKPOINT_FILES.items()
        }
        for path in resolved.values():
            if not path.is_file():
                raise FileNotFoundError(f"HIVE-3D checkpoint file not found: {path}")
    else:
        from huggingface_hub import hf_hub_download

        resolved = {
            name: Path(hf_hub_download(repo_id=str(source), filename=relative_path))
            for name, relative_path in _HIVE_CHECKPOINT_FILES.items()
        }

    return HiveCheckpointPaths(**resolved)


def load_model_weights(model: nn.Module, path: Union[str, PathLike[str]]) -> None:
    """Load a safetensors state dictionary into a model with strict checks."""
    state_dict = load_file(str(path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
