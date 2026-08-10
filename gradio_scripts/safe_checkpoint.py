"""Restricted loading helpers for tensor-only PyTorch checkpoints."""

from pathlib import Path

import torch


def load_tensor_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dict without allowing arbitrary pickle globals."""
    state_dict = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError(
            "checkpoint state dictionary must contain only string-to-tensor entries"
        )
    return state_dict
