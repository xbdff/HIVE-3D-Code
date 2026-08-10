import numpy as np
import torch
from PIL import Image

from hive_3d.voxel_selection import (
    extract_best_scoring_component,
    find_relevant_voxels_by_mask_path,
)


def test_flat_attention_returns_scores_and_fallback_index(tmp_path):
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((14, 14), 255, dtype=np.uint8)).save(mask)
    scores, indices = find_relevant_voxels_by_mask_path(
        mask, torch.ones((13, 3, 6), dtype=torch.float32)
    )
    assert scores.shape == (3,)
    assert indices == [0]


def test_component_selection_uses_xyz_not_batch_xy():
    coords = torch.tensor(
        [[0, 1, 1, z] for z in range(5)]
        + [[0, 1, 1, z] for z in range(10, 15)]
    )
    scores = torch.tensor([1.0] * 5 + [0.1] * 5)
    selected = extract_best_scoring_component(
        coords, list(range(10)), scores, grid_size=16, min_voxels=5
    )
    assert selected == list(range(5))
