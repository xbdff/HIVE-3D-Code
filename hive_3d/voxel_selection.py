"""CPU-testable voxel scoring and connected-component selection helpers."""

import numpy as np
from PIL import Image
from scipy.ndimage import generate_binary_structure, label
import torch
import torch.nn.functional as F


def find_relevant_voxels_by_mask_path(
    mask_path,
    attn_map,
    additional_tokens_num=5,
    patch_size=14,
    heads=(0, 4, 12),
    threshold=0.2,
):
    """Score voxels against a mask and return scores plus selected indices."""
    device = attn_map.device
    target_dtype = attn_map.dtype

    attn = attn_map[list(heads)].mean(0)
    attn_patches = attn[:, additional_tokens_num:]
    _, num_patch_tokens = attn_patches.shape

    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img)
    mask_ts = torch.from_numpy(mask_np).to(device, dtype=target_dtype) / 255.0

    patch_weights = F.avg_pool2d(
        mask_ts.unsqueeze(0).unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    ).flatten()

    total_weights = torch.zeros(
        num_patch_tokens, device=device, dtype=target_dtype
    )
    n_copy = min(len(patch_weights), num_patch_tokens)
    total_weights[:n_copy] = patch_weights[:n_copy]
    voxel_scores = torch.matmul(attn_patches, total_weights)

    v_max = voxel_scores.max()
    v_min = voxel_scores.min()
    if (v_max - v_min) < 1e-10:
        return voxel_scores, [torch.argmax(voxel_scores).item()]

    voxel_relevance = (voxel_scores - v_min) / (v_max - v_min)
    indices = torch.nonzero(voxel_relevance > threshold).squeeze(1).tolist()
    if not indices:
        indices = [torch.argmax(voxel_scores).item()]

    return voxel_relevance, indices


def extract_best_scoring_component(
    coords,
    initial_indices,
    voxel_scores,
    grid_size=16,
    min_voxels=5,
):
    """Select the highest-scoring 6-connected XYZ component."""
    if len(initial_indices) == 0:
        return []

    device = coords.device
    indices_tensor = torch.as_tensor(
        initial_indices, device=device, dtype=torch.long
    )
    sel_coords = coords[indices_tensor, 1:4].cpu().numpy().astype(int)
    sel_scores = voxel_scores[indices_tensor].cpu().numpy()

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    coords_clipped = np.clip(sel_coords, 0, grid_size - 1)
    grid[
        coords_clipped[:, 0],
        coords_clipped[:, 1],
        coords_clipped[:, 2],
    ] = 1

    strict_structure = generate_binary_structure(3, 1)
    labeled_grid, num_features = label(grid, structure=strict_structure)
    if num_features == 0:
        return []

    voxel_block_ids = labeled_grid[
        coords_clipped[:, 0],
        coords_clipped[:, 1],
        coords_clipped[:, 2],
    ]
    block_total_scores = np.bincount(voxel_block_ids, weights=sel_scores)
    block_sizes = np.bincount(voxel_block_ids)
    block_total_scores[0] = -1e10
    block_total_scores[block_sizes < min_voxels] = -1e10

    best_block_id = np.argmax(block_total_scores)
    if block_total_scores[best_block_id] <= -1e9:
        print("Warning: no valid connected component found.")
        return []

    keep_mask = voxel_block_ids == best_block_id
    final_indices = indices_tensor[torch.from_numpy(keep_mask).to(device)]
    print(
        f"Strict alignment: {num_features} candidates, "
        f"selected ID {best_block_id}, voxels {len(final_indices)}, "
        f"score {block_total_scores[best_block_id]:.4f}"
    )
    return final_indices.tolist()
