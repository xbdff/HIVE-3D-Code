import numpy as np
from PIL import Image
from scipy.ndimage import generate_binary_structure, label
import torch
import trimesh

from .voxel_selection import (
    extract_best_scoring_component,
    find_relevant_voxels_by_mask_path,
)


def density(coords):
    """
    Convert sparse voxel coordinates to a dense voxel tensor.

    Args:
        coords: Tensor of shape [N, 4] containing [batch_idx, x, y, z].

    Returns:
        Tensor of shape [1, 1, 64, 64, 64] with occupied voxels set to 1.
    """
    device = coords.device
    dense_tensor = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    # Only batch 0 is supported.
    x = coords[:, 1].long()
    y = coords[:, 2].long()
    z = coords[:, 3].long()
    # Filter out-of-range coordinates.
    mask = (z >= 0) & (z < 64) & (y >= 0) & (y < 64) & (x >= 0) & (x < 64)
    z = z[mask]
    y = y[mask]
    x = x[mask]
    dense_tensor[0, 0, x, y, z] = 1.0
    return dense_tensor 


def upsample_coords_adaptive(coords: "torch.Tensor", 
                             target_size: int = 32, 
                             max_size: int = 64) -> "torch.Tensor":
    """
    Adaptively upsample coordinates toward the target bounding-box size.

    Args:
        coords (torch.Tensor): Coordinate tensor of shape [N, 4].
        target_size (int): Stop when any axis reaches this size.
        max_size (int): Maximum allowed bounding-box size.

    Returns:
        torch.Tensor: Upsampled and centered coordinates.
    """
    import torch

    if coords.shape[0] == 0:
        return torch.empty(0, 4, dtype=coords.dtype, device=coords.device)

    # Work on a copy.
    working_coords = coords.clone()

    # Iteratively upsample.
    while True:
        if working_coords.shape[0] == 0:
            # Stop if the point set becomes empty.
            break
            
        xyz = working_coords[:, 1:]
        if torch.is_floating_point(xyz):
            xyz = torch.round(xyz).to(torch.long)
        else:
            xyz = xyz.to(torch.long)

        # Compute the current bounding-box size.
        if xyz.shape[0] > 0:
            min_xyz = xyz.min(dim=0).values
            max_xyz = xyz.max(dim=0).values
            bbox_size = max_xyz - min_xyz + 1
        else:
            bbox_size = xyz.new_tensor([0, 0, 0])

        # Stop when any axis reaches the target.
        if torch.any(bbox_size >= target_size):
            print(f"Stopping upsampling: Bbox size {bbox_size.tolist()} reached target >= {target_size}.")
            break
        
        # Stop if the next step would exceed the limit.
        predicted_next_size = bbox_size * 2 - 1
        if torch.any(predicted_next_size >= max_size):
            print(f"Stopping upsampling: Predicted next bbox size {predicted_next_size.tolist()} would exceed max_size >= {max_size}.")
            break

        print(f"Current bbox size: {bbox_size.tolist()}. Upsampling...")
        
        # Normalize coordinates and upsample by 2.
        relocated_xyz = xyz - min_xyz
        
        # Scale.
        scaled_relocated_xyz = relocated_xyz * 2
        
        # Fill each voxel with eight offsets.
        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=scaled_relocated_xyz.dtype, device=scaled_relocated_xyz.device)
        
        upsampled_relocated_xyz = scaled_relocated_xyz.unsqueeze(1) + offsets
        upsampled_relocated_xyz_flat = upsampled_relocated_xyz.view(-1, 3)
        
        # Restore the coordinate origin.
        final_upsampled_xyz = upsampled_relocated_xyz_flat + min_xyz
        
        # Replicate the feature column.
        feature_col = working_coords[:, 0].unsqueeze(1)
        replicated_feature_col_flat = feature_col.repeat(1, 8).view(-1, 1)
        
        # Update coordinates for the next iteration.
        working_coords = torch.cat((replicated_feature_col_flat.to(final_upsampled_xyz.dtype), final_upsampled_xyz), dim=1)

    # Center the final coordinates.
    final_xyz = working_coords[:, 1:]
    if torch.is_floating_point(final_xyz):
        final_xyz = torch.round(final_xyz).to(torch.long)
    else:
        final_xyz = final_xyz.to(torch.long)
        
    if final_xyz.shape[0] == 0:
        return working_coords # Return the empty tensor.

    # Compute the current bounding-box center.
    cur_min = final_xyz.min(dim=0, keepdim=True).values
    cur_max = final_xyz.max(dim=0, keepdim=True).values
    cur_center = (cur_min + cur_max) // 2
    
    # Center in the target grid.
    target_center = final_xyz.new_tensor([[max_size // 2 - 1] * 3], dtype=final_xyz.dtype)
    shift = target_center - cur_center
    
    shifted_upsampled_xyz = final_xyz + shift
    
    # Reattach the feature column.
    result = torch.cat((working_coords[:, 0].unsqueeze(1).to(shifted_upsampled_xyz.dtype), shifted_upsampled_xyz), dim=1)
    
    return result


def find_dark_patches_fixed_size(
    image_path: str,
    patch_size: int = 14,
    pixel_threshold: int = 50
) -> list[int]:
    """
    Return patches containing more than the threshold number of black pixels.

    Args:
        image_path (str): Path to a binary mask.
        patch_size (int): Patch side length.
        pixel_threshold (int): Minimum black-pixel count.

    Returns:
        list[int]: Indices of matching patches.
    """
    try:
        # Read the mask as grayscale.
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: file not found: {image_path}")
        return []
    except Exception as e:
        print(f"Error reading image: {e}")
        return []

    img_array = np.array(img)
    img_height, img_width = img_array.shape

    # Require dimensions divisible by the patch size.
    if img_height % patch_size != 0 or img_width % patch_size != 0:
        print(f"Error: image size ({img_height}, {img_width}) is not divisible by patch size {patch_size}.")
        return []

    # Compute the patch grid.
    grid_h = img_height // patch_size
    grid_w = img_width // patch_size
    
    dark_patch_indices = []

    # Scan all patches.
    for i in range(grid_h):  # Patch row.
        for j in range(grid_w):  # Patch column.
            # Locate the patch.
            start_row = i * patch_size
            end_row = start_row + patch_size
            start_col = j * patch_size
            end_col = start_col + patch_size

            # Extract the patch.
            patch = img_array[start_row:end_row, start_col:end_col]

            # Count black pixels.
            black_pixel_count = np.sum(patch == 0)

            # Apply the threshold.
            if black_pixel_count > pixel_threshold:
                # Flatten the grid index.
                patch_index = i * grid_w + j
                dark_patch_indices.append(patch_index)

    return dark_patch_indices


def save_selected_voxel(coords, selected_coords, output_path):
    selected_coords = selected_coords.cpu().numpy()
    coords = coords.cpu().numpy()[:, 1:]
    encoding = trimesh.voxel.encoding.SparseBinaryEncoding(coords)
    voxel_grid = trimesh.voxel.VoxelGrid(encoding)
    colors_on_voxel = np.zeros(
        (
            int(voxel_grid.shape[0]),
            int(voxel_grid.shape[1]),
            int(voxel_grid.shape[2]),
            3,
        ),
        dtype=np.float32,
    )
    colors = np.ones((len(coords), 3))
    colors[:] = [1, 0, 0]
    if selected_coords is not None:
        colors[selected_coords, :] = [0, 1, 0]
    for i, coord in enumerate(coords):
        colors_on_voxel[coord[0], coord[1], coord[2]] = colors[i]
    voxel_mesh = voxel_grid.as_boxes(colors_on_voxel)
    voxel_mesh.export(output_path)

    

def remove_indices_from_sparsetensor(sparse_tensor, remove_indices):
    """
    Remove indices from a spconv SparseTensor.

    Args:
        sparse_tensor: Input sparse tensor.
        remove_indices: Indices to remove.

    Returns:
        A new sparse tensor.
    """

    remove_indices = torch.tensor(remove_indices, device=sparse_tensor.feats.device)
    total_indices = torch.arange(sparse_tensor.feats.shape[0], device=sparse_tensor.feats.device)
    keep_mask = ~torch.isin(total_indices, remove_indices)

    new_feats = sparse_tensor.feats[keep_mask]
    new_coords = sparse_tensor.coords[keep_mask]

    # Build a new sparse tensor.
    new_sparse_tensor = type(sparse_tensor)(feats=new_feats, coords=new_coords)
    return new_sparse_tensor


def refine_voxel_alignment(coords, indices, k=16, fill_threshold=0.6, clear_threshold=0.4):
    """
    Refine selected voxels with float32 nearest-neighbor voting.

    Memory use is approximately N^2 * 4 bytes.
    """
    device = coords.device
    n_voxels = coords.size(0)
    
    # Compute distances in float32.
    coords_f = coords.float()
    
    # Use the optimized cdist implementation.
    dist_matrix = torch.cdist(coords_f, coords_f, p=2)
    
    # Select nearest neighbors.
    _, nn_idx = torch.topk(dist_matrix, k=k, largest=False, sorted=False)
    
    # Release the distance matrix.
    del dist_matrix
    
    # Build the selection mask.
    is_selected = torch.zeros(n_voxels, dtype=torch.bool, device=device)
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=device, dtype=torch.long)
    is_selected[indices.long()] = True
    
    # Measure the selected-neighbor ratio.
    neighbor_status = is_selected[nn_idx] # (N, k)
    selected_ratios = neighbor_status.float().mean(dim=1)
    
    # Apply fill and clear thresholds.
    new_selected_mask = is_selected.clone()
    
    # Fill holes.
    new_selected_mask[(~is_selected) & (selected_ratios >= fill_threshold)] = True
    # Remove isolated voxels.
    new_selected_mask[is_selected & (selected_ratios < clear_threshold)] = False
    
    # Return long indices.
    return torch.where(new_selected_mask)[0]


def extract_largest_component_upsampled(orig_coords, matched_indices, grid_size=64, connectivity=1):
    """
    Keep the largest connected component after upsampling.

    Args:
        orig_coords (Tensor): Full coordinate set of shape [N, 4].
        matched_indices (Tensor): Matched indices of shape [K].
        grid_size (int): Grid resolution.
        connectivity (int): 1 for 6-, 2 for 18-, or 3 for 26-connectivity.
    """
    if matched_indices.numel() == 0:
        return torch.empty((0, orig_coords.shape[-1]), device=orig_coords.device), \
               torch.empty((0,), device=orig_coords.device)

    device = orig_coords.device
    
    # Gather matched coordinates.
    subset_coords = orig_coords[matched_indices.long()]
    
    # Move XYZ coordinates to CPU for component labeling.
    coords_xyz = subset_coords[:, 1:].cpu().numpy().astype(int)
    
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Clip coordinates and fill the grid.
    coords_clipped = np.clip(coords_xyz, 0, grid_size - 1)
    grid[coords_clipped[:, 0], coords_clipped[:, 1], coords_clipped[:, 2]] = True

    # Label connected components.
    struct = generate_binary_structure(3, connectivity)
    labeled_grid, num_features = label(grid, structure=struct)
    
    if num_features <= 1:
        # Return directly if already connected.
        return subset_coords, matched_indices

    # Select the largest component.
    component_sizes = np.bincount(labeled_grid.ravel())
    component_sizes[0] = 0  # Ignore the background.
    largest_label = component_sizes.argmax()

    # Map component labels back to matched indices.
    pixel_labels = labeled_grid[coords_clipped[:, 0], coords_clipped[:, 1], coords_clipped[:, 2]]
    keep_mask = (pixel_labels == largest_label)
    
    # Filter points and indices.
    mask_torch = torch.from_numpy(keep_mask).to(device)
    refined_points = subset_coords[mask_torch]
    refined_indices = matched_indices[mask_torch]

    print(f"Connectivity cleanup: kept the largest of {num_features} components ({len(refined_points)} voxels)")
    
    return refined_points, refined_indices
