import torch


def trilinear_interpolation(sparse_tensor, query_points):
    """
    Interpolate sparse-tensor features at query points.

    Args:
        sparse_tensor: SparseTensor with coords [N, 4] and feats [N, 8].
        query_points: Query coordinates [M, 3].

    Returns:
        Interpolated features [M, 8].
    """
    coords = sparse_tensor.coords
    feats = sparse_tensor.feats
    
    # Compute spatial bounds.
    min_coords = torch.min(coords[:,1:], dim=0).values 
    max_coords = torch.max(coords[:,1:], dim=0).values
    
    # Normalize query coordinates to [0, 1].
    norm_points = (query_points - min_coords) / (max_coords - min_coords)
    
    # Find the eight surrounding grid vertices.
    floor = torch.floor(norm_points)
    ceil = torch.ceil(norm_points)
    
    # Compute interpolation weights.
    weights = norm_points - floor
    
    # Build vertex coordinates.
    c000 = torch.cat([torch.zeros(len(query_points),1), floor], dim=1)
    c001 = torch.cat([torch.zeros(len(query_points),1), floor[:,:2], ceil[:,2:]], dim=1)
    c010 = torch.cat([torch.zeros(len(query_points),1), floor[:,0:1], ceil[:,1:2], floor[:,2:]], dim=1)
    c011 = torch.cat([torch.zeros(len(query_points),1), floor[:,0:1], ceil[:,1:], floor[:,2:]], dim=1)
    c100 = torch.cat([torch.zeros(len(query_points),1), ceil[:,0:1], floor[:,1:]], dim=1)
    c101 = torch.cat([torch.zeros(len(query_points),1), ceil[:,0:1], floor[:,1:2], ceil[:,2:]], dim=1)
    c110 = torch.cat([torch.zeros(len(query_points),1), ceil[:,0:2], floor[:,2:]], dim=1)
    c111 = torch.cat([torch.zeros(len(query_points),1), ceil], dim=1)
    
    # Look up vertex features.
    def get_feat(c):
        indices = []
        for i in range(len(c)):
            idx = torch.where((coords == c[i]).all(dim=1))[0]
            indices.append(idx[0] if len(idx) > 0 else -1)
        return feats[indices]
    
    f000 = get_feat(c000)
    f001 = get_feat(c001)
    f010 = get_feat(c010) 
    f011 = get_feat(c011)
    f100 = get_feat(c100)
    f101 = get_feat(c101)
    f110 = get_feat(c110)
    f111 = get_feat(c111)
    
    # Trilinear interpolation.
    wx, wy, wz = weights[:,0:1], weights[:,1:2], weights[:,2:3]
    
    interpolated = (f000 * (1-wx) * (1-wy) * (1-wz) +
                   f001 * (1-wx) * (1-wy) * wz +
                   f010 * (1-wx) * wy * (1-wz) +
                   f011 * (1-wx) * wy * wz +
                   f100 * wx * (1-wy) * (1-wz) +
                   f101 * wx * (1-wy) * wz +
                   f110 * wx * wy * (1-wz) +
                   f111 * wx * wy * wz)
                   
    return interpolated


# Scale spatial coordinates.
def process_coords(new_coords):
    # Extract spatial coordinates.
    spatial = new_coords[:, 1:4]
        
    # Use the minimum corner as the origin.
    origin = spatial.min(dim=0).values.to(torch.int32)  # [3]


    # Shift to the origin.
    shifted = spatial - origin
    

    # Double each spatial axis.
    scaled = shifted * 2

    # Convert to int32.
    scaled_int = scaled.round().to(torch.int32)

    # Restore the original origin.
    final_spatial = scaled_int + origin * 1
    #final_spatial = scaled_int

    # Reattach the batch column.
    batch_col = new_coords[:, 0:1].to(torch.int32)
    final_coords = torch.cat([batch_col, final_spatial], dim=1)  # [K,4]
    return final_coords

def interpolate_sparse_tensor_to_half_voxels_1(sparse_tensor, sp):
    """
    Interpolate features at half-voxel positions.

    Args:
        sparse_tensor: Sparse tensor with coordinates [N, 4] and features [N, 8].
        sp: Sparse tensor module.

    Returns:
        Sparse tensor containing nonzero interpolated features.
    """
    coords = sparse_tensor.coords  # [N,4]
    feats = sparse_tensor.feats   # [N,8]
    device = feats.device
    min_valid_corners = 1

    # Compute bounds.
    min_xyz = coords[:, 1:].min(dim=0).values
    max_xyz = coords[:, 1:].max(dim=0).values

    # Build a dense feature grid.
    grid_size = (max_xyz - min_xyz + 1).int().tolist()  # [X, Y, Z]
    dense_feats = torch.zeros([*grid_size, feats.shape[1]], device=device)
    for i in range(coords.shape[0]):
        x, y, z = (coords[i, 1:] - min_xyz).int()
        dense_feats[x, y, z] = feats[i]

    # Generate all half-integer query points.
    X, Y, Z = grid_size
    xs = torch.arange(0.5, X-1+0.5, 1, device=device)
    ys = torch.arange(0.5, Y-1+0.5, 1, device=device)
    zs = torch.arange(0.5, Z-1+0.5, 1, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
    query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [M,3]

    # Restore the original coordinate system.
    query_points_real = query_points + min_xyz  # [M,3]

    # Interpolate.
    def trilinear_interp(dense_feats, q):
        # q has shape [M, 3] in the dense grid.
        M, F = q.shape[0], dense_feats.shape[-1]
        out_feats = torch.zeros((M, F), device=device)
        for i in range(M):
            x, y, z = q[i]
            x0, y0, z0 = torch.floor(torch.tensor([x, y, z])).long()
            xd, yd, zd = x - x0, y - y0, z - z0
            # Clamp to valid cells.
            x0 = min(max(x0, 0), X-2)
            y0 = min(max(y0, 0), Y-2)
            z0 = min(max(z0, 0), Z-2)
            c000 = dense_feats[x0,   y0,   z0  ]
            c001 = dense_feats[x0,   y0,   z0+1]
            c010 = dense_feats[x0,   y0+1, z0  ]
            c011 = dense_feats[x0,   y0+1, z0+1]
            c100 = dense_feats[x0+1, y0,   z0  ]
            c101 = dense_feats[x0+1, y0,   z0+1]
            c110 = dense_feats[x0+1, y0+1, z0  ]
            c111 = dense_feats[x0+1, y0+1, z0+1]

            # Count nonzero corners.
            corners = [c000, c001, c010, c011, c100, c101, c110, c111]
            non_zero_corners = sum(1 for c in corners if torch.any(c != 0))

            # Interpolate when enough corners are valid.
            if non_zero_corners >= min_valid_corners:
                out_feats[i] = (
                    c000 * (1-xd)*(1-yd)*(1-zd) +
                    c001 * (1-xd)*(1-yd)*zd +
                    c010 * (1-xd)*yd*(1-zd) +
                    c011 * (1-xd)*yd*zd +
                    c100 * xd*(1-yd)*(1-zd) +
                    c101 * xd*(1-yd)*zd +
                    c110 * xd*yd*(1-zd) +
                    c111 * xd*yd*zd
                )
                out_feats[i] = out_feats[i] * 8 / non_zero_corners
        return out_feats

    interpolated_feats = trilinear_interp(dense_feats, query_points)

    

    # Keep nonzero features.
    mask = (interpolated_feats.abs().sum(dim=1) != 0)
    new_coords_xyz = query_points_real[mask]  # [K,3]
    new_feats = interpolated_feats[mask]      # [K,8]
    batch_idx = torch.zeros(new_coords_xyz.shape[0], 1, device=device, dtype=coords.dtype)
    new_coords = torch.cat([batch_idx, new_coords_xyz], dim=1)# [K,4]
    new_coords = torch.cat([coords, new_coords], dim=0)
    new_feats = torch.cat([feats, new_feats], dim=0)


    new_coords = process_coords(new_coords)

    # Return the new sparse tensor.
    return sp.SparseTensor(new_feats, new_coords)

def interpolate_sparse_tensor_to_half_voxels_2(sparse_tensor, sp):
    # Read original coordinates and features.
    original_coords = sparse_tensor.coords # [N_orig, 4]
    # Features have shape [N_orig, 8].
    original_feats = sparse_tensor.feats       # [N_orig, 8]

    # Map original coordinates to features.
    original_feature_map = {tuple(c.tolist()): original_feats[i] for i, c in enumerate(original_coords)}

    # Extract integer spatial coordinates.
    spatial_coords_orig = original_coords[:, 1:4] # [N_orig, 3] (int32)

    # Find the minimum corner.
    min_spatial_orig = spatial_coords_orig.min(dim=0).values  # [3] (int32)

    # Map original voxels into the doubled grid.
    shifted_spatial_orig = spatial_coords_orig - min_spatial_orig # Remains int32.
    shifted_spatial_plus_one_orig = shifted_spatial_orig + 1 # Remains int32.
    # Store each original voxel at the upper grid corner.
    scaled_spatial_orig = shifted_spatial_plus_one_orig * 2 # [N_orig, 3] (int32)

    # Extract the batch column.
    batch_col_orig = original_coords[:, 0:1] # [N_orig, 1] (int32)

    # Reattach the batch column.
    scaled_full_coords = torch.cat([batch_col_orig, scaled_spatial_orig], dim=1) # [N_orig, 4]


    # Generate eight corners around each scaled voxel.
    offsets_3d = []
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                offsets_3d.append([dx, dy, dz])

    offsets_3d = torch.tensor(offsets_3d, dtype=scaled_spatial_orig.dtype, device=scaled_spatial_orig.device) # (8, 3)

    # Broadcast scaled coordinates against all offsets.
    # scaled_spatial_orig: (N_orig, 3) -> (N_orig, 1, 3)
    # offsets_3d: (8, 3) -> (1, 8, 3)
    all_8_spatial_coords_around_scaled = scaled_spatial_orig.unsqueeze(1) - offsets_3d.unsqueeze(0) # (N_orig, 8, 3)

    # Add the batch column.
    # batch_col_orig: (N_orig, 1) -> (N_orig, 1, 1) -> (N_orig, 8, 1)
    batch_col_expanded = batch_col_orig.unsqueeze(1).repeat(1, 8, 1) # (N_orig, 8, 1)

    # Concatenate batch and spatial coordinates.
    all_8_coords_around_scaled = torch.cat([batch_col_expanded, all_8_spatial_coords_around_scaled], dim=2) # (N_orig, 8, 4)

    # Flatten all 8N coordinates.
    all_8_coords_flat = all_8_coords_around_scaled.reshape(-1, 4) # (N_orig * 8, 4) # Use reshape


    # Deduplicate final coordinates.
    unique_final_coords, original_indices_in_8N = torch.unique(all_8_coords_flat, return_inverse=True, dim=0) # [N_final, 4]
    # N_final is the number of unique points.

    # Map each unique coordinate to its index.
    unique_coords_to_index = {tuple(c.tolist()): i for i, c in enumerate(unique_final_coords)}


    # Initialize final features.
    final_feats = torch.zeros((unique_final_coords.size(0), original_feats.size(1)), dtype=original_feats.dtype, device=original_feats.device)


    # Copy original voxel features into the doubled grid.
    scaled_full_coords_list = scaled_full_coords.tolist()
    for i_orig, c_scaled_full in enumerate(scaled_full_coords_list):
        # Find the final index.
        if tuple(c_scaled_full) in unique_coords_to_index:
            idx_in_final = unique_coords_to_index[tuple(c_scaled_full)]
            # Assign the original feature.
            final_feats[idx_in_final] = original_feats[i_orig]


    # Interpolate newly generated voxels.
    scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}

    # Build 26-neighbor offsets.
    neighborhood_offsets_3d_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if not (dx == 0 and dy == 0 and dz == 0):
                    neighborhood_offsets_3d_26.append([dx, dy, dz])
    neighborhood_offsets_3d_26 = torch.tensor(neighborhood_offsets_3d_26, dtype=torch.int32, device=unique_final_coords.device) # (26, 3)

    # Interpolate with inverse-distance weights.
    epsilon = 1e-6 # Avoid division by zero.

    # original_feature_map provides source features.

    # Interpolate coordinates not present in scaled originals.
    for idx_in_final, final_c in enumerate(unique_final_coords):
        final_c_tuple = tuple(final_c.tolist())

        if final_c_tuple not in scaled_full_coords_set: # Newly generated point.

            neighbor_weights_and_feats = [] # Store (raw_weight, neighbor_feat) for found original neighbors

            # Approximate the corresponding original position.
            final_c_spatial_float = final_c[1:4].float()
            approx_orig_spatial_float = (final_c_spatial_float / 2.0 - 1.0) + min_spatial_orig.float() # Use float for inversion.

            # Build original-space 3x3x3 offsets.
            orig_neighborhood_offsets_3d = []
            for dx in [-1, 0, 1]:
                 for dy in [-1, 0, 1]:
                     for dz in [-1, 0, 1]:
                         orig_neighborhood_offsets_3d.append([dx, dy, dz])
            orig_neighborhood_offsets_3d = torch.tensor(orig_neighborhood_offsets_3d, dtype=torch.int32, device=original_coords.device) # Includes the center.


            # approx_orig_spatial_int_center = approx_orig_spatial_float.round().to(torch.int32)
            # # Scan original voxels around the estimated center.
            # for orig_offset_3d in orig_neighborhood_offsets_3d:
            #     neighbor_orig_spatial_int = approx_orig_spatial_int_center + orig_offset_3d
            #     # Build full neighbor coordinates.
            #     neighbor_orig_full_c_list = [final_c[0].item()] + neighbor_orig_spatial_int.tolist()
            #     neighbor_orig_full_c_tuple = tuple(neighbor_orig_full_c_list)

            #     # Look up original neighbors.
            #     if neighbor_orig_full_c_tuple in original_feature_map:
            #          neighbor_feat = original_feature_map[neighbor_orig_full_c_tuple]

            #          # Compute distance between scaled and original spaces.

            # Interpolate from transformed original voxels in the scaled
            # 3x3x3 neighborhood using scaled-space distances.

            # Build fast coordinate and feature lookups.
            scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}
            # Map scaled coordinates to original features.
            scaled_to_original_feature_map = {tuple(scaled_full_coords[i].tolist()): original_feats[i] for i in range(scaled_full_coords.size(0))}


            # Scan the scaled 3x3x3 neighborhood.
            final_c_spatial_int = final_c[1:4]
            for offset_3d in neighborhood_offsets_3d_26: # 26 offsets.
                # Compute scaled neighbor coordinates.
                neighbor_scaled_spatial_int = final_c_spatial_int + offset_3d
                # Reattach the batch coordinate.
                neighbor_scaled_full_c_list = [final_c[0].item()] + neighbor_scaled_spatial_int.tolist()
                neighbor_scaled_full_c_tuple = tuple(neighbor_scaled_full_c_list)

                # Use transformed original voxels only.
                if neighbor_scaled_full_c_tuple in scaled_full_coords_set:
                    # Retrieve the original feature.
                    neighbor_feat = scaled_to_original_feature_map[neighbor_scaled_full_c_tuple]

                    # Compute scaled-space distance.
                    neighbor_scaled_spatial_tensor = torch.tensor(neighbor_scaled_full_c_list[1:], dtype=torch.float32, device=final_c.device)
                    dist = torch.norm(final_c[1:4].float() - neighbor_scaled_spatial_tensor)

                    raw_weight = 1.0 / (dist + epsilon)
                    neighbor_weights_and_feats.append((raw_weight, neighbor_feat))

            # Compute the normalized weighted average.
            if neighbor_weights_and_feats: # Neighbors found.
                raw_weights = torch.tensor([item[0] for item in neighbor_weights_and_feats], dtype=original_feats.dtype, device=original_feats.device)
                neighbor_feats_stack = torch.stack([item[1] for item in neighbor_weights_and_feats])

                sum_raw_weights = torch.sum(raw_weights)

                if sum_raw_weights > 0:
                    normalized_weights = raw_weights / sum_raw_weights
                    final_feats[idx_in_final] = torch.sum(normalized_weights.unsqueeze(1) * neighbor_feats_stack, dim=0)
                else:
                     final_feats[idx_in_final] = torch.zeros_like(original_feats[0])
            else:
                final_feats[idx_in_final] = torch.zeros_like(original_feats[0])


    # Build the output sparse tensor.
    output_sparse_tensor = sp.SparseTensor(final_feats, unique_final_coords.int())


    return output_sparse_tensor

def interpolate_sparse_tensor_to_half_voxels_3(sparse_tensor, sp):
    # Read original coordinates and features.
    original_coords = sparse_tensor.coords # [N_orig, 4]
    # Features have shape [N_orig, 8].
    original_feats = sparse_tensor.feats       # [N_orig, 8]

    # Map original coordinates to features.
    original_feature_map = {tuple(c.tolist()): original_feats[i] for i, c in enumerate(original_coords)}

    # Extract integer spatial coordinates.
    spatial_coords_orig = original_coords[:, 1:4] # [N_orig, 3] (int32)

    # Find the minimum corner.
    min_spatial_orig = spatial_coords_orig.min(dim=0).values  # [3] (int32)

    # Map original voxels into the doubled grid.
    shifted_spatial_orig = spatial_coords_orig - min_spatial_orig # Remains int32.
    shifted_spatial_plus_one_orig = shifted_spatial_orig + 1 # Remains int32.
    # Store each original voxel at the upper grid corner.
    scaled_spatial_orig = shifted_spatial_plus_one_orig * 2 # [N_orig, 3] (int32)

    # Extract the batch column.
    batch_col_orig = original_coords[:, 0:1] # [N_orig, 1] (int32)

    # Reattach the batch column.
    scaled_full_coords = torch.cat([batch_col_orig, scaled_spatial_orig], dim=1) # [N_orig, 4]


    # Generate eight corners around each scaled voxel.
    offsets_3d = []
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                offsets_3d.append([dx, dy, dz])

    offsets_3d = torch.tensor(offsets_3d, dtype=scaled_spatial_orig.dtype, device=scaled_spatial_orig.device) # (8, 3)

    # Broadcast scaled coordinates against all offsets.
    # scaled_spatial_orig: (N_orig, 3) -> (N_orig, 1, 3)
    # offsets_3d: (8, 3) -> (1, 8, 3)
    all_8_spatial_coords_around_scaled = scaled_spatial_orig.unsqueeze(1) - offsets_3d.unsqueeze(0) # (N_orig, 8, 3)

    # Add the batch column.
    # batch_col_orig: (N_orig, 1) -> (N_orig, 1, 1) -> (N_orig, 8, 1)
    batch_col_expanded = batch_col_orig.unsqueeze(1).repeat(1, 8, 1) # (N_orig, 8, 1)

    # Concatenate batch and spatial coordinates.
    all_8_coords_around_scaled = torch.cat([batch_col_expanded, all_8_spatial_coords_around_scaled], dim=2) # (N_orig, 8, 4)

    # Flatten all 8N coordinates.
    all_8_coords_flat = all_8_coords_around_scaled.reshape(-1, 4) # (N_orig * 8, 4) # Use reshape


    # Deduplicate final coordinates.
    unique_final_coords, original_indices_in_8N = torch.unique(all_8_coords_flat, return_inverse=True, dim=0) # [N_final, 4]
    # N_final is the number of unique points.

    # Map each unique coordinate to its index.
    unique_coords_to_index = {tuple(c.tolist()): i for i, c in enumerate(unique_final_coords)}


    # Initialize final features.
    final_feats = torch.zeros((unique_final_coords.size(0), original_feats.size(1)), dtype=original_feats.dtype, device=original_feats.device)


    # Copy original voxel features into the doubled grid.
    scaled_full_coords_list = scaled_full_coords.tolist()
    for i_orig, c_scaled_full in enumerate(scaled_full_coords_list):
        # Find the final index.
        if tuple(c_scaled_full) in unique_coords_to_index:
            idx_in_final = unique_coords_to_index[tuple(c_scaled_full)]
            # Assign the original feature.
            final_feats[idx_in_final] = original_feats[i_orig]


    # Interpolate newly generated voxels.
    scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}

    # Build 26-neighbor offsets.
    neighborhood_offsets_3d_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if not (dx == 0 and dy == 0 and dz == 0):
                    neighborhood_offsets_3d_26.append([dx, dy, dz])
    neighborhood_offsets_3d_26 = torch.tensor(neighborhood_offsets_3d_26, dtype=torch.int32, device=unique_final_coords.device) # (26, 3)

    # Interpolate with inverse-distance weights.
    epsilon = 1e-6 # Avoid division by zero.

    # original_feature_map provides source features.

    # Track source and newly interpolated features.
    current_voxel_map = {tuple(scaled_full_coords[i].tolist()): original_feats[i] for i in range(scaled_full_coords.size(0))}

    # Interpolate coordinates not present in scaled originals.
    for idx_in_final, final_c in enumerate(unique_final_coords):
        final_c_tuple = tuple(final_c.tolist())

        if final_c_tuple not in scaled_full_coords_set: # Newly generated point.

            neighbor_weights_and_feats = [] # Store (raw_weight, neighbor_feat) for found original neighbors

            # Approximate the corresponding original position.
            final_c_spatial_float = final_c[1:4].float()
            approx_orig_spatial_float = (final_c_spatial_float / 2.0 - 1.0) + min_spatial_orig.float() # Use float for inversion.

            # Build original-space 3x3x3 offsets.
            orig_neighborhood_offsets_3d = []
            for dx in [-1, 0, 1]:
                 for dy in [-1, 0, 1]:
                     for dz in [-1, 0, 1]:
                         orig_neighborhood_offsets_3d.append([dx, dy, dz])
            orig_neighborhood_offsets_3d = torch.tensor(orig_neighborhood_offsets_3d, dtype=torch.int32, device=original_coords.device) # Includes the center.


            # approx_orig_spatial_int_center = approx_orig_spatial_float.round().to(torch.int32)
            # # Scan original voxels around the estimated center.
            # for orig_offset_3d in orig_neighborhood_offsets_3d:
            #     neighbor_orig_spatial_int = approx_orig_spatial_int_center + orig_offset_3d
            #     # Build full neighbor coordinates.
            #     neighbor_orig_full_c_list = [final_c[0].item()] + neighbor_orig_spatial_int.tolist()
            #     neighbor_orig_full_c_tuple = tuple(neighbor_orig_full_c_list)

            #     # Look up original neighbors.
            #     if neighbor_orig_full_c_tuple in original_feature_map:
            #          neighbor_feat = original_feature_map[neighbor_orig_full_c_tuple]

            #          # Compute cross-space distance.

            # Interpolate in the scaled 3x3x3 neighborhood.

            # Build fast coordinate and feature lookups.
            scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}
            # Map scaled coordinates to original features.
            scaled_to_original_feature_map = {tuple(scaled_full_coords[i].tolist()): original_feats[i] for i in range(scaled_full_coords.size(0))}


            # Scan the scaled 3x3x3 neighborhood.
            final_c_spatial_int = final_c[1:4]
            for offset_3d in neighborhood_offsets_3d_26: # 26 offsets.
                # Compute scaled neighbor coordinates.
                neighbor_scaled_spatial_int = final_c_spatial_int + offset_3d
                # Reattach the batch coordinate.
                neighbor_scaled_full_c_list = [final_c[0].item()] + neighbor_scaled_spatial_int.tolist()
                neighbor_scaled_full_c_tuple = tuple(neighbor_scaled_full_c_list)

                # Look up source or previously interpolated features.
                if neighbor_scaled_full_c_tuple in current_voxel_map:
                    neighbor_feat = current_voxel_map[neighbor_scaled_full_c_tuple]

                    # Compute scaled-space distance.

                    neighbor_scaled_spatial_tensor = torch.tensor(neighbor_scaled_full_c_list[1:], dtype=torch.float32, device=final_c.device)
                    dist = torch.norm(final_c[1:4].float() - neighbor_scaled_spatial_tensor)

                    raw_weight = 1.0 / (dist + epsilon)
                    neighbor_weights_and_feats.append((raw_weight, neighbor_feat))

            # Compute the normalized weighted average.
            if neighbor_weights_and_feats: # Neighbors found.
                raw_weights = torch.tensor([item[0] for item in neighbor_weights_and_feats], dtype=original_feats.dtype, device=original_feats.device)
                neighbor_feats_stack = torch.stack([item[1] for item in neighbor_weights_and_feats])

                sum_raw_weights = torch.sum(raw_weights)

                if sum_raw_weights > 0:
                    normalized_weights = raw_weights / sum_raw_weights
                    final_feats[idx_in_final] = torch.sum(normalized_weights.unsqueeze(1) * neighbor_feats_stack, dim=0)
                else:
                     final_feats[idx_in_final] = torch.zeros_like(original_feats[0])
            else:
                final_feats[idx_in_final] = torch.zeros_like(original_feats[0])

            # Cache the new feature for later interpolation.
            current_voxel_map[final_c_tuple] = final_feats[idx_in_final] # Use final_feats[idx_in_final] as the computed feature


    # Build the output sparse tensor.
    output_sparse_tensor = sp.SparseTensor(final_feats, unique_final_coords.int())


    return output_sparse_tensor


def interpolate_sparse_tensor_to_half_voxels_4(sparse_tensor, sp):
    # Extract the original coordinates and features.
    # original_coords is expected to be int32 with shape [N_orig, 4].
    original_coords = sparse_tensor.coords # [N_orig, 4]
    # original_feats is expected to have shape [N_orig, 8].
    original_feats = sparse_tensor.feats       # [N_orig, 8]

    # Map original voxel coordinates to features for fast lookup.
    original_feature_map = {tuple(c.tolist()): original_feats[i] for i, c in enumerate(original_coords)}

    # Extract the original integer spatial coordinates.
    spatial_coords_orig = original_coords[:, 1:4] # [N_orig, 3] (int32)

    # Find the minimum coordinate on each axis.
    min_spatial_orig = spatial_coords_orig.min(dim=0).values  # [3] (int32)

    # Transform each voxel into the enlarged space: (spatial - min + 1) * 2.
    shifted_spatial_orig = spatial_coords_orig - min_spatial_orig # Still int32
    shifted_spatial_plus_one_orig = shifted_spatial_orig + 1 # Still int32
    # This is the upper corner of each voxel's enlarged region.
    scaled_spatial_orig = shifted_spatial_plus_one_orig * 2 # [N_orig, 3] (int32)

    # Extract the batch column.
    batch_col_orig = original_coords[:, 0:1] # [N_orig, 1] (int32)

    # Build full coordinates in the enlarged space.
    scaled_full_coords = torch.cat([batch_col_orig, scaled_spatial_orig], dim=1) # [N_orig, 4]


    # Generate the eight corners around each scaled point.
    # Each offset component is 0 or 1.
    offsets_3d = []
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                offsets_3d.append([dx, dy, dz])

    offsets_3d = torch.tensor(offsets_3d, dtype=scaled_spatial_orig.dtype, device=scaled_spatial_orig.device) # (8, 3)

    # Expand the scaled coordinates and subtract all eight offsets.
    # scaled_spatial_orig: (N_orig, 3) -> (N_orig, 1, 3)
    # offsets_3d: (8, 3) -> (1, 8, 3)
    all_8_spatial_coords_around_scaled = scaled_spatial_orig.unsqueeze(1) - offsets_3d.unsqueeze(0) # (N_orig, 8, 3)

    # Add the batch column.
    # batch_col_orig: (N_orig, 1) -> (N_orig, 1, 1) -> (N_orig, 8, 1)
    batch_col_expanded = batch_col_orig.unsqueeze(1).repeat(1, 8, 1) # (N_orig, 8, 1)

    # Concatenate batch and spatial coordinates.
    all_8_coords_around_scaled = torch.cat([batch_col_expanded, all_8_spatial_coords_around_scaled], dim=2) # (N_orig, 8, 4)

    # Flatten the 8N coordinates.
    all_8_coords_flat = all_8_coords_around_scaled.reshape(-1, 4) # (N_orig * 8, 4) # Use reshape


    # Deduplicate the final coordinates.
    unique_final_coords, original_indices_in_8N = torch.unique(all_8_coords_flat, return_inverse=True, dim=0) # [N_final, 4]
    # N_final is the number of unique points.

    # Map each unique coordinate to its index.
    unique_coords_to_index = {tuple(c.tolist()): i for i, c in enumerate(unique_final_coords)}


    # Initialize the output features.
    final_feats = torch.zeros((unique_final_coords.size(0), original_feats.size(1)), dtype=original_feats.dtype, device=original_feats.device)


    # Copy original voxel features to their scaled coordinates.
    scaled_full_coords_list = scaled_full_coords.tolist()
    for i_orig, c_scaled_full in enumerate(scaled_full_coords_list):
        # Find the scaled coordinate in the unique coordinate set.
        if tuple(c_scaled_full) in unique_coords_to_index:
            idx_in_final = unique_coords_to_index[tuple(c_scaled_full)]
            # Assign the original feature.
            final_feats[idx_in_final] = original_feats[i_orig]


    # Compute features for generated voxels.
    # Generated voxels are not part of scaled_full_coords.
    scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}

    # Define the 26 offsets in a 3x3x3 neighborhood, excluding the center.
    neighborhood_offsets_3d_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if not (dx == 0 and dy == 0 and dz == 0):
                    neighborhood_offsets_3d_26.append([dx, dy, dz])
    neighborhood_offsets_3d_26 = torch.tensor(neighborhood_offsets_3d_26, dtype=torch.int32, device=unique_final_coords.device) # (26, 3)

    # Interpolate generated voxel features.
    epsilon = 1e-6 # Avoid division by zero

    # original_feature_map provides original voxel features.

    # Cache scaled original voxels and newly computed features.
    current_voxel_map = {tuple(scaled_full_coords[i].tolist()): original_feats[i] for i in range(scaled_full_coords.size(0))}

    # Interpolate points that are not scaled original voxels.
    for idx_in_final, final_c in enumerate(unique_final_coords):
        final_c_tuple = tuple(final_c.tolist())

        if final_c_tuple not in scaled_full_coords_set: # Generated point
            # Find nearby original voxels for interpolation.

            neighbor_weights_and_feats = [] # Store (raw_weight, neighbor_feat) for found original neighbors

            # Approximate the corresponding position in the original space.
            final_c_spatial_float = final_c[1:4].float()
            # Adjust this inverse transform if the coordinate convention changes.
            approx_orig_spatial_float = (final_c_spatial_float / 2.0 - 1.0) + min_spatial_orig.float() # Use float for the inverse transform

            # Define a 3x3x3 neighborhood in the original space.
            orig_neighborhood_offsets_3d = []
            for dx in [-1, 0, 1]:
                 for dy in [-1, 0, 1]:
                     for dz in [-1, 0, 1]:
                         orig_neighborhood_offsets_3d.append([dx, dy, dz])
            orig_neighborhood_offsets_3d = torch.tensor(orig_neighborhood_offsets_3d, dtype=torch.int32, device=original_coords.device) # (27, 3), including the center


            # approx_orig_spatial_int_center = approx_orig_spatial_float.round().to(torch.int32)
            # # Iterate over original voxels near the approximate center.
            # for orig_offset_3d in orig_neighborhood_offsets_3d:
            #     neighbor_orig_spatial_int = approx_orig_spatial_int_center + orig_offset_3d
            #     # Build the full neighbor coordinate in the same batch.
            #     neighbor_orig_full_c_list = [final_c[0].item()] + neighbor_orig_spatial_int.tolist()
            #     neighbor_orig_full_c_tuple = tuple(neighbor_orig_full_c_list)

            #     # Look up the neighbor in the original voxel map.
            #     if neighbor_orig_full_c_tuple in original_feature_map:
            #          neighbor_feat = original_feature_map[neighbor_orig_full_c_tuple]

            #          # Avoid mixing distances from different coordinate spaces.

            # Interpolate in the scaled 3x3x3 neighborhood.
            # Use transformed original voxels and measure distance in scaled space.

            # Build lookup structures for scaled original voxels.
            scaled_full_coords_set = {tuple(c.tolist()) for c in scaled_full_coords}
            scaled_to_original_feature_map = {tuple(scaled_full_coords[i].tolist()): original_feats[i] for i in range(scaled_full_coords.size(0))}


            # Scan the scaled 3x3x3 neighborhood around the generated point.
            final_c_spatial_int = final_c[1:4]
            for offset_3d in neighborhood_offsets_3d_26: # 26 offsets
                # Compute the neighbor coordinate in scaled space.
                neighbor_scaled_spatial_int = final_c_spatial_int + offset_3d
                # Build the full batch and spatial coordinate.
                neighbor_scaled_full_c_list = [final_c[0].item()] + neighbor_scaled_spatial_int.tolist()
                neighbor_scaled_full_c_tuple = tuple(neighbor_scaled_full_c_list)

                # Look up scaled original and previously generated features.
                if neighbor_scaled_full_c_tuple in current_voxel_map:
                    neighbor_feat = current_voxel_map[neighbor_scaled_full_c_tuple]

                    # Measure spatial distance in scaled space.

                    neighbor_scaled_spatial_tensor = torch.tensor(neighbor_scaled_full_c_list[1:], dtype=torch.float32, device=final_c.device)
                    dist = torch.norm(final_c[1:4].float() - neighbor_scaled_spatial_tensor)

                    raw_weight = 1.0 / (dist + epsilon)

                    # Boost transformed original-voxel weights.
                    if neighbor_scaled_full_c_tuple in scaled_full_coords_set:
                        # Original transformed voxel.
                        amplified_weight = raw_weight * 10.0
                    else:
                        # Previously interpolated voxel.
                        amplified_weight = raw_weight

                    neighbor_weights_and_feats.append((amplified_weight, neighbor_feat))

            # Compute the normalized weighted mean.
            if neighbor_weights_and_feats: # Neighbors found
                raw_weights = torch.tensor([item[0] for item in neighbor_weights_and_feats], dtype=original_feats.dtype, device=original_feats.device)
                neighbor_feats_stack = torch.stack([item[1] for item in neighbor_weights_and_feats])

                sum_raw_weights = torch.sum(raw_weights)

                if sum_raw_weights > 0:
                    normalized_weights = raw_weights / sum_raw_weights
                    final_feats[idx_in_final] = torch.sum(normalized_weights.unsqueeze(1) * neighbor_feats_stack, dim=0)
                else:
                     final_feats[idx_in_final] = torch.zeros_like(original_feats[0])
            else:
                final_feats[idx_in_final] = torch.zeros_like(original_feats[0])

            # Cache the new feature for later interpolation.
            current_voxel_map[final_c_tuple] = final_feats[idx_in_final] # Use final_feats[idx_in_final] as the computed feature


    # Build the output sparse tensor.
    output_sparse_tensor = sp.SparseTensor(final_feats, unique_final_coords.int())


    return output_sparse_tensor

# Voxel-filling upsampling.
def upsample(sparse_tensor, sp):
    # Extract coordinates and features.
    new_coords = sparse_tensor.coords # [N, 4]
    feats = sparse_tensor.feats       # [N, 8]

    # Extract spatial coordinates.
    spatial = new_coords[:, 1:4]

    # Use the minimum on each axis as the origin.
    origin = spatial.min(dim=0).values.to(torch.int32)  # [3]


    # Shift coordinates to the origin.
    shifted = spatial - origin

    # Add one before scaling.
    shifted = shifted + 1


    # Double each spatial axis.
    scaled = shifted * 2

    # Convert to int32.
    scaled_int = scaled.round().to(torch.int32)

    # Optionally restore the origin.
    # final_spatial_scaled = scaled + origin * 1  # Restore the origin
    # final_spatial_scaled = scaled            # Keep shifted coordinates

    # Define the eight corner offsets.
    # 000, 100, 010, 001, 110, 101, 011, 111
    # Subtract 0 or 1 from each scaled coordinate.
    # scaled is a floating-point tensor with shape (N, 3).
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=scaled.dtype, device=scaled.device)

    # Expand the scaled coordinates and apply offsets.
    # scaled: (N, 3) -> (N, 1, 3) -> (N, 8, 3)
    # offsets: (8, 3) -> (1, 8, 3)
    new_spatial_coords = scaled.unsqueeze(1) - offsets.unsqueeze(0) # (N, 8, 3)

    # Convert to int32.
    new_spatial_coords_int = new_spatial_coords.round().to(torch.int32) # (N, 8, 3)
    #new_spatial_coords_int = new_spatial_coords_int - 1 +origin
    new_spatial_coords_int = new_spatial_coords_int - 1 

    # Expand the batch column from (N, 1) to (N, 8, 1).
    batch_col_expanded = new_coords[:, 0:1].unsqueeze(1).repeat(1, 8, 1) # (N, 8, 1)

    # Concatenate batch and spatial coordinates.
    final_coords_expanded = torch.cat([batch_col_expanded, new_spatial_coords_int], dim=2) # (N, 8, 4)

    # Flatten the coordinates.
    final_coords = final_coords_expanded.view(-1, 4) # (N * 8, 4)

    # Replicate the features.
    # feats: (N, 8) -> (N, 1, 8) -> (N, 8, 8)
    feats_expanded = feats.unsqueeze(1).repeat(1, 8, 1) # (N, 8, 8)

    # Flatten the feature tensor.
    final_feats = feats_expanded.view(-1, feats.size(1)) # (N * 8, 8)

    

    # Build the output sparse tensor.
    output_sparse_tensor = sp.SparseTensor(final_feats, final_coords.int())



    return output_sparse_tensor


def upsample_2(sparse_tensor):
    # Extract coordinates and features.
    new_coords = sparse_tensor.coords # [N, 4]
    feats = sparse_tensor.feats       # [N, 8]

    # Extract spatial coordinates.
    spatial = new_coords[:, 1:4]

    # Use the minimum on each axis as the origin.
    origin = spatial.min(dim=0).values.to(torch.int32)  # [3]


    # Shift coordinates to the origin.
    shifted = spatial - origin

    # Add one before scaling.
    shifted = shifted + 1


    # Double each spatial axis.
    scaled = shifted * 2

    # Convert to int32.
    scaled_int = scaled.round().to(torch.int32)

    # Optionally restore the origin.
    # final_spatial_scaled = scaled + origin * 1  # Restore the origin
    # final_spatial_scaled = scaled            # Keep shifted coordinates

    # Define the eight corner offsets.
    # 000, 100, 010, 001, 110, 101, 011, 111
    # Subtract 0 or 1 from each scaled coordinate.
    # scaled is a floating-point tensor with shape (N, 3).
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=scaled.dtype, device=scaled.device)

    # Expand the scaled coordinates and apply offsets.
    # scaled: (N, 3) -> (N, 1, 3) -> (N, 8, 3)
    # offsets: (8, 3) -> (1, 8, 3)
    new_spatial_coords = scaled.unsqueeze(1) - offsets.unsqueeze(0) # (N, 8, 3)

    # Convert to int32.
    new_spatial_coords_int = new_spatial_coords.round().to(torch.int32) # (N, 8, 3)
    #new_spatial_coords_int = new_spatial_coords_int - 1 +origin
    new_spatial_coords_int = new_spatial_coords_int - 1 

    # Expand the batch column from (N, 1) to (N, 8, 1).
    batch_col_expanded = new_coords[:, 0:1].unsqueeze(1).repeat(1, 8, 1) # (N, 8, 1)

    # Concatenate batch and spatial coordinates.
    final_coords_expanded = torch.cat([batch_col_expanded, new_spatial_coords_int], dim=2) # (N, 8, 4)

    # Flatten the coordinates.
    final_coords = final_coords_expanded.view(-1, 4) # (N * 8, 4)

    # Replicate the features.
    # feats: (N, 8) -> (N, 1, 8) -> (N, 8, 8)
    feats_expanded = feats.unsqueeze(1).repeat(1, 8, 1) # (N, 8, 8)

    # Flatten the feature tensor.
    final_feats = feats_expanded.view(-1, feats.size(1)) # (N * 8, 8)

    return final_coords
