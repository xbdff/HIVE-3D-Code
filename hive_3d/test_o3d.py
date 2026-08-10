import open3d as o3d
import open3d.core as o3c
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import torch

def colored_point_cloud_registration(source_path, target_path):
    """
    Register colored point clouds and return the transformation matrix.
    
    Args:
        source_path: Source point-cloud path.
        target_path: Target point-cloud path.
    
    Returns:
        A 4x4 NumPy transformation matrix.
    """
    # Load point clouds.
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    print(len(target.points))
    """voxel_size = estimate_voxel_size(source, len(target.points))  # Estimate voxel size for 10,000 target points.
    source = color_aware_downsample(source, voxel_size)"""
    
    # Require color data for colored ICP.
    if not source.has_colors() or not target.has_colors():
        raise ValueError("Colored ICP requires point-cloud colors.")
    
    # Registration settings.
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    
    # Multi-scale registration.
    for scale in range(3):
        iter_count = max_iter[scale]
        radius = voxel_radius[scale]
        
        # Downsample point clouds.
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        
        # Estimate normals.
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        
        # Run colored ICP.
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, 
            target_down, 
            radius, 
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-2,
                relative_rmse=1e-2,
                max_iteration=iter_count
            )
        )
        
        
        # Update the transformation.
        current_transformation = result_icp.transformation
        print(result_icp)
        print(current_transformation)
    
    return current_transformation

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=200))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_path, target_path,voxel_size = 0.0156):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size = 0.0156):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.90),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_local_refinement(source, target, result_ransac, voxel_size):
    # ICP uses a tighter threshold than RANSAC.
    distance_threshold = voxel_size * 1.5
    
    print(":: Running ICP refinement...")
    
    # Use robust point-to-plane ICP for shape differences.
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        # Point-to-plane works well for planar surfaces.
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            # Tukey loss suppresses large mismatches.
            o3d.pipelines.registration.TukeyLoss(k=0.1) 
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return result

def color_aware_downsample(pcd, voxel_size):
    """
    Downsample while preserving each voxel's mean color.
    """
    # Read point and color arrays.
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Compute voxel coordinates.
    voxel_coords = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels.
    unique_voxels, inverse_indices, counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True)
    
    # Build downsampled arrays.
    down_points = []
    down_colors = []
    
    for i in range(len(unique_voxels)):
        # Select points in this voxel.
        indices = np.where(inverse_indices == i)[0]
        
        # Compute the voxel centroid.
        voxel_center = points[indices].mean(axis=0)
        down_points.append(voxel_center)
        
        # Compute the mean color.
        voxel_color = colors[indices].mean(axis=0)
        down_colors.append(voxel_color)
    
    # Build the output point cloud.
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(np.array(down_points))
    down_pcd.colors = o3d.utility.Vector3dVector(np.array(down_colors))
    
    return down_pcd

def estimate_voxel_size(pcd, target_points):
    """
    Estimate a voxel size for the target point count.

    Args:
        pcd: Input point cloud.
        target_points: Desired point count.

    Returns:
        Estimated voxel size.
    """
    # Read points.
    points = np.asarray(pcd.points)
    n_points = len(points)
    
    # Skip downsampling when already below the target.
    if target_points >= n_points:
        return 0.0
    
    # Compute bounding-box dimensions.
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bbox_size = max_bound - min_bound
    
    # Compute mean point density.
    volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
    
    # Handle planar or linear point clouds.
    if volume < 1e-9:
        # Use the largest extent.
        max_dim = np.max(bbox_size)
        if max_dim < 1e-9:
            return 0.0  # Too small to downsample.
        
        # Estimate linear density.
        density = n_points / max_dim
        # Estimate voxel size from target spacing.
        return max_dim * target_points / n_points
    
    # Compute point density.
    density = n_points / volume
    
    # Compute target density.
    target_density = target_points / volume
    
    # Density ratio.
    volume_ratio = density / target_density
    
    # Derive voxel size from the density ratio.
    voxel_size = (1 / density) ** (1/3) * (n_points / target_points) ** (1/3)
    
    # Equivalent to volume_ratio ** (1/3).
    
    return max(voxel_size, 1e-9)  # Keep the result positive.



def apply_4x4_transform_and_save(
    input_path: str,
    output_path: str,
    transform_matrix: np.ndarray = np.eye(4),
) -> None:
    """
    Apply a 4x4 rigid transform to a point cloud and save it as PLY.

    Args:
        input_path (str): Input point-cloud path.
        output_path (str): Output PLY path.
        transform_matrix (np.ndarray): 4x4 rigid transform [R | t].
    """
    # Validate the transform.
    if transform_matrix.shape != (4, 4):
        raise ValueError("The transformation matrix must be 4x4.")
    
    # Extract rotation and translation.
    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = transform_matrix[:3, 3]
    # Load the point cloud.
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        raise ValueError("The point-cloud file is empty or invalid.")
    # Extract Nx3 point coordinates.
    points = np.asarray(pcd.points)  # shape=(N, 3)
    # Apply R @ p + t.
    transformed_points = (rotation_matrix @ points.T).T + translation_vector
    # Update points.
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    # Rotate normals if present.
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        transformed_normals = (rotation_matrix @ normals.T).T  # Normals rotate without translation.
        pcd.normals = o3d.utility.Vector3dVector(transformed_normals)
    # Save as PLY.
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved transformed point cloud to: {output_path}")


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result










def get_semantic_constrained_correspondences(source_pcd, target_pcd, source_fpfh, target_fpfh, results, feature_threshold=0.5):
    """
    Find FPFH inlier pairs constrained by patch correspondences.
    """
    # Convert FPFH features to NumPy [N, 33].
    src_feat = np.asarray(source_fpfh.data).T
    tar_feat = np.asarray(target_fpfh.data).T
    
    # Map each global point to its patch.
    corres = []
    current_src_offset = 0
    current_dst_offset = 0
    
    print(":: Matching features within semantic patches...")
    
    for item in results:
        src_pts = item['src_data']['points']
        dst_pts = item['dst_data']['points']
        
        if src_pts is None or dst_pts is None:
            continue
            
        n_src = len(src_pts)
        n_dst = len(dst_pts)
        
        # Extract features for this patch.
        patch_src_feat = src_feat[current_src_offset : current_src_offset + n_src]
        patch_tar_feat = tar_feat[current_dst_offset : current_dst_offset + n_dst]
        
        # Find nearest features within the patch.
        for i in range(n_src):
            diff = patch_tar_feat - patch_src_feat[i]
            dist = np.linalg.norm(diff, axis=1)
            best_match_idx = np.argmin(dist)
            
            # Record the best global index pair.
            corres.append([current_src_offset + i, current_dst_offset + best_match_idx])
            
        current_src_offset += n_src
        current_dst_offset += n_dst
        
    return o3d.utility.Vector2iVector(np.array(corres))



def execute_semantic_global_registration(src_coords, dst_coords, results, voxel_size=0.0156):
    """
    Compute the transform that aligns moving dst_coords to fixed src_coords.
    """
    
    # Build Open3D point clouds.
    def prepare_pcd(tensor):
        pcd = o3d.geometry.PointCloud()
        # Convert [N, 4] to [N, 3].
        pts = tensor[:, 1:] if tensor.shape[1] == 4 else tensor
        pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy().astype(np.float64))
        return pcd

    # Open3D moves source toward target, so dst is moving and src is fixed.
    moving_pcd = prepare_pcd(dst_coords) # Moving source.
    fixed_pcd = prepare_pcd(src_coords)  # Fixed target.
    
    # Compute normals and FPFH features.
    radius_normal = voxel_size * 2
    moving_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fixed_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        moving_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        fixed_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # Extract semantically constrained feature pairs.
    moving_feat = np.asarray(moving_fpfh.data).T
    fixed_feat = np.asarray(fixed_fpfh.data).T
    corres_pairs = []

    print(":: Matching FPFH features within semantic patch constraints (Mapping: DST -> SRC)...")
    for item in results:
        # Get local indices for this patch in both clouds.
        l_fixed_idx = item['src_data']['indices']
        l_moving_idx = item['dst_data']['indices']
        
        if l_fixed_idx is None or l_moving_idx is None or len(l_fixed_idx) == 0 or len(l_moving_idx) == 0:
            continue
            
        # Extract patch features.
        patch_fixed_feat = fixed_feat[l_fixed_idx.cpu().numpy()]
        patch_moving_feat = moving_feat[l_moving_idx.cpu().numpy()]
        
        # Match each moving dst point to the best fixed src point.
        for i, moving_local_idx in enumerate(l_moving_idx):
            # Compare this dst feature with all src features in the patch.
            diff = patch_fixed_feat - patch_moving_feat[i]
            dist = np.linalg.norm(diff, axis=1)
            best_match_in_patch_idx = np.argmin(dist)
            
            # Store [moving_index, fixed_index].
            corres_pairs.append([moving_local_idx.item(), l_fixed_idx[best_match_in_patch_idx].item()])

    if len(corres_pairs) < 3:
        print("[Error] Not enough correspondences found for RANSAC.")
        dummy_res = o3d.pipelines.registration.RegistrationResult()
        dummy_res.transformation = np.eye(4)
        return dummy_res

    # Run RANSAC registration.
    correspondences = o3d.utility.Vector2iVector(np.array(corres_pairs))
    distance_threshold = voxel_size * 1.5
    
    print(f":: Running RANSAC with {len(corres_pairs)} semantic pairs (DST -> SRC)...")
    
    # Pass moving source first and fixed target second.
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        moving_pcd, fixed_pcd, correspondences,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999)
    )
    
    print(f":: Registration Fitness: {result.fitness:.4f}")
    return result
