import numpy as np
import open3d as o3d
import teaserpp_python
import time

class TeaserRegistrator:
    def __init__(self, voxel_size=0.0156, noise_bound=None, estimate_scaling=False):
        self.voxel_size = voxel_size
        # Default the noise bound to the voxel size.
        self.noise_bound = noise_bound if noise_bound else voxel_size
        self.estimate_scaling = estimate_scaling
        
        # Configure the TEASER++ solver.
        self.params = teaserpp_python.RobustRegistrationSolver.Params()
        self.params.cbar2 = 1
        self.params.noise_bound = self.noise_bound
        self.params.estimate_scaling = self.estimate_scaling
        self.params.rotation_estimation_algorithm = \
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        self.params.rotation_gnc_factor = 1.4
        self.params.rotation_max_iterations = 100
        self.params.rotation_cost_threshold = 1e-12

    def preprocess(self, pcd):
        """Preprocess a point cloud and compute FPFH features."""
        # Downsample.
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals.
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH features.
        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return pcd_down, pcd_fpfh

    def find_correspondences(self, src_fpfh, tgt_fpfh):
        """Find initial feature correspondences for TEASER++."""
        # Find nearest neighbors in FPFH space.
        src_fpfh_arr = np.asarray(src_fpfh.data).T
        tgt_fpfh_arr = np.asarray(tgt_fpfh.data).T
        
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tgt_fpfh_arr)
        distances, indices = nn.kneighbors(src_fpfh_arr)
        
        # Pair each source point with its nearest target.
        corrs = np.concatenate([
            np.arange(len(indices)).reshape(-1, 1),
            indices.reshape(-1, 1)
        ], axis=1)
        
        return corrs
    

    def find_correspondences_mutual(self, src_fpfh, tgt_fpfh):
        """
        Compute mutual nearest-neighbor feature matches.
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        # Convert FPFH from 33xN to Nx33.
        src_fpfh_arr = np.asarray(src_fpfh.data).T
        tgt_fpfh_arr = np.asarray(tgt_fpfh.data).T
        
        # Search source to target.
        nn_tgt = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tgt_fpfh_arr)
        _, indices_ab = nn_tgt.kneighbors(src_fpfh_arr)
        indices_ab = indices_ab.flatten() # Source-to-target indices.

        # Search target to source.
        nn_src = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src_fpfh_arr)
        _, indices_ba = nn_src.kneighbors(tgt_fpfh_arr)
        indices_ba = indices_ba.flatten() # Target-to-source indices.

        # Keep mutual matches only.
        src_indices = np.arange(len(indices_ab))
        # Select mutually consistent indices.
        mutual_mask = (indices_ba[indices_ab] == src_indices)
        
        # Extract final pairs.
        final_src_idx = src_indices[mutual_mask]
        final_tgt_idx = indices_ab[mutual_mask]

        # Return an array of shape [M, 2].
        corrs = np.stack([final_src_idx, final_tgt_idx], axis=1)
        
        print(f":: One-way matches: {len(src_fpfh_arr)}; mutual matches: {len(corrs)}")
        return corrs
    


    def solve_registration(self, source_pcd, target_pcd):
        """
        Run preprocessing, matching, and robust registration.
        """
        # Preprocess.
        print(":: Preprocessing...")
        src_down, src_fpfh = self.preprocess(source_pcd)
        tgt_down, tgt_fpfh = self.preprocess(target_pcd)

        # Match features.
        print(":: Finding initial matches...")
        corrs = self.find_correspondences_mutual(src_fpfh, tgt_fpfh)
        
        # Extract matched 3D coordinates.
        src_pts = np.asarray(src_down.points)[corrs[:, 0]].T
        tgt_pts = np.asarray(tgt_down.points)[corrs[:, 1]].T

        # Solve with TEASER++.
        print(":: Running TEASER++...")
        solver = teaserpp_python.RobustRegistrationSolver(self.params)
        
        start = time.time()
        solver.solve(src_pts, tgt_pts)
        end = time.time()

        solution = solver.getSolution()
        
        # Package the result.
        T = np.eye(4)
        T[:3, :3] = solution.rotation * solution.scale
        T[:3, 3] = solution.translation

        return {
            "transformation": T,
            "scale": solution.scale,
            "time": end - start,
            "src_down": src_down,
            "tgt_down": tgt_down
        }
