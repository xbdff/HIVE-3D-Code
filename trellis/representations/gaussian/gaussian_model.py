import json
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d
import open3d as o3d
from scipy.spatial.transform import Rotation
import math



class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
        
    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        #scale = self._scaling.detach().cpu().numpy()  # Store raw log scaling.
        scale = (self._scaling + self.scale_bias).detach().cpu().numpy()  # Store the reversible intermediate value.
        #scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            #rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            #rotation = np.matmul(transform, rotation)
            #rotation = utils3d.numpy.matrix_to_quaternion(rotation)
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_for_open3d(self, path, transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        """
        Save an Open3D-compatible PLY after opacity filtering.
        """
        # Convert stored opacity logits to [0, 1].
        opacity = self.get_opacity.detach().cpu().numpy().flatten()
        
        # Apply the opacity threshold.
        threshold = 0.1
        mask = opacity > threshold
        
        # Filter positions.
        xyz = self.get_xyz.detach().cpu().numpy()[mask]
        
        # Filter colors.
        f_dc = self._features_dc.detach().cpu().numpy()[mask]
        # Remove the singleton dimension.
        colors = f_dc.squeeze(1)
        # Map colors to [0, 1].
        colors = 1 / (1 + np.exp(-colors))
        
        # Transform filtered points.
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
        
        # Build the Open3D point cloud.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as standard PLY.
        o3d.io.write_point_cloud(path, pcd)
        
        print(f"Saved Open3D-compatible PLY to {path}")


    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)

            # Convert quaternions to matrices, transform, then convert back.
            #rotation_matrix = utils3d.numpy.quaternion_to_matrix(rots)
            #rotation_matrix = np.matmul(transform, rotation_matrix)
            #rots = utils3d.numpy.matrix_to_quaternion(rotation_matrix)
            #rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            #rotation = np.matmul(rotation, transform)
            #rotation = utils3d.numpy.matrix_to_quaternion(rotation)
            
        
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.tensor(scales, dtype=torch.float, device=self.device)
        #scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        #adjusted = torch.square(scales) - self.mininum_kernel_size ** 2
        #adjusted = torch.clamp(adjusted, min=1e-8)
        self._scaling = scales - self.scale_bias
        #self._scaling = self.inverse_scaling_activation(torch.sqrt(adjusted)) - self.scale_bias
        #self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]
        
    def tensor_to_list(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().tolist()
        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()
        return tensor
    

    def get_serializable_params(self):
        params = {
            # Initialization parameters.
            **self.init_params,
            # Dynamic parameters.
            "active_sh_degree": self.active_sh_degree,
            "aabb": self.tensor_to_list(self.aabb),
            # Optional PLY path or metadata.
            "metadata": {
                "num_gaussians": len(self._xyz) if self._xyz is not None else 0,
                "device": self.device,
            }
        }
        return params
    


    def save_to_json(self, json_path):
        params = self.get_serializable_params()
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=4)


            
    @classmethod
    def load_from_json(cls, json_path, device='cuda'):
        with open(json_path, 'r') as f:
            params = json.load(f)
        # Reconstruct the object.
        gaussian = cls(
            aabb=params['aabb'],
            sh_degree=params['sh_degree'],
            mininum_kernel_size=params['mininum_kernel_size'],
            scaling_bias=params['scaling_bias'],
            opacity_bias=params['opacity_bias'],
            scaling_activation=params['scaling_activation'],
            device=device
        )
        gaussian.active_sh_degree = params['active_sh_degree']
        return gaussian
    

    def scale_gaussian(self, scale_factor=1.0, centroid=None):
        """Scale Gaussian positions and sizes around a centroid.

        Args:
            scale_factor (float or torch.Tensor): Positive scale factor.
            centroid (torch.Tensor, optional): Center of shape [1, 3] or [3].
                Uses the point-cloud mean when omitted.
        
        Returns:
            The modified Gaussian object.
        """
        # Validate required parameters.
        assert self._xyz is not None, "Gaussian object must have xyz parameters"
        assert scale_factor > 0, "Scale factor should be positive"

        # Convert the scale factor to the model device.
        if not isinstance(scale_factor, torch.Tensor):
            scale_factor = torch.tensor(
                scale_factor, 
                dtype=torch.float32, 
                device=self.device
            )
        else:
            scale_factor = scale_factor.to(self.device)

        # Determine the scaling center.
        if centroid is None:
            # Use the mean position.
            curr_centroid = self._xyz.mean(dim=0, keepdim=True) # [1, 3]
        else:
            # Normalize the provided shape to [1, 3].
            curr_centroid = centroid.to(self.device).reshape(1, 3)

        # Scale positions around the centroid.
        self._xyz = curr_centroid + scale_factor * (self._xyz - curr_centroid)

        # Scale intrinsic Gaussian sizes.
        new_scaling_linear = self.scaling_activation(self._scaling) * scale_factor
        self._scaling = self.inverse_scaling_activation(new_scaling_linear)

        return self
    

    def scale_gaussian_xyz(self, scale_factor=0.5):
        """Scale Gaussian positions and sizes.
    
        Args:
            gaussian: Gaussian object with `_xyz` and `_scaling`.
            scale_factor (float): Positive scale factor.
        
        Returns:
            The modified Gaussian object.
        """
        # Validate required parameters.
        assert self._xyz is not None, "Gaussian object must have xyz parameters"
        assert scale_factor > 0, "Scale factor should be positive"

        # Use the minimum corner as the origin.
        min_xyz = torch.min(self._xyz, dim=0).values  # Shape: [3].

        # Scale positions.
        delta = self._xyz - min_xyz.unsqueeze(0)
        new_xyz = min_xyz.unsqueeze(0) + scale_factor * delta
        self._xyz = new_xyz


        return self

    def merge_gaussians(self, other):
        """
        Merge another Gaussian model into this model.
        
        Args:
            other: Compatible Gaussian model.
        """
        # Validate compatibility.
        if self.device != other.device:
            raise ValueError("Device mismatch: current model is on {}, other model is on {}".format(self.device, other.device))
        
        if self.active_sh_degree != other.active_sh_degree:
            raise ValueError("Spherical harmonic degree mismatch: current {}, other {}".format(
                self.active_sh_degree, other.active_sh_degree))
        
        # Merge attribute tensors.
        self._xyz = torch.cat([self._xyz, other._xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, other._features_dc], dim=0)
        
        if self.active_sh_degree > 0:
            self._features_rest = torch.cat([self._features_rest, other._features_rest], dim=0)
        
        self._scaling = torch.cat([self._scaling, other._scaling], dim=0)
        self._rotation = torch.cat([self._rotation, other._rotation], dim=0)
        self._opacity = torch.cat([self._opacity, other._opacity], dim=0)
        
        # Expand the AABB to include both models.
        combined_min = torch.minimum(
            self.aabb[:3], 
            other.aabb[:3]
        )
        combined_max = torch.maximum(
            self.aabb[3:], 
            other.aabb[3:]
        )
        self.aabb = torch.cat([combined_min, combined_max], dim=0)

    def apply_rigid_transform_to_gaussians(self, transform_matrix):
        """
        Apply a 4x4 rigid transform to the Gaussian model.
        
        Args:
            gaussians: Gaussian model.
            transform_matrix: NumPy array or torch tensor of shape [4, 4].
        
        Returns:
            The transformed Gaussian model.
        """
        # Move the transform to the model device.
        if isinstance(transform_matrix, np.ndarray):
            transform_matrix = torch.tensor(transform_matrix, 
                                        dtype=torch.float32, 
                                        device=self.device)
        elif isinstance(transform_matrix, torch.Tensor):
            transform_matrix = transform_matrix.to(self.device)
        else:
            raise TypeError("transform_matrix must be a NumPy array or torch.Tensor")
        

      

        # Extract rotation and translation.
        rotation_matrix = transform_matrix[:3, :3]    # 3x3 rotation.
        translation_vector = transform_matrix[:3, 3]  # 3D translation.
        
        # Transform positions of shape [N, 3].
        current_xyz = self.get_xyz

        
        # Apply new_xyz = (R @ xyz^T)^T + T.
        transformed_xyz = torch.mm(current_xyz, rotation_matrix.T) + translation_vector
        self.from_xyz(transformed_xyz)
        
       
        
        # Convert the rotation matrix to a quaternion.
        rotation_scipy = Rotation.from_matrix(rotation_matrix.cpu().numpy())
        transform_quat = rotation_scipy.as_quat()  # Format: (x, y, z, w).
        
        # Move the quaternion to the model device.
        transform_quat = torch.tensor(transform_quat, 
                                    dtype=torch.float32, 
                                    device=self.device)
        
        # Reorder to (w, x, y, z).
        transform_quat = torch.tensor([
            transform_quat[3],  # w
            transform_quat[0],  # x
            transform_quat[1],  # y
            transform_quat[2]   # z
        ], device=self.device)
        
        # Current rotations have shape [N, 4] in (w, x, y, z) order.
        current_rotation = self.get_rotation
        
        # Quaternion multiplication.
        def quaternion_multiply(q1, q2):
            """Return the quaternion product q1 * q2."""
            w1, x1, y1, z1 = q1.unbind(dim=-1)
            w2, x2, y2, z2 = q2.unbind(dim=-1)
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            
            return torch.stack((w, x, y, z), dim=-1)
        
        # Left-multiply each Gaussian by the global rotation.
        transformed_rotation = quaternion_multiply(
            transform_quat.unsqueeze(0),  # Expand to [1, 4].
            current_rotation
        )
        
        # Normalize quaternions.
        norms = torch.norm(transformed_rotation, dim=1, keepdim=True)
        transformed_rotation = transformed_rotation / norms
        
        self.from_rotation(transformed_rotation)
        
        # Rigid transforms preserve scale and other attributes.
        
        return self



    def get_mean_distance_to_center(self, robust: bool = True) -> torch.Tensor:
        """
        Return the mean distance to the center for opacity above 0.1.
        """
        # Get activated opacity.
        opacity = self.get_opacity 
        mask = (opacity > 0.1).view(-1) # Flatten to a boolean mask.
        
        # Filter visible points.
        points_all = self.get_xyz
        points = points_all[mask]
        
        # Return a small value when no point is visible.
        if points.shape[0] == 0:
            return torch.tensor(1e-7, device=points_all.device, dtype=points_all.dtype)

        # Compute the filtered center.
        center = torch.mean(points, dim=0)
        
        # Compute offset vectors.
        diff = points - center
        
        # Compute distances to the center.
        if diff.is_cuda:
            sq_dist = torch.sum(diff * diff, dim=1)
            distances = torch.sqrt(sq_dist)
        else:
            distances = torch.linalg.norm(diff, dim=1)
        
        # Optional robust statistic.
        if robust:
            mean_distance = torch.median(distances)
        else:
            mean_distance = torch.mean(distances)
        
        return torch.clamp_min(mean_distance, 1e-7)
    


    def save_point_cloud_ply(self, path, transform=None):
        """
        Saves the Gaussian centers and their principal colors as a standard point cloud
        in PLY format, which is easily readable by libraries like Open3D.

        Args:
            path (str): The path to save the PLY file.
            transform (list, optional): A 3x3 transformation matrix to apply to the points.
        """

        print(f"Exporting as a standard point cloud to {path}...")
        
        # 1. Get the geometry: the XYZ centers of the Gaussians
        #    This is the core of the point cloud.
        xyz = self.get_xyz.detach().cpu().numpy()

        # 2. Get the color: Use the DC component of the Spherical Harmonics
        #    This represents the base color of each Gaussian.
        #    The original features are in SH space, we need to convert them to RGB.
        #    The DC component (first SH coefficient) needs to be scaled and shifted.
        #    The formula is typically: RGB = 0.5 * SH_DC + 0.5
        features_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        
        # C0 constant from SH equations (not always needed, but good practice)
        C0 = 0.28209479177387814 
        colors = features_dc * C0 + 0.5
        colors = np.clip(colors, 0.0, 1.0) # Ensure colors are in the [0, 1] range

        # 3. (Optional) Apply transformation if provided
        if transform is not None:
            transform_matrix = np.array(transform)
            xyz = xyz @ transform_matrix.T # Apply transformation matrix

        # 4. Use Open3D for robust and standard PLY saving
        try:
            import open3d as o3d
            
            # Create an Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # Open3D expects colors in the [0, 1] range, which we already ensured
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Write the PLY file. `write_ascii=True` makes it human-readable.
            o3d.io.write_point_cloud(path, pcd, write_ascii=True)
            
            print(f"Successfully saved {len(xyz)} points to {path}")

        except ImportError:
            print("Warning: Open3D is not installed. Falling back to a manual PLY writer.")
            print("Manual writer will not be as robust. Please install Open3D: pip install open3d")
            # Fallback manual writer if Open3D is not available
            self._manual_save_point_cloud_ply(path, xyz, colors)

    def _manual_save_point_cloud_ply(self, path, xyz, colors):
        """A simple fallback for writing PLY if Open3D is not available."""
        # Convert colors to 0-255 uint8 range for standard PLY
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        num_points = len(xyz)
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(num_points):
                p = xyz[i]
                c = colors_uint8[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
