import torch
from collections import defaultdict
from . import sparse as sp

def build_bvh_and_split(coords, feats, target_num_regions=8):
    # Convert input to a tensor.
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)

    # Compute spatial bounds.
    min_coords = torch.min(coords[:, 1:], dim=0).values
    max_coords = torch.max(coords[:, 1:], dim=0).values
    
    # Define a BVH node.
    class BVHNode:
        def __init__(self, bbox_min, bbox_max, voxel_indices):
            self.bbox_min = bbox_min
            self.bbox_max = bbox_max
            self.voxel_indices = voxel_indices
            self.left = None
            self.right = None
    
    # Build the BVH recursively.
    def split_node(node, depth):
        if len(node.voxel_indices) <= len(coords) // target_num_regions:
            return
            
        # Split along the longest axis.
        extents = node.bbox_max - node.bbox_min
        split_axis = torch.argmax(extents).item()
        
        # Use the midpoint.
        split_pos = (node.bbox_min[split_axis] + node.bbox_max[split_axis]) / 2
        
        # Partition voxels.
        left_indices = []
        right_indices = []
        for idx in node.voxel_indices:
            if coords[idx, split_axis + 1] < split_pos:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
                
        # Stop if the split is empty.
        if len(left_indices) == 0 or len(right_indices) == 0:
            return
            
        # Create child nodes.
        left_bbox_max = node.bbox_max.clone()
        left_bbox_max[split_axis] = split_pos
        node.left = BVHNode(node.bbox_min, left_bbox_max, left_indices)
        
        right_bbox_min = node.bbox_min.clone() 
        right_bbox_min[split_axis] = split_pos
        node.right = BVHNode(right_bbox_min, node.bbox_max, right_indices)
        
        # Recurse into children.
        split_node(node.left, depth + 1)
        split_node(node.right, depth + 1)
    
    # Build the root.
    root = BVHNode(min_coords, max_coords, list(range(len(coords))))
    split_node(root, 0)
    
    # Collect leaf voxels.
    def collect_leaves(node, leaves):
        if node.left is None and node.right is None:
            leaves.append(node)
        if node.left:
            collect_leaves(node.left, leaves)
        if node.right:
            collect_leaves(node.right, leaves)
            
    leaves = []
    collect_leaves(root, leaves)
    
    # Build one sparse tensor per leaf.
    sparse_tensors = []
    indices=[]
    for leaf in leaves:
        leaf_coords = coords[leaf.voxel_indices]
        leaf_feats = feats[leaf.voxel_indices]
        slat = sp.SparseTensor(leaf_feats, leaf_coords)
        sparse_tensors.append(slat) 
        indices.append(leaf.voxel_indices)
        
    return sparse_tensors,indices

 
