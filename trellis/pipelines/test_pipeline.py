from typing import *
import torch
import json
from os import PathLike
from PIL import Image
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from . import samplers
from ..modules import sparse as sp
from contextlib import contextmanager
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from ..modules.sparse.attention import SparseMultiHeadAttentionWithAttentionMap
from trellis.modules.bvh import build_bvh_and_split
from collections import deque
from ..models.sparse_structure_flow import IPAdapterSparseStructureFlowModel
from..models.ip_adapter_modules import LatentProjector
from .checkpoint_io import load_model_weights, resolve_hive_checkpoint_paths

class TestImageToSlatPipeline(TrellisImageTo3DPipeline):
    """
    Test pipeline for image-to-SLAT generation.

    Supports intermediate hooks and dynamic SLAT sampler replacement.
    """
    def __init__(
        self,
        models=None, 
        sparse_structure_sampler=None,
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model=None,
    ):
        # Initialize the parent when models are available.
        super().__init__(
            models,
            sparse_structure_sampler,
            slat_sampler,
            slat_normalization,
            image_cond_model,
        )
        # Test-only state.
        self.hook_handles = []       # Registered PyTorch hooks.


    @staticmethod
    def from_pretrained(
        path: str,
        hive_model: Union[str, PathLike[str]],
        **kwargs
    ) -> "TestImageToSlatPipeline":
        """
        Load pretrained models with sampler overrides and debug hooks.
        """
        # Create the base pipeline.
        parent_pipeline = super(TestImageToSlatPipeline, TestImageToSlatPipeline).from_pretrained(path)
        
        # Create this subclass and copy parent state.
        new_pipeline = TestImageToSlatPipeline()
        new_pipeline.__dict__ = parent_pipeline.__dict__
        args = parent_pipeline._pretrained_args
        
        # Override SLAT parameters from keyword arguments.
        slat_args = args['slat_sampler']['args'].copy()
        slat_args.update(kwargs.get('slat_sampler_args', {}))
        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**slat_args)
        
        new_pipeline.hook_handle_list = []
            
        # =====================================================================
        # Load two models from separate config and weight files.
        # =====================================================================
        print(f"Loading additional models...")
        checkpoint_paths = resolve_hive_checkpoint_paths(hive_model)

        # Load model A.
        print(f"  - Loading model A...")
        print(f"    - Config: {checkpoint_paths.denoiser_config}")
        print(f"    - Weights: {checkpoint_paths.denoiser_weights}")

        # Read model A configuration.
        with open(checkpoint_paths.denoiser_config, 'r') as f:
            model_a_config = json.load(f)

        # Instantiate model A.
        model_a = IPAdapterSparseStructureFlowModel(**model_a_config)
        # Load model A weights.
        load_model_weights(model_a, checkpoint_paths.denoiser_weights)

        # Load model B.
        print(f"  - Loading model B...")
        print(f"    - Config: {checkpoint_paths.projector_config}")
        print(f"    - Weights: {checkpoint_paths.projector_weights}")
        
        # Read model B configuration.
        with open(checkpoint_paths.projector_config, 'r') as f:
            model_b_config = json.load(f)

        # Instantiate model B.
        model_b = LatentProjector(**model_b_config)
        # Load model B weights.
        load_model_weights(model_b, checkpoint_paths.projector_weights)


        # Add both models to the pipeline.
        new_pipeline.models['denoiser'] = model_a
        new_pipeline.models['projection'] = model_b

        new_pipeline.models['denoiser'].eval()
        new_pipeline.models['projection'].eval()

        # Move models to the target device.
        device = parent_pipeline.device
        new_pipeline.models['denoiser'].to(device)
        new_pipeline.models['projection'].to(device)

        print("Additional models loaded.")
        # =====================================================================
            
        return new_pipeline
    
    def downsample_coords(self,coords, factor=(2,2,2), require_upsample=False):
        DIM = coords.shape[-1] - 1
        assert DIM == len(
            factor
        ), "Input coordinates must have the same dimension as the downsample factor."

        coord = list(coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i + 1] = coord[i + 1] // f

        MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])  # code [valid_voxel_num]
        code, idx = code.unique(return_inverse=True)
        new_coords = torch.stack(
            [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1,
        )
        print(new_coords.shape, idx.shape)
        if not require_upsample:
            return new_coords
        else:
            return new_coords, idx
        

    def upsample_coords(self, new_coords, orig_coords, factor=(2, 2, 2)):
        """
        Map downsampled coordinates back to original points and indices.

        Args:
            new_coords (Tensor): Downsampled coordinates [M, D+1].
            orig_coords (Tensor): Original coordinates [N, D+1].
            factor (tuple): Downsampling factor.

        Returns:
            Tuple[Tensor, Tensor]:
                - Matched original coordinates [K, D+1].
                - Original coordinate indices [K, 1].
        """
        device = new_coords.device
        DIM = new_coords.shape[-1] - 1  # Spatial dimensions.
        
        # Precompute downsampled original coordinates.
        precomputed_map = defaultdict(list)
        
        # Downsample original coordinates.
        orig_sample = orig_coords.clone()
        for i in range(DIM):
            orig_sample[:, i+1] = orig_coords[:, i+1] // factor[i]
        
        # Map downsampled coordinates to original values and indices.
        for idx, (sample_point, orig_point) in enumerate(zip(orig_sample, orig_coords)):
            # Use tuple keys.
            key = tuple(sample_point.tolist())
            # Store coordinates and indices.
            precomputed_map[key].append((orig_point.clone(), idx))
        
        # Collect matching original coordinates and indices.
        matched_points = []
        matched_indices = []
        
        # Scan requested coordinates.
        for point in new_coords:
            # Convert to a lookup key.
            key = tuple(point.tolist())
            
            # Gather all matches.
            if key in precomputed_map:
                # Retrieve coordinate/index pairs.
                for orig_point, idx in precomputed_map[key]:
                    matched_points.append(orig_point)
                    matched_indices.append(idx)
        
        # Return empty tensors when no points match.
        if not matched_points:
            empty_coords = torch.empty((0, DIM+1), device=device, dtype=torch.long)
            empty_indices = torch.empty((0,), device=device, dtype=torch.long)
            return empty_coords, empty_indices
        
        # Convert results to tensors.
        matched_points_tensor = torch.stack(matched_points, dim=0)
        matched_indices_tensor = torch.tensor(matched_indices, device=device, dtype=torch.long)
        
        return matched_points_tensor, matched_indices_tensor


    @torch.no_grad()
    def Image_to_Slat(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        preprocess_image: bool = True,
    ) -> dict:
        """
        A pipeline to generate Structured Latents.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return slat
    
    @torch.no_grad()
    def Slat_to_GS(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian'],
    ) -> dict:
        return self.decode_slat(slat, formats)


    def get_valid_patches(self, images: List[Image.Image]):
        patch_size = 14
        valid_tokens_list = []
        for image in images:
            image_np = np.array(image)
            mask = image_np.sum(-1) != 0
            H, W = mask.shape
            h_blocks = H // patch_size
            w_blocks = W // patch_size

            # Crop to a divisible region.
            mask_cropped = mask[: h_blocks * patch_size, : w_blocks * patch_size]
            mask_patches = mask_cropped.reshape(
                h_blocks, patch_size, w_blocks, patch_size
            )
            mask_patches = np.any(mask_patches, axis=(1, 3))  # [H//14, W//14]
            # flatten the mask_patches
            selected_patches = mask_patches.flatten()
            selected_index = np.nonzero(selected_patches)[0].tolist()
            valid_tokens_list.append(selected_index)
        return valid_tokens_list


    # Temporarily replace attention modules and register hooks.
    @contextmanager
    def inject_cross_attention_hooks(
        self,
        flow_model,
        attention_hook_fn,
        drop_hook_fn,
        get_attention: bool = True,
    ):
        hook_handle_list = []
        original_cross_attns = []

        try:
            for module in flow_model.blocks:
                original_cross_attns.append(module.cross_attn)
                module.cross_attn = SparseMultiHeadAttentionWithAttentionMap(
                    module.cross_attn
                )

                handle = module.cross_attn.register_forward_hook(
                    attention_hook_fn if get_attention else drop_hook_fn
                )
                hook_handle_list.append(handle)

            yield  # Run the managed block.

        finally:
            for handle in hook_handle_list:
                handle.remove()

            for module, original_attn in zip(flow_model.blocks, original_cross_attns):
                module.cross_attn = original_attn


    # Sample sparse 3D features while capturing cross-attention maps.
    def sample_slat_attention_inpaint(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        get_attention: bool = True,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, torch.Tensor]]:
        """
        Sample SLAT features with locally scoped attention collection.
        """
        # Local attention state.
        local_attn_state = {
            "maps": [],                     # Processed sequence attention.
            "voxel_maps": [],               # Voxel-level attention.
            "current_mean": None,           # Running attention mean.
            "current_voxel_mean": None,     # Running voxel attention.
            "count": 0,                     # Accumulation count.
            "interval": 24,                 # Save interval.
        }

        # Local hook.
        def local_attention_hook(module, inputs, outputs):
            output, attn_scores = outputs
            if attn_scores is None:
                return output
            # if withOri:
            #     attn_scores = attn_scores[:,:,:-1374]
            attn = torch.softmax(attn_scores, dim=-1)
            voxel_attn = torch.softmax(attn_scores, dim=-2)

            s = local_attn_state
            if s["current_mean"] is None:
                s["current_mean"] = attn
                s["current_voxel_mean"] = voxel_attn
                s["count"] = 1
            elif s["count"] % s["interval"] == 0:
                s["maps"].append(s["current_mean"])
                s["voxel_maps"].append(s["current_voxel_mean"])
                s["current_voxel_mean"] = voxel_attn
                s["current_mean"] = attn
                s["count"] = 1
            else:
                s["current_mean"] = (s["current_mean"] * s["count"] + attn) / (
                    s["count"] + 1
                )
                s["current_voxel_mean"] = (
                    s["current_voxel_mean"] * s["count"] + voxel_attn
                ) / (s["count"] + 1)
                s["count"] += 1
            return output

        # Local output adapter.
        def local_drop_hook(module, inputs, outputs):
            return outputs[0]  # Drop the attention map.

        flow_model = self.models["slat_flow_model"]
        with self.inject_cross_attention_hooks(
            flow_model,
            get_attention=get_attention,
            attention_hook_fn=local_attention_hook,
            drop_hook_fn=local_drop_hook,
        ):
            # Initialize noise.
            if noise is None:
                noise = sp.SparseTensor(
                    feats=torch.randn(coords.shape[0], flow_model.in_channels).to(
                        self.device
                    ),
                    coords=coords,
                )

            # Sample.
            sampler_params = {**self.slat_sampler_params, **sampler_params}
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                **kwargs,
            ).samples

            # Postprocess.
            std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
            slat = slat * std + mean

            if get_attention:
                local_attn_state["maps"].append(local_attn_state["current_mean"])
                local_attn_state["voxel_maps"].append(
                    local_attn_state["current_voxel_mean"]
                )
                return slat, local_attn_state

        return slat

    def exclude_patches_by_voxel_mask_attn(
        self,
        reference_image,
        attn_map,
        keep_voxel_index,
        ori_cond,
        additional_tokens_num=5,
        heads=[0, 4, 12],              
    ):
        attn_map = attn_map[heads].mean(0)
        valid_tokens = self.get_valid_patches([reference_image])[0]
        addition_valid_tokens = [x + additional_tokens_num for x in valid_tokens]
        attn_map = attn_map[keep_voxel_index][:, addition_valid_tokens].sum(
            0
        )  # voxel_tokens
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_valid_image_indices = torch.nonzero(attn_map > 0.55).squeeze(1)
        attn_image_indices = [valid_tokens[i] for i in attn_valid_image_indices]
        attn_tokens_indices = list(range(additional_tokens_num)) + [
            additional_tokens_num + x for x in attn_image_indices
        ]
        ori_cond["cond"] = ori_cond["cond"][:, attn_tokens_indices]
        ori_cond["neg_cond"] = ori_cond["neg_cond"][:, attn_tokens_indices]
        return ori_cond, attn_image_indices


    def find_relevant_voxels_by_patches(
            self,
            reference_image,
            attn_map,
            patch_indices,
            ori_cond=None,
            additional_tokens_num=5,
            heads=[0, 4, 12],
            threshold=0.3
        ):
        """
        Find voxels strongly related to selected image patches.

        Args:
            reference_image: Reference image used to find valid patches.
            attn_map: Attention map [heads, voxels, tokens].
            patch_indices: Original image-patch indices.
            additional_tokens_num: Number of auxiliary tokens.
            heads: Attention heads to use.
            threshold: Relevance threshold.

        Returns:
            Voxel indices above the threshold.
        """
        # Average selected attention heads.
        attn_map = attn_map[heads].mean(0)  # Shape: [voxels, tokens].
        #print(f"Attention map shape after head selection and mean: {attn_map.shape}")
        
        # Find valid patches.
        valid_tokens = self.get_valid_patches([reference_image])[0]
        valid_token_indices = [additional_tokens_num + idx for idx in valid_tokens]
        
        # Map original indices to valid positions.
        token_index_map = {}
        for i, token_idx in enumerate(valid_tokens):
            token_index_map[token_idx] = valid_token_indices[i]
        
        # Convert patch indices to attention positions.
        target_token_indices = []
        for patch_idx in patch_indices:
            if patch_idx in token_index_map:
                target_token_indices.append(token_index_map[patch_idx])
        
        # Return early when no patch is valid.
        if not target_token_indices:
            return []
        
        # Sum target-patch attention per voxel.
        voxel_relevance = attn_map[:, target_token_indices].sum(dim=1)
        #print(f"Voxel relevance shape: {voxel_relevance.shape}")
        
        # Normalize relevance scores.
        voxel_relevance = (voxel_relevance - voxel_relevance.min()) / \
                        (voxel_relevance.max() - voxel_relevance.min() )  
        
        
        # Apply the relevance threshold.
        relevant_voxel_indices = torch.nonzero(voxel_relevance > threshold).squeeze(1).tolist()
        
        # Find the highest-scoring voxel.
        if len(voxel_relevance) > 0:
            max_score_voxel = torch.argmax(voxel_relevance).item()
        else:
            max_score_voxel = None

        
        return relevant_voxel_indices, max_score_voxel
    


    def find_connected_component(self, coords, part_indices, superpoint, neighbor_type=6):
        """
        Find the component in part_indices connected to a superpoint.

        Args:
            coords: All voxel coordinates [N, 4].
            part_indices: Partial indices into coords [M].
            superpoint: Seed index into coords.
            neighbor_type: 6 or 26 connectivity.

        Returns:
            Indices in the connected component.
        """
        coords_xyz = coords[:, 1:4]
        if not torch.is_tensor(part_indices):
            part_indices = torch.tensor(part_indices, device=coords.device)
        part_coords_xyz = coords_xyz[part_indices]

        part_set = set(part_indices.tolist())
        coord_to_index = {tuple(coords_xyz[idx].tolist()): idx for idx in part_indices.tolist()}

        if neighbor_type == 6:
            neighbors = torch.tensor([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], device=coords.device)
        elif neighbor_type == 26:
            neighbors = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if not (i==0 and j==0 and k==0)], device=coords.device)
        else:
            raise ValueError("neighbor_type must be 6 or 26")

        if superpoint not in part_set:
            return []

        visited = set()
        queue = deque([superpoint])
        visited.add(superpoint)
        result_indices = []

        while queue:
            current_idx = queue.popleft()
            result_indices.append(current_idx)
            current_xyz = coords_xyz[current_idx]

            for offset in neighbors:
                neighbor_xyz = tuple((current_xyz + offset).tolist())
                neighbor_idx = coord_to_index.get(neighbor_xyz, None)
                if neighbor_idx is not None and neighbor_idx in part_set and neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        return result_indices


    @torch.no_grad()
    def img_partial_image_partial_voxel_autoCat(
        self,
        voxel_indices: List[int], # the indices for voxel
        reference_img: Image.Image,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        preprocess_image: bool = True,
    ):
        torch.manual_seed(seed)
        if preprocess_image:
            reference_img = self.preprocess_image(reference_img)

        reference_cond = self.get_cond([reference_img])

        coords = self.sample_sparse_structure(
            reference_cond, sampler_params=sparse_structure_sampler_params
        )

        gt_slat, ori_attn_dict = self.sample_slat_attention_inpaint(
            reference_cond, coords, slat_sampler_params
        )

        #result,indices = build_bvh_and_split(gt_slat.coords, gt_slat.feats, target_num_regions=2)

        ori_attn_map = torch.stack(ori_attn_dict["voxel_maps"], dim=0).mean(0)
        print(f"Attention map shape: {ori_attn_map.shape}")

        Dcoords, downsample_idx = self.downsample_coords(coords, require_upsample=True)
        #downsample_coords_idx = torch.unique(downsample_idx[voxel_indices])
        #print(f"Downsampled coords shape: {Dcoords.shape}, Downsampled indices: {downsample_coords_idx.shape}")

        coords_match = self.upsample_coords(new_coords=Dcoords[voxel_indices],factor=(2, 2, 2),orig_coords=coords)

        #downsample_coords_idx = torch.unique(self.downsample_coords(coords = coords,require_upsample=True)[1][indices[0]])

        filtered_cond,image_indices = self.exclude_patches_by_voxel_mask_attn(
            reference_img, ori_attn_map, voxel_indices, reference_cond
        )

        return filtered_cond,image_indices,gt_slat,coords_match

    
    def sample_sparse_structure_SR(
        self,
        cond: dict,                   # Conditioning.
        num_samples: int = 1,         # Number of samples.
        sampler_params: dict = {},    # Sampler controls.
        dense_coords: torch.Tensor = None,  # Dense coordinate tensor.
        #t: float = 0.5               # Noise-coordinate interpolation factor.
    ) -> torch.Tensor:                # Active voxel coordinates [N, 4].
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Encode occupancy latent
        encoder = self.models['sparse_structure_encoder']
        projector = self.models['projection']
        coords_new = encoder(dense_coords)
        print(f"Encoded coordinate shape: {coords_new.shape}")
        if coords_new.dim() == 5 and coords_new.shape[0] == 1:
            coords_new_no_batch = coords_new.squeeze(0)
        elif coords_new.dim() == 4:
            coords_new_no_batch = coords_new
        else:
            # Only batch size 1 is supported.
            raise ValueError(f"Unsupported shape for transformation: {coords_new.shape}")

        # Match the training-time flattening layout.
        
        C, D, H, W = coords_new_no_batch.shape
        
        # Move W to the leading dimension.
        # [C, D, H, W] -> [W, C, D, H]
        coords_new_permuted = coords_new_no_batch.permute(3, 0, 1, 2)
        
        # Flatten C, D, and H.
        # [W, C, D, H] -> [W, C*D*H]
        coords_new_transformed = coords_new_permuted.reshape(W, -1)
        '''C, D, H, W = coords_new_no_batch.shape 
        z = coords_new_no_batch.view(C, 4, 4, 4, 4, 4, 4)
        z = z.permute(1, 3, 5, 0, 2, 4, 6)
        coords_new_transformed = z.reshape(-1, C * 4 * 4 * 4)
        print(f"Transformed coordinate shape: {coords_new_transformed.shape}")'''
        
        # End layout transform.

        # Expected shape: [16, 2048].
        print("Shape after transformation (input to projector):", coords_new_transformed.shape)

        # Add a leading batch dimension.
        projector_input = coords_new_transformed.unsqueeze(0)

        # Expected shape: [1, 16, 2048].
        print("Shape of final input to projector:", projector_input.shape)

        new_cond = projector(projector_input) # Use the 3D input.
        neg_new_cond = torch.zeros_like(new_cond)

        # Log shapes.
        print("Shape of original cond:", cond['cond'].shape)
        print("Shape of new structure latent:", new_cond.shape)
        # Align devices.
        cond['cond'] = cond['cond'].to(new_cond.device)
        cond['neg_cond'] = cond['neg_cond'].to(new_cond.device)
        # Concatenate conditions.
        cond['cond'] = torch.cat([cond['cond'], new_cond], dim=1)
        cond['neg_cond'] = torch.cat([cond['neg_cond'], neg_new_cond], dim=1)

        print("Shape after concatenation:", cond['cond'].shape)

        # Sample occupancy latent
        denoiser = self.models['denoiser']
        reso = denoiser.resolution
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        noise = torch.randn(num_samples, denoiser.in_channels, reso, reso, reso).to(self.device)
        print(f"After noise allocation: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        #coords_new = (1-t) * coords_new + t * noise
        print(f"Peak memory after mixing: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            denoiser,
            noise,
            **cond,
            **sampler_params,
            #start_ratio = t,
            verbose=True
        ).samples
        print(f"After flow model: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        return coords
    
    @torch.no_grad()
    def run_SR(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 1,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = False,
        dense_coords: torch.Tensor = None,  # Dense coordinate tensor.
        #t: float = 0.5               # Noise-coordinate interpolation factor.
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure_SR(cond, num_samples, sparse_structure_sampler_params, dense_coords)
        print("Super-resolved coordinate shape:", coords.shape)
        cond = self.get_cond([image])
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)


    def sample_sparse_structure_voxel(
        self,
        dense_coords: torch.Tensor,   # Input voxels [1, C, D, H, W].
        num_samples: int = 1,         # Number of samples.
        sampler_params: dict = {},    # Sampler parameters.
    ) -> torch.Tensor:
        """
        Encode and flatten voxels, sample, then return coordinates.
        """
        
        # =====================================================
        # Encode and flatten voxel conditioning.
        # =====================================================
        encoder = self.models['sparse_structure_encoder']
        # No projection layer is used.
        
        # Encode.
        with torch.no_grad():
            coords_new = encoder(dense_coords)
            # print(f"Encoded coordinate shape: {coords_new.shape}")

            # Handle the batch dimension; currently single-sample only.
            if coords_new.dim() == 5 and coords_new.shape[0] == 1:
                coords_new_no_batch = coords_new.squeeze(0)
            elif coords_new.dim() == 4:
                coords_new_no_batch = coords_new
            else:
                raise ValueError(f"Unsupported shape for transformation: {coords_new.shape}")

            # Flatten from [C, D, H, W].
            C, D, H, W = coords_new_no_batch.shape
            
            # [C, D, H, W] -> [W, C, D, H]
            coords_new_permuted = coords_new_no_batch.permute(3, 0, 1, 2)
            
            # [W, C, D, H] -> [W, C*D*H]
            coords_new_transformed = coords_new_permuted.reshape(W, -1)
            # print(f"Flattened shape: {coords_new_transformed.shape}")

            # Add a batch dimension without projection.
            # [W, Feature_Dim] -> [1, W, Feature_Dim]
            latents = coords_new_transformed.unsqueeze(0)
        
        # =====================================================
        # Build the condition dictionary.
        # =====================================================
        # Use flattened features directly.
        cond = {
            'cond': latents,
            'neg_cond': torch.zeros_like(latents)
        }
        
        # Replicate conditions for multiple samples.
        if num_samples > 1:
            cond['cond'] = cond['cond'].repeat(num_samples, 1, 1)
            cond['neg_cond'] = cond['neg_cond'].repeat(num_samples, 1, 1)

        # =====================================================
        # Sample.
        # =====================================================
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        # Generate initial noise.
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        
        # Merge sampler parameters.
        run_sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        # Run flow-matching sampling.
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **run_sampler_params,
            verbose=False # Enable for debugging.
        ).samples
        
        # =====================================================
        # Decode.
        # =====================================================
        decoder = self.models['sparse_structure_decoder']
        # Extract coordinates.
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords
