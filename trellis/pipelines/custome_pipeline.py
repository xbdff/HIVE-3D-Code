from typing import *
import torch
from PIL import Image
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from . import samplers


class TestImageTo3DPipeline(TrellisImageTo3DPipeline):
    """
    A pipeline for me to test the ability of the original SLat generation method.
    """

    def __init__(
        self,
        models=None,
        sparse_structure_sampler=None,
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model=None,
    ):
        super().__init__(
            models,
            sparse_structure_sampler,
            slat_sampler,
            slat_normalization,
            image_cond_model,
        )
        self.hook_handle_list = []

    @staticmethod
    def from_pretrained(path: str) -> "TestImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TestImageTo3DPipeline, TestImageTo3DPipeline).from_pretrained(
            path
        )
        new_pipeline = TestImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        new_pipeline.hook_handle_list = []

        return new_pipeline

    @torch.no_grad()
    def test_image_conditioned_voxel_texturing(
        self,
        coords: torch.Tensor,
        image: Image.Image,
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline. Use the first image to control the structure and the second image to control the style.
        image to control the latent.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)

        cond = self.get_cond([image])
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats), image