import os
import random
import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from dotenv import load_dotenv

# Add the adjacent HYPIR package.
from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import GPTCaptioner

try:
    from .hypir_sources import resolve_hypir_weight
    from .runtime_config import has_gpt_captioning_config
except ImportError:
    from hypir_sources import resolve_hypir_weight
    from runtime_config import has_gpt_captioning_config

class HYPIRWrapper:
    def __init__(
        self,
        config_path: str,
        device: str = "cuda",
        max_size=None,
        use_gpt: bool | None = None,
    ):
        """
        Initialize the HYPIR model.

        Args:
            config_path: YAML configuration path.
            device: Inference device.
            max_size: Maximum `(width, height)` or `"w,h"`.
            use_gpt: Enable GPT captioning.
        """
        self.device = device
        self.max_size = None
        self.to_tensor = transforms.ToTensor()
        
        # Load environment variables.
        load_dotenv()
        self.use_gpt = (
            has_gpt_captioning_config(os.environ)
            if use_gpt is None
            else use_gpt
        )

        # Parse the size limit.
        if max_size is not None:
            if isinstance(max_size, str):
                self.max_size = tuple(int(x) for x in max_size.split(","))
            else:
                self.max_size = max_size
        
        # Initialize GPT captioning.
        self.captioner = None
        if self.use_gpt:
            if not has_gpt_captioning_config(os.environ):
                print("Warning: GPT env vars missing. GPT captioning disabled.")
                self.use_gpt = False
            else:
                self.captioner = GPTCaptioner(
                    api_key=os.getenv("GPT_API_KEY"),
                    base_url=os.getenv("GPT_BASE_URL"),
                    model=os.getenv("GPT_MODEL"),
                )

        # Load the model configuration.
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        print(f"Loading HYPIR model from {config_path}...")
        self.config = OmegaConf.load(config_path)
        
        if self.config.base_model_type == "sd2":
            print(f"Loading HYPIR base model from Hugging Face: {self.config.base_model_path}")
            weight_path = resolve_hypir_weight(
                repo_id=self.config.weight_repo_id,
                filename=self.config.weight_filename,
                revision=self.config.weight_revision,
                expected_sha256=self.config.weight_sha256,
            )
            print(
                "Loading HYPIR enhancement weights from Hugging Face: "
                f"{self.config.weight_repo_id}/{self.config.weight_filename}"
            )
            self.model = SD2Enhancer(
                base_model_path=self.config.base_model_path,
                weight_path=weight_path,
                lora_modules=self.config.lora_modules,
                lora_rank=self.config.lora_rank,
                model_t=self.config.model_t,
                coeff_t=self.config.coeff_t,
                device=self.device,
            )
            self.model.init_models()
            print("HYPIR Model loaded successfully.")
        else:
            raise ValueError(f"Unsupported model type: {self.config.base_model_type}")

    def enhance_image(self, 
                      image: Image.Image, 
                      prompt: str = "", 
                      upscale: int = 1, 
                      patch_size: int = 512, 
                      stride: int = 256, 
                      seed: int = -1) -> Image.Image:
        """
        Enhance one image.

        Returns:
            The enhanced PIL image, or None on failure.
        """
        # Set the seed.
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        set_seed(seed)
        
        # Convert the image format.
        image = image.convert("RGB")
        
        # Enforce the size limit.
        if self.max_size is not None:
            out_w, out_h = tuple(int(x * upscale) for x in image.size)
            if out_w * out_h > self.max_size[0] * self.max_size[1]:
                print(f"Error: Resolution {out_w}x{out_h} exceeds limit {self.max_size}.")
                return None

        # Generate an automatic prompt.
        if prompt == "auto":
            if self.use_gpt and self.captioner:
                print("Generating caption with GPT...")
                prompt = self.captioner(image)
                print(f"Generated prompt: {prompt}")
            else:
                print("Error: 'auto' prompt requested but GPT is not configured.")
                return None

        # Convert the image to a tensor.
        image_tensor = self.to_tensor(image).unsqueeze(0)
        
        # Run inference.
        try:
            pil_image = self.model.enhance(
                lq=image_tensor,
                prompt=prompt,
                upscale=upscale,
                patch_size=patch_size,
                stride=stride,
                return_type="pil",
            )[0]
            return pil_image
        except Exception as e:
            print(f"HYPIR Inference Failed: {e}")
            return None
