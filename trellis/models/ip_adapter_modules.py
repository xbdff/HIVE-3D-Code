import torch
import torch.nn as nn
from ..modules.utils import convert_module_to_f16, convert_module_to_f32

class LatentProjector(nn.Module):
    """
    Project latent sequences from C_in to the transformer's C_out.
    """
    def __init__(self, in_features: int, out_features: int, num_tokens: int):
        """
        Initialize the latent projector.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output conditioning dimension.
            num_tokens (int): Input sequence length.
        """
        super().__init__()
        self.num_tokens = num_tokens

        # Map each token from C_in to C_out.
        self.proj = nn.Linear(in_features, out_features)

        # Normalize projected features.
        self.norm = nn.LayerNorm(out_features)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize a latent sequence.

        Args:
            latent (torch.Tensor): Input of shape [B, N, C_in].

        Returns:
            torch.Tensor: Output of shape [B, N, C_out].
        """
        # Validate the sequence length.
        if latent.shape[1] != self.num_tokens:
            raise ValueError(
                f"Input latent sequence length mismatch. "
                f"The projector was configured for {self.num_tokens} tokens, "
                f"but received an input with {latent.shape[1]} tokens."
            )
        
        # Project features.
        projected_latent = self.proj(latent)
        
        # Normalize the result.
        projected_latent = self.norm(projected_latent)
        
        return projected_latent
    

    def convert_to_fp16(self) -> None:
        pass
        return self
