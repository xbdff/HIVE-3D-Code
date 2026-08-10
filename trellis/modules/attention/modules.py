from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
    

class IPMultiHeadAttention(nn.Module):
    """
    Multi-head attention with IP-Adapter conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
        # IP-Adapter parameters.
        num_ip_tokens: int = 16,
        scale: float = 1.0,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.qk_rms_norm = qk_rms_norm

        # Keep the query projection unchanged.
        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)

        # Split key and value projections.
        self.to_k = nn.Linear(self.ctx_channels, channels, bias=qkv_bias)
        self.to_v = nn.Linear(self.ctx_channels, channels, bias=qkv_bias)

        # Add key/value projections for IP latents.
        self.to_k_ip = nn.Linear(self.ctx_channels, channels, bias=qkv_bias)
        self.to_v_ip = nn.Linear(self.ctx_channels, channels, bias=qkv_bias)
        
        # ---------------------------

        # Normalize queries and text keys when RMSNorm is enabled.
        if self.qk_rms_norm:
            # Reuse MultiHeadRMSNorm.
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)
        
        # IP-Adapter settings.
        self.num_ip_tokens = num_ip_tokens
        self.scale = scale

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        assert context is not None, "IPMultiHeadAttention always requires context."
        
        # Compute queries.
        q = self.to_q(x)
        if self.qk_rms_norm:
            # Reshape for norm, then reshape back
            q_for_norm = q.reshape(B, L, self.num_heads, self.head_dim)
            q_for_norm = self.q_rms_norm(q_for_norm)
            q = q_for_norm.reshape(B, L, C)

        # Split text and IP contexts.
        end_pos = context.shape[1] - self.num_ip_tokens
        text_context = context[:, :end_pos, :]
        ip_context = context[:, end_pos:, :]

        # Compute text keys and values.
        k_text = self.to_k(text_context)
        v_text = self.to_v(text_context)
        if self.qk_rms_norm:
            k_text_for_norm = k_text.reshape(B, -1, self.num_heads, self.head_dim)
            k_text_for_norm = self.k_rms_norm(k_text_for_norm)
            k_text = k_text_for_norm.reshape(B, -1, C)

        # Compute IP-Adapter keys and values.
        k_ip = self.to_k_ip(ip_context)
        v_ip = self.to_v_ip(ip_context)
        # IP keys and values are not normalized.

        # 4. Reshape for Multi-Head Attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_text = k_text.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_text = v_text.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_ip = k_ip.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_ip = v_ip.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Run text and IP attention independently.
        attn_output_text = F.scaled_dot_product_attention(q, k_text, v_text)
        attn_output_ip = F.scaled_dot_product_attention(q, k_ip, v_ip)

        # Fuse both attention outputs.
        attn_output = attn_output_text + self.scale * attn_output_ip

        # Restore the shape and apply the output projection.
        attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
        output = self.to_out(attn_output)

        return output
