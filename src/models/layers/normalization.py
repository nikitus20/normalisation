import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class NormFactory:
    """Factory class for creating different normalization layers"""
    @staticmethod
    def create(norm_type: str, dim: int, **kwargs):
        """
        Create a normalization layer
        
        Args:
            norm_type: Type of normalization (layernorm, rmsnorm, custom, etc.)
            dim: Hidden dimension
            **kwargs: Additional arguments for specific norm types
        """
        if norm_type.lower() == "layernorm":
            return nn.LayerNorm(dim, **kwargs)
        elif norm_type.lower() == "rmsnorm":
            return RMSNorm(dim, **kwargs)
        elif norm_type.lower() == "custom":
            return CustomNorm(dim, **kwargs)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

class RMSNorm(nn.Module):
    """RMSNorm normalization module"""
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        
        # Apply weight if using affine transformation
        if self.weight is not None:
            x_norm = x_norm * self.weight
            
        return x_norm

class CustomNorm(nn.Module):
    """
    Custom normalization layer that can be modified for experiments
    This is just a placeholder - modify with your custom approach
    """
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6, 
        affine: bool = True,
        norm_method: str = "mean_std",  # Options: mean_std, rms, l2, etc.
        adaptive_weight: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        self.norm_method = norm_method
        self.adaptive_weight = adaptive_weight
        
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            
        if adaptive_weight:
            # Create an adaptive weighting mechanism
            self.adaptive_layer = nn.Sequential(
                nn.Linear(2, 32),  # Input: [mean, std]
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Calculate statistics
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        
        # Different normalization methods
        if self.norm_method == "mean_std":
            # Standard LayerNorm-style normalization
            x_norm = (x - mean) / std
        elif self.norm_method == "rms":
            # RMSNorm-style normalization
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
            x_norm = x / rms
        elif self.norm_method == "l2":
            # L2 normalization
            l2 = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
            x_norm = x / l2
        else:
            raise ValueError(f"Unknown normalization method: {self.norm_method}")
            
        # Apply adaptive weighting if enabled
        if self.adaptive_weight:
            # Use statistics to determine adaptive weight
            stats = torch.cat([mean, std], dim=-1)
            adapt_weight = self.adaptive_layer(stats).unsqueeze(-1)
            x_norm = x_norm * adapt_weight
            
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
            
        return x_norm

class TransformerBlock(nn.Module):
    """
    Generic Transformer block with configurable normalization
    Can be configured as PreLN, PostLN, or with custom normalization strategy
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: str = "layernorm",
        norm_position: str = "pre",  # "pre" or "post"
        norm_kwargs: Dict = None
    ):
        super().__init__()
        self.dim = dim
        self.norm_position = norm_position
        norm_kwargs = norm_kwargs or {}
        
        # Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Normalization layers
        self.norm1 = NormFactory.create(norm_type, dim, **norm_kwargs)
        self.norm2 = NormFactory.create(norm_type, dim, **norm_kwargs)
        
        self.dropout = nn.Dropout(dropout)

    def _apply_attention(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Handle attention mask
        if attention_mask is not None:
            # Convert from [B, L] to [B, L, L]
            expanded_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            # Convert to additive attention mask
            inverted_mask = 1.0 - expanded_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.bool(), float("-inf")
            )
        else:
            attention_mask = None
        
        # Apply attention
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attention_mask,
            need_weights=False
        )
        
        return attn_output

    def forward(self, x, attention_mask=None):
        # Implement different normalization strategies
        if self.norm_position == "pre":
            # PreLN: Apply normalization before attention and MLP
            attn_output = self._apply_attention(self.norm1(x), attention_mask)
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
        elif self.norm_position == "post":
            # PostLN: Apply normalization after attention and MLP
            attn_output = self._apply_attention(x, attention_mask)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self.mlp(x))
        elif self.norm_position == "custom":
            # Custom normalization strategy - example:
            # Apply normalization before attention but after MLP
            attn_output = self._apply_attention(self.norm1(x), attention_mask)
            x = x + attn_output
            mlp_output = self.mlp(x)
            x = self.norm2(x + mlp_output)
        else:
            raise ValueError(f"Unknown normalization position: {self.norm_position}")
            
        return x