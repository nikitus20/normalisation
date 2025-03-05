import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import math
import json
import os

from .layers.normalization import NormFactory, TransformerBlock

class LlamaConfig:
    """Configuration class for Llama model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_position: str = "pre",  # "pre" or "post"
        custom_norm_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.custom_norm_kwargs = custom_norm_kwargs or {}
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create a config from a dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_file: str):
        """Load a config from a JSON file"""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    
    def save_pretrained(self, save_directory: str):
        """Save config to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Llama71M(nn.Module):
    """Llama model with 71M parameters and flexible normalization"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        # Position embeddings - rotary positional embeddings are implemented inside attention
        
        # Create transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                mlp_dim=config.intermediate_size,
                dropout=config.hidden_dropout,
                attention_dropout=config.attention_dropout,
                norm_type=config.norm_type,
                norm_position=config.norm_position,
                norm_kwargs={
                    "eps": config.rms_norm_eps,
                    **config.custom_norm_kwargs
                }
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization layer
        self.norm_f = NormFactory.create(
            config.norm_type, 
            config.hidden_size, 
            eps=config.rms_norm_eps,
            **config.custom_norm_kwargs
        )
        
        # Language model head
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.token_embeddings.weight = self.lm_head.weight
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Normal initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True
    ):
        # Get embeddings
        x = self.token_embeddings(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        
        # Apply final normalization
        x = self.norm_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Reshape for cross entropy
            logits_view = logits.view(-1, self.config.vocab_size)
            labels_view = labels.view(-1)
            # Compute loss
            loss = F.cross_entropy(
                logits_view, 
                labels_view, 
                ignore_index=-100
            )
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": x
            }
        else:
            return (loss, logits) if loss is not None else logits
            
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, config=None):
        """
        Load a pretrained model from a directory
        Supports loading from HuggingFace-style checkpoints
        """
        # Load config if not provided
        if config is None:
            config_file = os.path.join(pretrained_model_path, "config.json")
            if os.path.exists(config_file):
                config = LlamaConfig.from_json(config_file)
            else:
                raise ValueError(
                    f"Config file not found at {config_file} and no config provided"
                )
        
        # Create model with config
        model = cls(config)
        
        # Load weights
        checkpoint_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            
            # Handle potential key mismatches
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                # Remove potential 'model.' prefix
                if k.startswith("model."):
                    k = k[6:]  # Remove 'model.'
                
                # Map different key patterns (e.g., layers vs transformer.h)
                if k.startswith("transformer.h."):
                    layer_num = k.split(".")[2]
                    k = k.replace(f"transformer.h.{layer_num}", f"layers.{layer_num}")
                
                # Check if key exists in model
                if k in model_state_dict:
                    # Check if shapes match
                    if v.shape != model_state_dict[k].shape:
                        print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
                        continue
                    
                    filtered_state_dict[k] = v
                else:
                    print(f"Key {k} not found in model state dict")
            
            # Load the filtered state dict
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            
            if len(missing) > 0:
                print(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected}")
        else:
            print(f"No checkpoint found at {checkpoint_file}, initializing from scratch")
        
        return model
        
    def save_pretrained(self, save_directory: str):
        """Save model weights and config to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save weights
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, "pytorch_model.bin")
        )