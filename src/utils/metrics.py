import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union

class Metrics:
    """
    Class for tracking training metrics and visualization
    Specializing in tracking normalization-related metrics
    """
    def __init__(self):
        self.history = {
            "global_step": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "batch_time": [],
            "activation_stats": [],
            "memory_usage": []
        }
        self.start_time = time.time()
        
    def update(self, global_step: int, **kwargs):
        """
        Update metrics with new values
        
        Args:
            global_step: Current training step
            **kwargs: Metric values to update
        """
        # Record step
        self.history["global_step"].append(global_step)
        
        # Record time
        elapsed = time.time() - self.start_time
        if "batch_time" not in kwargs:
            self.history["batch_time"].append(elapsed)
            
        # Record other metrics
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
                
        # Check for activation statistics
        if "activation_stats" in kwargs:
            self.history["activation_stats"].append(kwargs["activation_stats"])
            
        # Check memory usage
        if "memory_usage" not in kwargs and torch.cuda.is_available():
            import torch
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            self.history["memory_usage"].append(memory_allocated)
            
    def get_history(self):
        """Get the full metrics history"""
        return self.history
    
    def load_history(self, history: Dict):
        """Load metrics history from dictionary"""
        self.history = history
        
    def save(self, output_dir: str, filename: str = "metrics.json"):
        """Save metrics to a file"""
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
    def load(self, metrics_path: str):
        """Load metrics from a file"""
        with open(metrics_path, "r") as f:
            self.history = json.load(f)
            
    def plot_loss(self, output_dir: str = None, show: bool = True):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        
        steps = self.history["global_step"]
        
        if self.history["train_loss"]:
            plt.plot(steps, self.history["train_loss"], label="Train Loss")
            
        if self.history["val_loss"]:
            # Plot val loss at evaluation steps
            val_steps = []
            val_losses = []
            for i, loss in enumerate(self.history["val_loss"]):
                if loss is not None:
                    val_steps.append(steps[i])
                    val_losses.append(loss)
            plt.plot(val_steps, val_losses, label="Validation Loss", marker="o")
            
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_learning_rate(self, output_dir: str = None, show: bool = True):
        """Plot learning rate schedule"""
        if not self.history["learning_rate"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        steps = self.history["global_step"]
        plt.plot(steps, self.history["learning_rate"])
        
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "lr_schedule.png"), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_gradient_norms(self, output_dir: str = None, show: bool = True):
        """Plot gradient norms over training"""
        if not self.history["grad_norm"]:
            return
            
        plt.figure(figsize=(10, 6))
        
        steps = self.history["global_step"]
        plt.plot(steps, self.history["grad_norm"])
        
        plt.xlabel("Training Steps")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norms During Training")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "grad_norms.png"), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_activation_stats(self, output_dir: str = None, show: bool = True):
        """Plot activation statistics (mean, variance) over training"""
        if not self.history["activation_stats"]:
            return
            
        # Extract activation means and variances
        layers = []
        means = []
        variances = []
        
        for stats in self.history["activation_stats"]:
            for layer, layer_stats in stats.items():
                if layer not in layers:
                    layers.append(layer)
                    means.append([])
                    variances.append([])
                
                layer_idx = layers.index(layer)
                means[layer_idx].append(layer_stats["mean"])
                variances[layer_idx].append(layer_stats["variance"])
                
        # Plot means and variances for each layer
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        for i, layer in enumerate(layers):
            plt.plot(means[i], label=f"Layer {layer}")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Activation")
        plt.title("Mean Activations by Layer")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plt.subplot(1, 2, 2)
        for i, layer in enumerate(layers):
            plt.plot(variances[i], label=f"Layer {layer}")
        plt.xlabel("Training Steps")
        plt.ylabel("Activation Variance")
        plt.title("Activation Variances by Layer")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "activation_stats.png"), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def compare_experiments(self, experiments: Dict[str, "Metrics"], metric: str = "val_loss", 
                           output_dir: str = None, show: bool = True):
        """
        Compare metrics across multiple experiments
        
        Args:
            experiments: Dictionary mapping experiment names to Metrics objects
            metric: Metric to compare
            output_dir: Directory to save plots
            show: Whether to display plots
        """
        plt.figure(figsize=(12, 8))
        
        for name, exp_metrics in experiments.items():
            history = exp_metrics.get_history()
            if metric in history and history[metric]:
                steps = history["global_step"]
                values = history[metric]
                plt.plot(steps, values, label=name)
                
        plt.xlabel("Training Steps")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Comparison of {metric.replace('_', ' ').title()} Across Experiments")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_all(self, output_dir: str, show: bool = False):
        """Generate all plots and save to directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_loss(output_dir, show)
        self.plot_learning_rate(output_dir, show)
        self.plot_gradient_norms(output_dir, show)
        self.plot_activation_stats(output_dir, show)
        
        # Save metrics data
        self.save(output_dir)
        
# Function to extract activation statistics from model
def get_activation_stats(model, layer_names=None):
    """
    Extract activation statistics (mean, variance) from specified layers
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to extract stats from (None for all layers)
    
    Returns:
        Dictionary mapping layer names to statistics
    """
    import torch
    
    stats = {}
    
    # Helper function to recursively get activation stats
    def _get_stats(module, name=""):
        if layer_names is None or name in layer_names:
            # Check if module has activation attribute
            if hasattr(module, "activation"):
                if isinstance(module.activation, torch.Tensor):
                    act = module.activation.detach()
                    stats[name] = {
                        "mean": float(act.mean().item()),
                        "variance": float(act.var().item()),
                        "min": float(act.min().item()),
                        "max": float(act.max().item())
                    }
        
        # Recurse through child modules
        for child_name, child in module.named_children():
            _get_stats(child, name + ("." if name else "") + child_name)
    
    # Start recursion
    _get_stats(model)
    
    return stats

# Hook to register for capturing activations
def register_activation_hooks(model, layer_types=None):
    """
    Register forward hooks to capture activations for specified layer types
    
    Args:
        model: PyTorch model
        layer_types: List of layer types to capture activations for
                    (None for all layers)
    
    Returns:
        List of hook handles
    """
    import torch.nn as nn
    
    hooks = []
    
    # Helper function to recursively register hooks
    def _register_hooks(module):
        if layer_types is None or any(isinstance(module, lt) for lt in layer_types):
            def hook(module, input, output):
                module.activation = output
            
            hooks.append(module.register_forward_hook(hook))
        
        # Recurse through child modules
        for child in module.children():
            _register_hooks(child)
    
    # Start recursion
    _register_hooks(model)
    
    return hooks