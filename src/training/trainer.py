import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from ..utils.metrics import Metrics

class Trainer:
    """
    Trainer class for language models with configurable normalization
    Supports various warmup strategies and metrics tracking
    """
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        output_dir: str = "outputs",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1000,
        eval_interval: int = 500,
        metrics_cls: Optional[Metrics] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.warmup_dataloader = warmup_dataloader
        self.optimizer = optimizer or self._create_optimizer()
        self.lr_scheduler = lr_scheduler or self._create_scheduler()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.metrics = metrics_cls or Metrics()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save run config
        self._save_config()
            
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        optimizer_type = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("learning_rate", 3e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        beta1 = self.config.get("beta1", 0.9)
        beta2 = self.config.get("beta2", 0.999)
        eps = self.config.get("eps", 1e-8)
        
        # Get parameters
        params = list(self.model.parameters())
        
        if optimizer_type == "adamw":
            from torch.optim import AdamW
            optimizer = AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps
            )
        elif optimizer_type == "adam":
            from torch.optim import Adam
            optimizer = Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps
            )
        elif optimizer_type == "sgd":
            from torch.optim import SGD
            momentum = self.config.get("momentum", 0.9)
            optimizer = SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        return optimizer
        
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        scheduler_type = self.config.get("scheduler", "cosine").lower()
        warmup_steps = self.config.get("warmup_steps", 1000)
        total_steps = self.config.get("total_steps", None)
        
        # Calculate total steps if not provided
        if total_steps is None:
            epochs = self.config.get("num_epochs", 1)
            steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
            total_steps = epochs * steps_per_epoch
            
        # Warmup ratio
        if warmup_steps < 1:  # interpret as ratio if < 1
            warmup_steps = int(warmup_steps * total_steps)
        
        if scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == "constant":
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
            
        return scheduler
        
    def _save_config(self):
        """Save training configuration"""
        config_path = os.path.join(self.output_dir, "training_config.json")
        
        # Get config as dict
        config_dict = {k: v for k, v in self.config.items()}
        
        # Add timestamp
        config_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Add model config if available
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "to_dict"):
                config_dict["model_config"] = self.model.config.to_dict()
            else:
                config_dict["model_config"] = self.model.config.__dict__
        
        # Save to file
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
    def _log_step(self, loss, lr, grad_norm=None, step_type="train"):
        """Log training step information"""
        self.logger.info(
            f"{step_type.capitalize()} Step: {self.global_step}, "
            f"Loss: {loss:.4f}, "
            f"LR: {lr:.8f}"
            + (f", Grad norm: {grad_norm:.4f}" if grad_norm is not None else "")
        )
        
    def _save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Checkpoint path
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"checkpoint-{self.global_step}.pt"
        )
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "metrics": self.metrics.get_history()
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if needed
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
            
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def _evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Record loss
                val_losses.append(loss.item())
                
        # Calculate average loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Check if this is the best model
        is_best = avg_val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_val_loss
            
        # Log results
        self.logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Update metrics
        self.metrics.update(
            global_step=self.global_step,
            val_loss=avg_val_loss
        )
        
        # Return to training mode
        self.model.train()
        
        return avg_val_loss, is_best
        
    def _train_step(self, batch):
        """Execute a single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Normalize loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
        
    def _train_epoch(self, dataloader, desc="Training"):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        
        # Set tqdm progress bar
        pbar = tqdm(dataloader, desc=desc)
        
        for step, batch in enumerate(pbar):
            # Execute training step
            loss = self._train_step(batch)
            step_loss += loss
            epoch_loss += loss
            
            # Update parameters after accumulation steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = None
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    ).item()
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # LR scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update global step
                self.global_step += 1
                
                # Get current learning rate
                lr = self.optimizer.param_groups[0]["lr"]
                
                # Update metrics
                self.metrics.update(
                    global_step=self.global_step,
                    train_loss=step_loss / self.gradient_accumulation_steps,
                    learning_rate=lr,
                    grad_norm=grad_norm
                )
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = step_loss / self.gradient_accumulation_steps
                    self._log_step(avg_loss, lr, grad_norm)
                    
                # Reset step loss
                step_loss = 0.0
                
                # Validation
                if self.global_step % self.eval_interval == 0:
                    val_loss, is_best = self._evaluate()
                    
                    # Save best model
                    if is_best:
                        self._save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self._save_checkpoint()
                    
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "lr": f"{lr:.8f}",
                    "step": self.global_step
                })
        
        # Return average epoch loss
        return epoch_loss / len(dataloader)
    
    def train(self):
        """Run full training"""
        self.logger.info("Starting training...")
        
        # Get training config
        num_epochs = self.config.get("num_epochs", 1)
        warmup_epochs = self.config.get("warmup_epochs", 0)
        
        # Run warmup on random data if requested
        if warmup_epochs > 0 and self.warmup_dataloader is not None:
            self.logger.info(f"Running {warmup_epochs} warmup epochs on random data...")
            for epoch in range(warmup_epochs):
                self.logger.info(f"Warmup Epoch {epoch+1}/{warmup_epochs}")
                self._train_epoch(self.warmup_dataloader, desc=f"Warmup Epoch {epoch+1}")
        
        # Main training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            epoch_loss = self._train_epoch(self.train_dataloader, desc=f"Epoch {epoch+1}")
            self.logger.info(f"Epoch {epoch+1} complete, loss: {epoch_loss:.4f}")
            
            # Validation after each epoch
            val_loss, is_best = self._evaluate()
            
            # Save checkpoint after each epoch
            self._save_checkpoint(is_best=is_best)
            
        self.logger.info("Training complete!")
        
        # Final evaluation
        final_val_loss, _ = self._evaluate()
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(final_model_path)
        else:
            torch.save(self.model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))
            
        self.logger.info(f"Final model saved to {final_model_path}")
        
        return {
            "best_val_loss": self.best_val_loss,
            "final_val_loss": final_val_loss,
            "metrics": self.metrics.get_history()
        }
        
    def load_checkpoint(self, checkpoint_path):
        """Load trainer state from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler
        if self.lr_scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        # Load metrics
        if "metrics" in checkpoint:
            self.metrics.load_history(checkpoint["metrics"])
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")