import os
import json
import argparse
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from ..models.llama import Llama71M, LlamaConfig
from ..models.bert import BertForMaskedLM, BertConfig  # You'll need to implement these
from ..models.nanogpt import NanoGPT, NanoGPTConfig  # You'll need to implement these
from ..data.data_manager import DataManager
from ..training.trainer import Trainer
from ..utils.metrics import Metrics

class Experiment:
    """
    Class to run normalization experiments on language models
    """
    def __init__(
        self,
        config_path: str,
        model_type: str = "llama",
        experiment_name: Optional[str] = None,
        seed: int = 42
    ):
        # Set random seed for reproducibility
        self._set_seed(seed)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set model type
        self.model_type = model_type.lower()
        
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.model_type}_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create output directory
        self.output_dir = os.path.join(
            self.config.get("output_dir", "outputs"),
            experiment_name
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Save experiment config
        self._save_experiment_config()
        
        # Initialize metrics
        self.metrics = Metrics()
            
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def _save_experiment_config(self):
        """Save experiment configuration to output directory"""
        config_path = os.path.join(self.output_dir, "experiment_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def create_model(self):
        """Create model based on configuration"""
        model_config = self.config.get("model", {})
        
        if self.model_type == "llama":
            # Create Llama config
            config = LlamaConfig(**model_config)
            
            # Create model
            model = Llama71M(config)
        elif self.model_type == "bert":
            # Create BERT config
            config = BertConfig(**model_config)
            
            # Create model
            model = BertForMaskedLM(config)
        elif self.model_type == "nanogpt":
            # Create NanoGPT config
            config = NanoGPTConfig(**model_config)
            
            # Create model
            model = NanoGPT(config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model
    
    def load_data(self):
        """Load and prepare datasets"""
        data_config = self.config.get("data", {})
        
        # Create data manager
        data_manager = DataManager(data_config)
        
        # Load tokenizer
        tokenizer = data_manager.load_tokenizer(
            model_type=self.model_type,
            model_name_or_path=data_config.get("tokenizer_path", self.model_type)
        )
        
        # Load dataset
        dataset_name = data_config.get("dataset_name", "wikitext")
        train_dataset, val_dataset = data_manager.load_dataset(
            dataset_name=dataset_name,
            split_ratio=data_config.get("val_split", 0.1)
        )
        
        # Create random dataset if using random warmup
        if self.config.get("training", {}).get("random_warmup", False):
            random_dataset = data_manager.create_random_dataset(
                size=data_config.get("random_warmup_size", 10000)
            )
        else:
            random_dataset = None
            
        # Create dataloaders
        batch_size = data_config.get("batch_size", 8)
        train_loader, val_loader, warmup_loader = data_manager.get_dataloaders(
            batch_size=batch_size,
            random_warmup=self.config.get("training", {}).get("random_warmup", False),
            random_warmup_size=data_config.get("random_warmup_size", 10000)
        )
        
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "warmup_loader": warmup_loader,
            "tokenizer": tokenizer,
            "data_manager": data_manager
        }
    
    def run(self):
        """Run the experiment"""
        print(f"Starting experiment: {self.experiment_name}")
        
        # Create model
        model = self.create_model()
        print(f"Created {self.model_type} model")
        
        # Load data
        data = self.load_data()
        print("Loaded datasets")
        
        # Get training config
        training_config = self.config.get("training", {})
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=training_config,
            train_dataloader=data["train_loader"],
            val_dataloader=data["val_loader"],
            warmup_dataloader=data["warmup_loader"],
            output_dir=self.output_dir,
            metrics_cls=self.metrics
        )
        print("Created trainer")
        
        # Run training
        results = trainer.train()
        print("Training complete!")
        
        # Save results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results
    
def run_normalization_experiment(experiment_type: str):
    """
    Run a specific normalization experiment
    
    Args:
        experiment_type: Type of experiment (preln, postln, custom)
    """
    # Create base config path
    if experiment_type.lower() == "preln":
        config_path = "configs/experiments/preln.yaml"
    elif experiment_type.lower() == "postln":
        config_path = "configs/experiments/postln.yaml"
    elif experiment_type.lower() == "custom":
        config_path = "configs/experiments/custom_norm.yaml"
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")
        
    # Create experiment name
    experiment_name = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiment
    experiment = Experiment(
        config_path=config_path,
        experiment_name=experiment_name
    )
    
    results = experiment.run()
    
    return results

def compare_normalization_approaches():
    """
    Run experiments comparing preLN, postLN and custom normalization
    """
    results = {}
    
    # Run experiments for each approach
    for exp_type in ["preln", "postln", "custom"]:
        print(f"\n{'-' * 50}")
        print(f"Running {exp_type} experiment")
        print(f"{'-' * 50}\n")
        
        exp_results = run_normalization_experiment(exp_type)
        results[exp_type] = exp_results
        
    # Compare results
    print("\nResults comparison:")
    print(f"{'-' * 50}")
    for exp_type, res in results.items():
        print(f"{exp_type.upper()}: Best val loss = {res['best_val_loss']:.4f}")
        
    # Save comparison
    output_dir = "outputs/comparisons"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    comparison_path = os.path.join(
        output_dir, 
        f"norm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nComparison saved to {comparison_path}")
    
    return results

def main():
    """Main entry point for running experiments"""
    parser = argparse.ArgumentParser(description="Run normalization experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/experiments/default.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "bert", "nanogpt"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison of all normalization approaches"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison of all approaches
        compare_normalization_approaches()
    else:
        # Run single experiment
        experiment = Experiment(
            config_path=args.config,
            model_type=args.model,
            experiment_name=args.experiment,
            seed=args.seed
        )
        experiment.run()

if __name__ == "__main__":
    main()