import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

class RandomTokenDataset(Dataset):
    """
    Dataset that generates random token sequences for model warmup
    
    This dataset is useful for initial training phases when we want
    to warm up model parameters using random data before fine-tuning
    on real data.
    """
    def __init__(
        self, 
        vocab_size: int, 
        sequence_length: int, 
        size: int = 10000,
        min_token_id: int = 1,  # Skip padding token (0)
        include_special_tokens: bool = False
    ):
        """
        Initialize the random dataset
        
        Args:
            vocab_size: Size of the vocabulary
            sequence_length: Length of sequences to generate
            size: Number of examples in the dataset
            min_token_id: Minimum token ID to generate
            include_special_tokens: Whether to include special tokens like [CLS], [SEP], etc.
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.size = size
        self.min_token_id = min_token_id
        self.include_special_tokens = include_special_tokens
        
        # Store special token IDs (to be filled if needed)
        self.special_token_ids = []
        
    def set_special_token_ids(self, special_token_ids: List[int]):
        """Set special token IDs to occasionally include in generated data"""
        self.special_token_ids = special_token_ids
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random tokens
        if self.include_special_tokens and self.special_token_ids and np.random.random() < 0.1:
            # 10% chance to include a special token at the beginning
            special_token = torch.tensor([np.random.choice(self.special_token_ids)])
            regular_tokens = torch.randint(
                self.min_token_id, 
                self.vocab_size, 
                (self.sequence_length - 1,)
            )
            input_ids = torch.cat([special_token, regular_tokens])
        else:
            # Regular random tokens
            input_ids = torch.randint(
                self.min_token_id, 
                self.vocab_size, 
                (self.sequence_length,)
            )
        
        # For causal LM, target is input shifted right by one
        labels = torch.cat([input_ids[1:], torch.tensor([self.min_token_id])])
        
        # Full attention mask
        attention_mask = torch.ones_like(input_ids).float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

class StructuredRandomDataset(Dataset):
    """
    Dataset that generates random data with some structure
    
    This dataset is designed to generate data with patterns
    that might help models learn better during warmup phase
    compared to completely random data.
    """
    def __init__(
        self, 
        vocab_size: int, 
        sequence_length: int, 
        size: int = 10000,
        pattern_type: str = "repeat",  # Options: "repeat", "arithmetic", "geometric"
        min_token_id: int = 1
    ):
        """
        Initialize structured random dataset
        
        Args:
            vocab_size: Size of vocabulary
            sequence_length: Length of sequences to generate
            size: Number of examples in dataset
            pattern_type: Type of pattern to generate
            min_token_id: Minimum token ID to use
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.size = size
        self.pattern_type = pattern_type
        self.min_token_id = min_token_id
        
    def __len__(self):
        return self.size
    
    def _generate_repeat_pattern(self):
        """Generate sequence with repeating patterns"""
        # Choose a pattern length between 2 and 5
        pattern_length = np.random.randint(2, 6)
        
        # Generate the base pattern
        pattern = torch.randint(
            self.min_token_id, 
            self.vocab_size, 
            (pattern_length,)
        )
        
        # Repeat pattern to fill sequence
        repeats = self.sequence_length // pattern_length + 1
        repeated = pattern.repeat(repeats)
        
        # Trim to sequence length
        return repeated[:self.sequence_length]
    
    def _generate_arithmetic_pattern(self):
        """Generate sequence with arithmetic progression"""
        # Choose a starting value and step
        start = np.random.randint(self.min_token_id, self.vocab_size // 2)
        step = np.random.randint(1, 10)
        
        # Generate sequence: start, start+step, start+2*step, ...
        values = torch.arange(0, self.sequence_length) * step + start
        
        # Apply modulo to keep within vocab size
        values = values % (self.vocab_size - self.min_token_id) + self.min_token_id
        
        return values
    
    def _generate_geometric_pattern(self):
        """Generate sequence with geometric/exponential pattern"""
        # Choose a starting value and ratio (as power)
        start = np.random.randint(self.min_token_id, self.vocab_size // 4)
        power = np.random.uniform(1.1, 1.5)
        
        # Generate sequence: start, start*power, start*power^2, ...
        values = torch.tensor([int(start * (power ** i)) for i in range(self.sequence_length)])
        
        # Apply modulo to keep within vocab size
        values = values % (self.vocab_size - self.min_token_id) + self.min_token_id
        
        return values
    
    def __getitem__(self, idx):
        # Generate pattern-based sequence according to selected pattern type
        if self.pattern_type == "repeat":
            input_ids = self._generate_repeat_pattern()
        elif self.pattern_type == "arithmetic":
            input_ids = self._generate_arithmetic_pattern()
        elif self.pattern_type == "geometric":
            input_ids = self._generate_geometric_pattern()
        else:
            # Fallback to random if pattern type not recognized
            input_ids = torch.randint(
                self.min_token_id, 
                self.vocab_size, 
                (self.sequence_length,)
            )
        
        # For causal LM, target is input shifted right by one
        labels = torch.cat([input_ids[1:], torch.tensor([self.min_token_id])])
        
        # Full attention mask
        attention_mask = torch.ones_like(input_ids).float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

def create_random_warmup_dataloader(
    vocab_size: int,
    sequence_length: int,
    batch_size: int,
    dataset_size: int = 10000,
    structured: bool = False,
    pattern_type: str = "repeat",
    num_workers: int = 4
):
    """
    Create a DataLoader with random data for warmup training
    
    Args:
        vocab_size: Size of vocabulary
        sequence_length: Length of token sequences
        batch_size: Batch size for the DataLoader
        dataset_size: Number of samples in the dataset
        structured: Whether to use structured patterns
        pattern_type: Type of pattern if structured=True
        num_workers: Number of worker processes for the DataLoader
        
    Returns:
        DataLoader with random data
    """
    if structured:
        dataset = StructuredRandomDataset(
            vocab_size=vocab_size, 
            sequence_length=sequence_length,
            size=dataset_size,
            pattern_type=pattern_type
        )
    else:
        dataset = RandomTokenDataset(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            size=dataset_size
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )