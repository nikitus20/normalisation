import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

class TextDataset(Dataset):
    """Basic text dataset for language modeling tasks"""
    def __init__(
        self, 
        texts: List[str], 
        tokenizer, 
        max_length: int = 512, 
        stride: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
            self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return sum(max(0, len(text) - self.max_length + 1) 
                   for text in self.tokenized_texts) or 1
    
    def __getitem__(self, idx):
        # Find which text and which position
        text_idx = 0
        for text in self.tokenized_texts:
            text_len = max(0, len(text) - self.max_length + 1) or 1
            if idx < text_len:
                break
            idx -= text_len
            text_idx += 1
        
        tokens = self.tokenized_texts[text_idx]
        
        # Extract sequence
        start_idx = idx
        end_idx = min(start_idx + self.max_length, len(tokens))
        
        # Get input and target
        input_ids = tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
        
        # For causal LM, target is input shifted by one
        labels = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.eos_token_id])])
        
        attention_mask = (input_ids != 0).float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

class RandomDataset(Dataset):
    """Generate random token data for warmup experiments"""
    def __init__(
        self, 
        vocab_size: int, 
        sequence_length: int, 
        size: int = 10000
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random tokens
        input_ids = torch.randint(1, self.vocab_size, (self.sequence_length,))
        
        # For causal LM, target is input shifted by one
        labels = torch.randint(1, self.vocab_size, (self.sequence_length,))
        
        attention_mask = torch.ones_like(input_ids).float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

class DataManager:
    """Manages dataset loading, processing, and providing DataLoaders"""
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.random_dataset = None

    def load_tokenizer(self, model_type: str, model_name_or_path: str):
        """Load the tokenizer based on model type"""
        if model_type.lower() == "llama":
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        elif model_type.lower() == "bert":
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        elif model_type.lower() == "nanogpt":
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer

    def load_dataset(self, dataset_name: str, split_ratio: float = 0.1):
        """Load dataset from HuggingFace datasets"""
        raw_dataset = load_dataset(dataset_name)
        
        # Handle different dataset structures
        if "train" in raw_dataset:
            train_texts = raw_dataset["train"]["text"]
            # Check if there's a validation set, otherwise split train
            if "validation" in raw_dataset:
                val_texts = raw_dataset["validation"]["text"]
            else:
                # Randomly split train set
                num_val = max(1, int(len(train_texts) * split_ratio))
                indices = np.random.permutation(len(train_texts))
                train_indices = indices[num_val:]
                val_indices = indices[:num_val]
                val_texts = [train_texts[i] for i in val_indices]
                train_texts = [train_texts[i] for i in train_indices]
        else:
            # Use the first available split and split it
            first_split = list(raw_dataset.keys())[0]
            all_texts = raw_dataset[first_split]["text"]
            num_val = max(1, int(len(all_texts) * split_ratio))
            indices = np.random.permutation(len(all_texts))
            train_indices = indices[num_val:]
            val_indices = indices[:num_val]
            train_texts = [all_texts[i] for i in train_indices]
            val_texts = [all_texts[i] for i in val_indices]
        
        # Create datasets
        self.train_dataset = TextDataset(
            train_texts, 
            self.tokenizer,
            max_length=self.config.get("max_length", 512),
            stride=self.config.get("stride", 128)
        )
        
        self.val_dataset = TextDataset(
            val_texts, 
            self.tokenizer,
            max_length=self.config.get("max_length", 512),
            stride=self.config.get("stride", 128)
        )
        
        return self.train_dataset, self.val_dataset

    def create_random_dataset(self, size: int = 10000):
        """Create random dataset for warmup experiments"""
        vocab_size = len(self.tokenizer)
        sequence_length = self.config.get("max_length", 512)
        
        self.random_dataset = RandomDataset(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            size=size
        )
        
        return self.random_dataset
    
    def get_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Create a DataLoader from a dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True)
        )
    
    def get_dataloaders(self, batch_size: int, random_warmup: bool = False, 
                       random_warmup_size: int = 1000):
        """Get train and validation DataLoaders"""
        if random_warmup:
            if self.random_dataset is None:
                self.create_random_dataset(size=random_warmup_size)
            warmup_loader = self.get_dataloader(
                self.random_dataset, 
                batch_size=batch_size,
                shuffle=True
            )
        else:
            warmup_loader = None
            
        train_loader = self.get_dataloader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = self.get_dataloader(
            self.val_dataset, 
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, warmup_loader