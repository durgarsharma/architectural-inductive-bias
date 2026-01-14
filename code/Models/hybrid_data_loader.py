"""
Data Loader for HYBRID Factorial Cipher Datasets
Place in: models/data_loader_hybrid.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import string


class HYBRIDCipherDataset(Dataset):
    """Dataset for HYBRID factorial cipher text pairs"""
    
    def __init__(self, data_path, max_length=512):
        self.max_length = max_length
        
        # Character vocabulary: a-z + space
        self.chars = string.ascii_lowercase + ' '
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)  # 27
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        text = text[:self.max_length]
        
        indices = []
        for char in text.lower():
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[' '])
        
        return indices
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Support both formats
        if 'cipher' in item:
            cipher_text = item['cipher']
            plain_text = item['plain']
        else:
            cipher_text = item['ciphertext']
            plain_text = item['plaintext']
        
        cipher_indices = self.text_to_indices(cipher_text)
        plain_indices = self.text_to_indices(plain_text)
        
        return {
            'cipher': torch.tensor(cipher_indices, dtype=torch.long),
            'plain': torch.tensor(plain_indices, dtype=torch.long),
            'length': len(cipher_indices)
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    cipher_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    plain_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    for i, item in enumerate(batch):
        length = item['length']
        cipher_batch[i, :length] = item['cipher']
        plain_batch[i, :length] = item['plain']
    
    return {
        'cipher': cipher_batch,
        'plain': plain_batch,
        'lengths': lengths
    }


def get_data_loaders(data_dir, cipher_type, batch_size=32, max_length=512, num_workers=0):
    """
    Create train/val/test data loaders for HYBRID
    
    Args:
        data_dir: Base directory with cipher datasets
        cipher_type: Type of cipher (e.g., 'caesar')
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_dir) / cipher_type
    
    train_dataset = HYBRIDCipherDataset(data_path / 'train.json', max_length)
    val_dataset = HYBRIDCipherDataset(data_path / 'val.json', max_length)
    test_dataset = HYBRIDCipherDataset(data_path / 'test.json', max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader