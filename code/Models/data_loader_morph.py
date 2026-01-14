"""
MORPH - CONSTRUCTED LANGUAGE
Data Loader for Cipher Datasets
Handles loading and preprocessing of cipher text data
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import string


class CipherDataset(Dataset):
    """Dataset for cipher text pairs"""
    
    def __init__(self, data_path, max_length=512):
        """
        Args:
            data_path: Path to JSON file with cipher pairs
            max_length: Maximum sequence length
        """
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
        # Truncate if needed
        text = text[:self.max_length]
        
        # Convert to indices
        indices = []
        for char in text.lower():
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character -> map to space
                indices.append(self.char_to_idx[' '])
        
        return indices
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])
    
    def __getitem__(self, idx):
        """
        Returns:
            cipher_indices: Encrypted text as indices
            plain_indices: Plaintext as indices
            length: Actual length (before padding)
        """
        item = self.data[idx]
        
        # Check which keys exist in the JSON
        # Support both 'ciphertext'/'plaintext' and 'cipher'/'plain' formats
        if 'ciphertext' in item:
            cipher_text = item['ciphertext']
            plain_text = item['plaintext']
        elif 'cipher' in item:
            cipher_text = item['cipher']
            plain_text = item['plain']
        else:
            raise KeyError(f"Could not find cipher text in item. Available keys: {list(item.keys())}")
        
        # Convert to indices
        cipher_indices = self.text_to_indices(cipher_text)
        plain_indices = self.text_to_indices(plain_text)
        
        # Ensure same length
        min_len = min(len(cipher_indices), len(plain_indices))
        cipher_indices = cipher_indices[:min_len]
        plain_indices = plain_indices[:min_len]
        
        return {
            'cipher': torch.tensor(cipher_indices, dtype=torch.long),
            'plain': torch.tensor(plain_indices, dtype=torch.long),
            'length': len(cipher_indices)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads sequences to same length within batch
    """
    # Find max length in batch
    max_len = max(item['length'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    cipher_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    plain_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Fill in data
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
    Create train/val/test data loaders
    
    Args:
        data_dir: Base directory with cipher datasets
        cipher_type: Type of cipher (e.g., 'caesar', 'atbash')
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_dir) / cipher_type
    
    # Create datasets
    train_dataset = CipherDataset(
        data_path / 'train.json',
        max_length=max_length
    )
    
    val_dataset = CipherDataset(
        data_path / 'val.json',
        max_length=max_length
    )
    
    test_dataset = CipherDataset(
        data_path / 'test.json',
        max_length=max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing Cipher Data Loader...")
    
    # Example usage
    data_dir = "data/cipher_datasets_linguistic_250/MORPH"
    cipher_type = "caesar"
    
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=data_dir,
            cipher_type=cipher_type,
            batch_size=4,
            max_length=512
        )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Cipher: {batch['cipher'].shape}")
        print(f"  Plain: {batch['plain'].shape}")
        print(f"  Lengths: {batch['lengths'].shape}")
        
        # Show example
        dataset = train_loader.dataset
        print(f"\nVocabulary size: {dataset.vocab_size}")
        print(f"Characters: {dataset.chars}")
        
        # Decode example
        cipher_text = dataset.indices_to_text(batch['cipher'][0].tolist())
        plain_text = dataset.indices_to_text(batch['plain'][0].tolist())
        
        print(f"\nExample (first 100 chars):")
        print(f"  Cipher: {cipher_text[:100]}")
        print(f"  Plain:  {plain_text[:100]}")
        
        print("\n✓ Data loader test passed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure cipher datasets have been generated first!")