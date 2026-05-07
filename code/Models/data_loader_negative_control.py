"""
Data Loader for Negative Control (AES/DES) Cipher Datasets
Handles loading and preprocessing of cryptographically secure cipher data
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import string


class NegativeControlDataset(Dataset):
    """Dataset for AES/DES encrypted text pairs"""
    
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
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
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
        return ''.join([self.idx_to_char.get(idx, ' ') for idx in indices])
    
    def __getitem__(self, idx):
        """
        Returns:
            cipher_indices: Encrypted text as indices
            plain_indices: Plaintext as indices
            length: Actual length (before padding)
        """
        item = self.data[idx]
        
        # Support both 'cipher'/'plain' and 'ciphertext'/'plaintext' formats
        if 'cipher' in item:
            cipher_text = item['cipher']
            plain_text = item['plain']
        elif 'ciphertext' in item:
            cipher_text = item['ciphertext']
            plain_text = item['plaintext']
        else:
            raise KeyError(f"Could not find cipher text. Available keys: {list(item.keys())}")
        
        # Convert to indices
        cipher_indices = self.text_to_indices(cipher_text)
        plain_indices = self.text_to_indices(plain_text)
        
        # Pad sequences
        length = len(cipher_indices)
        
        if length < self.max_length:
            padding = [0] * (self.max_length - length)
            cipher_indices = cipher_indices + padding
            plain_indices = plain_indices + padding
        
        return {
            'cipher': torch.tensor(cipher_indices, dtype=torch.long),
            'plain': torch.tensor(plain_indices, dtype=torch.long),
            'length': length
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable-length sequences
    """
    ciphers = torch.stack([item['cipher'] for item in batch])
    plains = torch.stack([item['plain'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    return {
        'cipher': ciphers,
        'plain': plains,
        'lengths': lengths
    }


def get_negative_control_loaders(
    cipher_dir,
    batch_size=32,
    max_length=512,
    num_workers=0
):
    """
    Create train/val/test data loaders for negative control experiments
    
    Args:
        cipher_dir: Directory containing train.json, val.json, test.json
                   e.g., 'data/cipher_datasets_negative_control/English/AES'
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    
    Example:
        >>> train_loader, val_loader, test_loader = get_negative_control_loaders(
        ...     cipher_dir='data/cipher_datasets_negative_control/English/AES',
        ...     batch_size=32
        ... )
    """
    cipher_path = Path(cipher_dir)
    
    # Verify directory exists
    if not cipher_path.exists():
        raise FileNotFoundError(f"Cipher directory not found: {cipher_path}")
    
    # Create datasets
    train_dataset = NegativeControlDataset(
        cipher_path / 'train.json',
        max_length=max_length
    )
    
    val_dataset = NegativeControlDataset(
        cipher_path / 'val.json',
        max_length=max_length
    )
    
    test_dataset = NegativeControlDataset(
        cipher_path / 'test.json',
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
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# Backward compatibility wrapper
def get_data_loaders(data_dir, cipher_type, batch_size=32, max_length=512, num_workers=0):
    """
    Wrapper function for backward compatibility
    Automatically detects if it's a negative control dataset
    
    Args:
        data_dir: Base directory
        cipher_type: 'aes', 'des', 'AES', 'DES', or classical cipher type
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    cipher_type_upper = cipher_type.upper()
    
    # Check if it's AES/DES (negative control)
    if cipher_type_upper in ['AES', 'DES']:
        cipher_dir = Path(data_dir) / cipher_type_upper
        
        # If cipher_dir doesn't exist, try lowercase
        if not cipher_dir.exists():
            cipher_dir = Path(data_dir) / cipher_type.lower()
        
        if not cipher_dir.exists():
            raise FileNotFoundError(
                f"Cipher directory not found. Tried:\n"
                f"  - {Path(data_dir) / cipher_type_upper}\n"
                f"  - {Path(data_dir) / cipher_type.lower()}"
            )
        
        print(f"Using negative control data loader for {cipher_type_upper}")
        return get_negative_control_loaders(
            cipher_dir=cipher_dir,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers
        )
    
    # Classical cipher - use standard path
    else:
        data_path = Path(data_dir) / cipher_type
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        print(f"Using standard data loader for {cipher_type}")
        
        # Create datasets
        train_dataset = NegativeControlDataset(
            data_path / 'train.json',
            max_length=max_length
        )
        
        val_dataset = NegativeControlDataset(
            data_path / 'val.json',
            max_length=max_length
        )
        
        test_dataset = NegativeControlDataset(
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
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val:   {len(val_dataset)}")
        print(f"  Test:  {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Test negative control data loader')
    parser.add_argument('--cipher-dir', type=str,
                       default='data/cipher_datasets_negative_control/English/AES',
                       help='Cipher directory')
    parser.add_argument('--test-wrapper', action='store_true',
                       help='Test the wrapper function')
    
    args = parser.parse_args()
    
    if args.test_wrapper:
        # Test wrapper with AES
        print("\n" + "="*70)
        print("Testing wrapper function with AES")
        print("="*70)
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir='data/cipher_datasets_negative_control/English',
            cipher_type='aes',
            batch_size=4
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Cipher: {batch['cipher'].shape}")
        print(f"  Plain:  {batch['plain'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        
    else:
        # Test direct function
        print("\n" + "="*70)
        print("Testing direct function")
        print("="*70)
        train_loader, val_loader, test_loader = get_negative_control_loaders(
            cipher_dir=args.cipher_dir,
            batch_size=4
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Cipher: {batch['cipher'].shape}")
        print(f"  Plain:  {batch['plain'].shape}")
        print(f"  Lengths: {batch['lengths']}")
        
        # Show sample
        dataset = train_loader.dataset
        print(f"\nSample cipher text: {dataset.indices_to_text(batch['cipher'][0].tolist()[:50])}")
        print(f"Sample plain text:  {dataset.indices_to_text(batch['plain'][0].tolist()[:50])}")