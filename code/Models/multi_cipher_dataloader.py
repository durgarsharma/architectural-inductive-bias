"""
Multi-Cipher Data Loader for Zero-Shot Learning
Loads and combines multiple ciphers for training, holds out one for testing
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json
from pathlib import Path
from typing import List, Tuple


class MultiCipherDataset(Dataset):
    """Dataset that combines multiple ciphers"""
    
    def __init__(self, cipher_types: List[str], data_dir: str, split: str = 'train'):
        """
        Args:
            cipher_types: List of cipher names to include (e.g., ['caesar', 'atbash'])
            data_dir: Base directory containing cipher datasets
            split: 'train', 'val', or 'test'
        """
        self.cipher_types = cipher_types
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load all cipher data
        self.samples = []
        self.cipher_labels = []
        
        for cipher_idx, cipher_type in enumerate(cipher_types):
            cipher_data_path = self.data_dir / cipher_type / f"{split}.json"
            
            if not cipher_data_path.exists():
                print(f"⚠️  Warning: {cipher_data_path} not found, skipping {cipher_type}")
                continue
            
            with open(cipher_data_path, 'r') as f:
                cipher_samples = json.load(f)
            
            # Add cipher label to each sample
            for sample in cipher_samples:
                sample['cipher_type'] = cipher_type
                sample['cipher_label'] = cipher_idx
                self.samples.append(sample)
                self.cipher_labels.append(cipher_idx)
        
        print(f"✓ Loaded {len(self.samples)} samples from {len(cipher_types)} ciphers ({split})")
        for cipher_type in cipher_types:
            count = sum(1 for s in self.samples if s['cipher_type'] == cipher_type)
            print(f"  - {cipher_type}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Vocabulary:
    """Simple character vocabulary"""
    
    def __init__(self):
        self.char_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        
        # Add lowercase letters
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz', start=2):
            self.char_to_idx[c] = i
        
        # Add digits
        for i, c in enumerate('0123456789', start=28):
            self.char_to_idx[c] = i
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx.get(c, 1) for c in text.lower()]
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char.get(i, '<UNK>') for i in indices if i != 0])


def collate_batch(batch):
    """Collate function for batching"""
    ciphers = []
    plains = []
    lengths = []
    titles = []
    cipher_types = []
    cipher_labels = []
    
    vocab = Vocabulary()
    
    # Handle different possible key names in the data
    def get_cipher_text(sample):
        if 'ciphertext' in sample:
            return sample['ciphertext']
        elif 'cipher_text' in sample:
            return sample['cipher_text']
        elif 'encrypted' in sample:
            return sample['encrypted']
        elif 'cipher' in sample:
            return sample['cipher']
        else:
            raise KeyError(f"Could not find cipher text in sample. Keys: {sample.keys()}")
    
    def get_plain_text(sample):
        if 'plaintext' in sample:
            return sample['plaintext']
        elif 'plain_text' in sample:
            return sample['plain_text']
        elif 'plain' in sample:
            return sample['plain']
        elif 'original' in sample:
            return sample['original']
        else:
            raise KeyError(f"Could not find plain text in sample. Keys: {sample.keys()}")
    
    # Get max length
    max_len = max(len(get_cipher_text(sample)) for sample in batch)
    
    for sample in batch:
        cipher_text = get_cipher_text(sample)
        plain_text = get_plain_text(sample)
        
        cipher_encoded = vocab.encode(cipher_text)
        plain_encoded = vocab.encode(plain_text)
        
        # Pad sequences
        cipher_padded = cipher_encoded + [0] * (max_len - len(cipher_encoded))
        plain_padded = plain_encoded + [0] * (max_len - len(plain_encoded))
        
        ciphers.append(cipher_padded)
        plains.append(plain_padded)
        lengths.append(len(cipher_encoded))
        titles.append(sample.get('title', 'Unknown'))
        cipher_types.append(sample.get('cipher_type', 'unknown'))
        cipher_labels.append(sample.get('cipher_label', -1))
    
    return {
        'cipher': torch.tensor(ciphers, dtype=torch.long),
        'plain': torch.tensor(plains, dtype=torch.long),
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'titles': titles,
        'cipher_types': cipher_types,
        'cipher_labels': torch.tensor(cipher_labels, dtype=torch.long)
    }


def create_zeroshot_dataloaders(
    train_ciphers: List[str],
    test_cipher: str,
    data_dir: str = 'data/cipher_datasets_1000',
    batch_size: int = 16,
    max_length: int = 512
):
    """
    Create dataloaders for zero-shot learning
    
    Args:
        train_ciphers: List of ciphers to train on (e.g., ['caesar', 'atbash', 'affine'])
        test_cipher: Cipher to hold out for testing (e.g., 'vigenere')
        data_dir: Base directory
        batch_size: Batch size
        max_length: Maximum sequence length
    
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    
    print(f"\n{'='*60}")
    print("Creating Zero-Shot Dataloaders")
    print(f"{'='*60}")
    print(f"Training ciphers: {train_ciphers}")
    print(f"Test cipher (zero-shot): {test_cipher}")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_dataset = MultiCipherDataset(train_ciphers, data_dir, split='train')
    val_dataset = MultiCipherDataset(train_ciphers, data_dir, split='val')
    test_dataset = MultiCipherDataset([test_cipher], data_dir, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    vocab = Vocabulary()
    
    print(f"\n✓ Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Vocab size: {vocab.vocab_size}\n")
    
    return train_loader, val_loader, test_loader, vocab


def get_all_zeroshot_configs():
    """
    Get all 6 zero-shot configurations (leave-one-cipher-out)
    
    Returns:
        List of (train_ciphers, test_cipher) tuples
    """
    all_ciphers = ['caesar', 'atbash', 'affine', 'vigenere', 
                   'substitution_fixed', 'substitution_random']
    
    configs = []
    for test_cipher in all_ciphers:
        train_ciphers = [c for c in all_ciphers if c != test_cipher]
        configs.append((train_ciphers, test_cipher))
    
    return configs


# Test the data loader
if __name__ == "__main__":
    print("Testing Multi-Cipher Data Loader\n")
    
    # Test configuration
    train_ciphers = ['caesar', 'atbash', 'affine']
    test_cipher = 'vigenere'
    
    train_loader, val_loader, test_loader, vocab = create_zeroshot_dataloaders(
        train_ciphers=train_ciphers,
        test_cipher=test_cipher,
        batch_size=4
    )
    
    print("\n" + "="*60)
    print("Testing batch loading...")
    print("="*60)
    
    # Get one batch from each loader
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    print(f"\nTrain batch:")
    print(f"  Cipher shape: {train_batch['cipher'].shape}")
    print(f"  Plain shape:  {train_batch['plain'].shape}")
    print(f"  Cipher types: {set(train_batch['cipher_types'])}")
    
    print(f"\nTest batch (zero-shot):")
    print(f"  Cipher shape: {test_batch['cipher'].shape}")
    print(f"  Plain shape:  {test_batch['plain'].shape}")
    print(f"  Cipher types: {set(test_batch['cipher_types'])}")
    
    print("\n✓ Multi-cipher data loader working correctly!")
    
    print("\n" + "="*60)
    print("All Zero-Shot Configurations:")
    print("="*60)
    
    configs = get_all_zeroshot_configs()
    for i, (train_ciphers, test_cipher) in enumerate(configs, 1):
        print(f"\n{i}. Test cipher: {test_cipher.upper()}")
        print(f"   Train on: {', '.join(train_ciphers)}")