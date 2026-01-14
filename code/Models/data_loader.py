"""
Data Loader for Cipher Decryption
Handles character encoding and batching
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class CharacterVocabulary:
    """Character-level vocabulary for cipher text"""
    
    def __init__(self):
        # Basic alphabet + space
        self.chars = ['<PAD>', '<UNK>'] + list('abcdefghijklmnopqrstuvwxyz') + list('0123456789')
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char2idx.get(char, 1) for char in text]  # 1 = <UNK>
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])


class CipherDataset(Dataset):
    """PyTorch Dataset for cipher pairs"""
    
    def __init__(self, data_file, vocab, max_length=512):
        """
        Args:
            data_file: Path to JSON file with cipher pairs
            vocab: CharacterVocabulary instance
            max_length: Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {Path(data_file).name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get texts
        plaintext = item['plaintext'][:self.max_length]
        ciphertext = item['ciphertext'][:self.max_length]
        
        # Encode to indices
        plain_indices = self.vocab.encode(plaintext)
        cipher_indices = self.vocab.encode(ciphertext)
        
        # Ensure same length (should be already, but safety check)
        min_len = min(len(plain_indices), len(cipher_indices))
        plain_indices = plain_indices[:min_len]
        cipher_indices = cipher_indices[:min_len]
        
        return {
            'cipher': torch.tensor(cipher_indices, dtype=torch.long),
            'plain': torch.tensor(plain_indices, dtype=torch.long),
            'length': min_len,
            'title': item['title']
        }


def collate_batch(batch):
    """
    Custom collate function to pad sequences in a batch
    """
    # Find max length in batch
    max_len = max(item['length'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    cipher_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    plain_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        length = item['length']
        cipher_batch[i, :length] = item['cipher']
        plain_batch[i, :length] = item['plain']
    
    return {
        'cipher': cipher_batch,
        'plain': plain_batch,
        'lengths': lengths,
        'titles': [item['title'] for item in batch]
    }


def create_dataloaders(cipher_type='caesar', 
                      data_dir='data/cipher_datasets',
                      batch_size=16,
                      max_length=512):
    """
    Create train, validation, and test dataloaders
    
    Args:
        cipher_type: 'caesar' or 'substitution'
        data_dir: Directory containing cipher datasets
        batch_size: Batch size for training
        max_length: Maximum sequence length
    
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    data_path = Path(data_dir) / cipher_type
    
    # Create vocabulary
    vocab = CharacterVocabulary()
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Create datasets
    train_dataset = CipherDataset(
        data_path / 'train.json',
        vocab,
        max_length=max_length
    )
    
    val_dataset = CipherDataset(
        data_path / 'val.json',
        vocab,
        max_length=max_length
    )
    
    test_dataset = CipherDataset(
        data_path / 'test.json',
        vocab,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0  # Set to 0 for Windows compatibility
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
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, vocab


# Test the data loader
if __name__ == "__main__":
    print("Testing Data Loader\n")
    print("="*60)
    
    try:
        train_loader, val_loader, test_loader, vocab = create_dataloaders(
            cipher_type='caesar',
            batch_size=4,
            max_length=512
        )
        
        print("\n" + "="*60)
        print("Sample Batch from Training Set")
        print("="*60)
        
        # Get one batch
        batch = next(iter(train_loader))
        
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Cipher shape: {batch['cipher'].shape}")
        print(f"Plain shape:  {batch['plain'].shape}")
        print(f"Lengths:      {batch['lengths'].tolist()}")
        
        print("\n" + "="*60)
        print("Sample Text Decoding")
        print("="*60)
        
        # Decode first sample
        sample_idx = 0
        cipher_text = vocab.decode(batch['cipher'][sample_idx].tolist()[:100])
        plain_text = vocab.decode(batch['plain'][sample_idx].tolist()[:100])
        
        print(f"\nTitle: {batch['titles'][sample_idx]}")
        print(f"\nCipher: {cipher_text}")
        print(f"\nPlain:  {plain_text}")
        
        print("\n✓ Data loader working correctly!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run generate_cipher_datasets.py first!")