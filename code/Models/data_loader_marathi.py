"""
Data Loader for Marathi Cipher Datasets
Handles: caesar, atbash, affine, vigenere, substitution_fixed, substitution_random
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class MarathiVocabulary:
    """Vocabulary for Marathi characters"""
    
    def __init__(self):
        # Base consonants (includes Marathi-specific ळ)
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ'
    
        # Independent vowels
        vowels = 'अआइईउऊऋएऐओऔ'
    
        # Dependent vowel signs (matras)
        matras = 'ािीुूृेैोौ'
    
        # Special Devanagari marks
        special = 'ंःँ्'
    
        # Devanagari numerals
        numerals = '०१२३४५६७८९'
    
        # Nukta (dot under consonants)
        nukta = '़'
    
        # Combine all (order matters for readability)
        self.alphabet = consonants + vowels + matras + special + numerals + nukta
    
        # Special tokens
        self.pad_token = '<PAD>'
        self.space_token = ' '
    
        # Create vocabulary
        self.chars = [self.pad_token] + list(self.alphabet) + [self.space_token]
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
        print(f"Marathi Vocabulary created:")
        print(f"  Consonants: {len(consonants)}")
        print(f"  Vowels: {len(vowels)}")
        print(f"  Matras: {len(matras)}")
        print(f"  Special marks: {len(special)}")
        print(f"  Numerals: {len(numerals)}")
        print(f"  Nukta: {len(nukta)}")
        print(f"  Total alphabet: {len(self.alphabet)}")
        print(f"  Vocab size (with PAD + SPACE): {self.vocab_size}")
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char2idx.get(char, 0) for char in text]
    
    def decode(self, indices):
        """Convert indices back to text"""
        # Skip PAD tokens (index 0) during decoding
        return ''.join([self.idx2char.get(idx, '') for idx in indices if idx != 0])


class CipherDatasetMarathi(Dataset):
    """Dataset for Marathi cipher text pairs"""
    
    def __init__(self, data_file, vocab, max_length=512):
        self.vocab = vocab
        self.max_length = max_length
        
        # Load data
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"  Loaded {len(self.data)} samples from {data_path.name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        cipher_text = item['cipher']
        plain_text = item['plain']
        title = item.get('title', f'Article {idx}')
        
        # Encode
        cipher_indices = self.vocab.encode(cipher_text)
        plain_indices = self.vocab.encode(plain_text)
        
        # Truncate if necessary
        if len(cipher_indices) > self.max_length:
            cipher_indices = cipher_indices[:self.max_length]
            plain_indices = plain_indices[:self.max_length]
        
        return {
            'cipher': cipher_indices,
            'plain': plain_indices,
            'title': title,
            'length': len(cipher_indices)
        }


def collate_batch_marathi(batch):
    """Collate function for batching"""
    # Find max length in batch
    max_len = max(item['length'] for item in batch)
    
    # Pad sequences
    cipher_batch = []
    plain_batch = []
    lengths = []
    titles = []
    
    for item in batch:
        cipher = item['cipher']
        plain = item['plain']
        length = item['length']
        title = item['title']
        
        # Pad
        cipher_padded = cipher + [0] * (max_len - length)
        plain_padded = plain + [0] * (max_len - length)
        
        cipher_batch.append(cipher_padded)
        plain_batch.append(plain_padded)
        lengths.append(length)
        titles.append(title)
    
    return {
        'cipher': torch.LongTensor(cipher_batch),
        'plain': torch.LongTensor(plain_batch),
        'lengths': torch.LongTensor(lengths),
        'titles': titles
    }


def create_dataloaders_marathi(cipher_type='caesar', 
                               data_dir='data/cipher_datasets_marathi_1000',
                               batch_size=16,
                               max_length=512):
    """
    Create train, validation, and test dataloaders for Marathi
    
    Args:
        cipher_type: Type of cipher (caesar, atbash, etc.)
        data_dir: Directory containing the cipher datasets
        batch_size: Batch size for training
        max_length: Maximum sequence length
    
    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    data_dir = Path(data_dir)
    
    print(f"\nLoading Marathi data for {cipher_type} cipher...")
    print(f"Data directory: {data_dir}")
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print(f"\nPlease run cipher generation first:")
        print(f"  python generate_cipher_datasets_marathi.py --cipher {cipher_type}")
        return None, None, None, None
    
    # Check if cipher directory exists
    cipher_dir = data_dir / cipher_type
    if not cipher_dir.exists():
        print(f"\n❌ Error: Cipher directory not found: {cipher_dir}")
        print(f"\nAvailable cipher directories:")
        for d in data_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        print(f"\nPlease run cipher generation for {cipher_type}:")
        print(f"  python generate_cipher_datasets_marathi.py --cipher {cipher_type}")
        return None, None, None, None
    
    # File paths
    train_file = cipher_dir / 'train.json'
    val_file = cipher_dir / 'val.json'
    test_file = cipher_dir / 'test.json'
    
    # Check if files exist
    missing_files = []
    for file in [train_file, val_file, test_file]:
        if not file.exists():
            missing_files.append(file.name)
    
    if missing_files:
        print(f"\n❌ Error: Missing data files: {', '.join(missing_files)}")
        print(f"In directory: {cipher_dir}")
        print(f"\nPlease run cipher generation:")
        print(f"  python generate_cipher_datasets_marathi.py --cipher {cipher_type}")
        return None, None, None, None
    
    # Create vocabulary
    print("\nCreating Marathi vocabulary...")
    vocab = MarathiVocabulary()
    
    # Create datasets
    print("\nLoading datasets...")
    try:
        train_dataset = CipherDatasetMarathi(train_file, vocab, max_length)
        val_dataset = CipherDatasetMarathi(val_file, vocab, max_length)
        test_dataset = CipherDatasetMarathi(test_file, vocab, max_length)
    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        return None, None, None, None
    
    # Create dataloaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_marathi,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_marathi,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_marathi,
        num_workers=0
    )
    
    print(f"\n{'='*60}")
    print(f"Marathi Data Loaders Created Successfully")
    print(f"{'='*60}")
    print(f"Cipher type: {cipher_type}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Test the data loader
    print("\n" + "="*60)
    print("Testing Marathi Data Loader")
    print("="*60)
    
    try:
        train_loader, val_loader, test_loader, vocab = create_dataloaders_marathi(
            cipher_type='caesar',
            data_dir='data/cipher_datasets_marathi_1000',
            batch_size=4
        )
        
        if train_loader is None:
            print("\n❌ Data loader creation failed. See errors above.")
            exit(1)
        
        print("\n✓ Data loaders created successfully!")
        
        # Test one batch
        print("\nTesting one batch...")
        batch = next(iter(train_loader))
        
        print(f"Cipher shape: {batch['cipher'].shape}")
        print(f"Plain shape: {batch['plain'].shape}")
        print(f"Lengths: {batch['lengths']}")
        print(f"\nSample (first 100 chars):")
        print(f"Cipher: {vocab.decode(batch['cipher'][0][:100].tolist())}")
        print(f"Plain:  {vocab.decode(batch['plain'][0][:100].tolist())}")
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()