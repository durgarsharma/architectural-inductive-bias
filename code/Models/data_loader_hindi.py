"""
Data Loader for Hindi Cipher Datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class HindiVocabulary:
    """Vocabulary for Hindi/Devanagari characters"""
    
    def __init__(self):
        # Devanagari characters
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        vowels = 'अआइईउऊऋएऐओऔ'
        matras = 'ािीुूृेैोौ'
        additional = 'ंःँ'
        
        # Combine all Hindi characters
        hindi_chars = consonants + vowels + matras + additional
        
        # Add common punctuation and special characters
        punctuation = ' .,!?;:\'"()-\n'
        digits = '0123456789'
        
        all_chars = hindi_chars + punctuation + digits
        
        # Create vocabulary
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        for char in all_chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
        
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
        print(f"Hindi vocabulary created: {self.vocab_size} characters")
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])


class HindiCipherDataset(Dataset):
    """Dataset for Hindi cipher pairs"""
    
    def __init__(self, data_path, vocab, max_length=512):
        self.vocab = vocab
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        cipher_text = item['ciphertext'][:self.max_length]
        plain_text = item['plaintext'][:self.max_length]
        
        cipher_indices = self.vocab.encode(cipher_text)
        plain_indices = self.vocab.encode(plain_text)
        
        return {
            'cipher': cipher_indices,
            'plain': plain_indices,
            'title': item['title'],
            'length': len(plain_indices)
        }


def collate_batch_hindi(batch):
    """Collate function for batching"""
    ciphers = [item['cipher'] for item in batch]
    plains = [item['plain'] for item in batch]
    titles = [item['title'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad sequences
    max_len = max(lengths)
    
    cipher_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    plain_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (cipher, plain) in enumerate(zip(ciphers, plains)):
        cipher_padded[i, :len(cipher)] = torch.tensor(cipher)
        plain_padded[i, :len(plain)] = torch.tensor(plain)
    
    return {
        'cipher': cipher_padded,
        'plain': plain_padded,
        'titles': titles,
        'lengths': torch.tensor(lengths)
    }


def create_dataloaders_hindi(cipher_type, data_dir, batch_size=16, max_length=512):
    """Create dataloaders for Hindi cipher dataset"""
    
    data_dir = Path(data_dir)
    cipher_dir = data_dir / cipher_type
    
    # Create vocabulary
    vocab = HindiVocabulary()
    
    # Create datasets
    train_dataset = HindiCipherDataset(
        cipher_dir / 'train.json',
        vocab,
        max_length=max_length
    )
    
    val_dataset = HindiCipherDataset(
        cipher_dir / 'val.json',
        vocab,
        max_length=max_length
    )
    
    test_dataset = HindiCipherDataset(
        cipher_dir / 'test.json',
        vocab,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_hindi,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_hindi,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_hindi,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, vocab