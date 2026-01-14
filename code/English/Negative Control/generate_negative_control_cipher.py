"""
AES/DES Cipher Generator - Negative Control
============================================
Generate cryptographically secure cipher datasets that should NOT be learnable
Purpose: Verify that neural networks learn cipher patterns, not overfit
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from Crypto.Cipher import AES, DES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import hashlib


class SecureCipherGenerator:
    """Generate AES and DES encrypted datasets"""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz '
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx_to_char = {i: c for i, c in enumerate(self.alphabet)}
    
    def text_to_bytes(self, text: str) -> bytes:
        """Convert text to bytes for encryption"""
        return text.encode('utf-8')
    
    def bytes_to_indices(self, data: bytes) -> List[int]:
        """Convert encrypted bytes to character indices (modulo 27)"""
        # Map bytes to our alphabet space
        indices = [b % 27 for b in data]
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[idx % 27] for idx in indices])
    
    def aes_encrypt(self, plaintext: str, key: bytes = None) -> tuple:
        """
        AES-256 encryption
        Returns: (ciphertext_as_indices, key, iv)
        """
        if key is None:
            key = get_random_bytes(32)  # AES-256
        
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Pad and encrypt
        plaintext_bytes = self.text_to_bytes(plaintext)
        padded = pad(plaintext_bytes, AES.block_size)
        ciphertext_bytes = cipher.encrypt(padded)
        
        # Convert to indices
        ciphertext_indices = self.bytes_to_indices(ciphertext_bytes)
        ciphertext_text = self.indices_to_text(ciphertext_indices)
        
        return ciphertext_text, key, iv
    
    def des_encrypt(self, plaintext: str, key: bytes = None) -> tuple:
        """
        DES encryption
        Returns: (ciphertext_as_indices, key, iv)
        """
        if key is None:
            key = get_random_bytes(8)  # DES uses 8-byte key
        
        iv = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_CBC, iv)
        
        # Pad and encrypt
        plaintext_bytes = self.text_to_bytes(plaintext)
        padded = pad(plaintext_bytes, DES.block_size)
        ciphertext_bytes = cipher.encrypt(padded)
        
        # Convert to indices
        ciphertext_indices = self.bytes_to_indices(ciphertext_bytes)
        ciphertext_text = self.indices_to_text(ciphertext_indices)
        
        return ciphertext_text, key, iv
    
    def generate_secure_cipher_dataset(
        self,
        input_dir: str,
        output_dir: str,
        cipher_type: str = 'aes',
        use_fixed_key: bool = True
    ):
        """
        Generate AES or DES encrypted dataset from existing corpus
        
        Args:
            input_dir: Directory with plain text articles (e.g., English corpus)
            output_dir: Output directory for secure cipher dataset
            cipher_type: 'aes' or 'des'
            use_fixed_key: If True, use same key for all (like classical ciphers)
                          If False, use random key per article (harder)
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) / cipher_type.upper()
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"GENERATING {cipher_type.upper()} NEGATIVE CONTROL DATASET")
        print("="*70)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Fixed key: {use_fixed_key}")
        print("="*70 + "\n")
        
        # Generate or load fixed key
        if use_fixed_key:
            if cipher_type == 'aes':
                fixed_key = hashlib.sha256(b"NEGATIVE_CONTROL_AES").digest()[:32]
            else:
                fixed_key = hashlib.sha256(b"NEGATIVE_CONTROL_DES").digest()[:8]
            print(f"✓ Using fixed key for all articles")
        else:
            fixed_key = None
            print(f"⚠️  Using random key per article (extremely hard)")
        
        # Load articles
        articles = self.load_articles(input_path)
        
        if not articles:
            print("\n❌ No articles found!")
            return
        
        print(f"✓ Loaded {len(articles)} articles\n")
        
        # Encrypt articles
        encrypted_pairs = []
        skipped_count = 0
        
        for i, article in enumerate(articles, 1):
            plaintext = article['text'].lower()
            
            # Debug: Show first article info
            if i == 1:
                print(f"  First article length: {len(plaintext)} chars")
                print(f"  First 100 chars: {plaintext[:100]}")
            
            # Skip very short articles (reduced threshold)
            if len(plaintext) < 50:
                skipped_count += 1
                continue
            
            # Encrypt
            if cipher_type == 'aes':
                ciphertext, key, iv = self.aes_encrypt(plaintext, fixed_key)
            elif cipher_type == 'des':
                ciphertext, key, iv = self.des_encrypt(plaintext, fixed_key)
            else:
                raise ValueError(f"Unknown cipher: {cipher_type}")
            
            encrypted_pairs.append({
                'id': i,
                'title': article.get('title', f'article_{i:03d}'),
                'cipher': ciphertext,
                'plain': plaintext,
                'length': len(plaintext),
                'key': base64.b64encode(key).decode('utf-8') if not use_fixed_key else 'fixed',
                'iv': base64.b64encode(iv).decode('utf-8')
            })
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(articles)} articles...")
        
        print(f"\n✓ Encrypted {len(encrypted_pairs)} articles")
        if skipped_count > 0:
            print(f"  ⚠️  Skipped {skipped_count} articles (too short < 50 chars)")
        
        # Split dataset
        splits = self.split_dataset(encrypted_pairs)
        
        # Save
        for split_name, split_data in splits.items():
            output_file = output_path / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'cipher_type': cipher_type.upper(),
            'encryption': 'CBC mode',
            'key_length': 256 if cipher_type == 'aes' else 64,
            'fixed_key': use_fixed_key,
            'purpose': 'Negative control - should NOT be learnable',
            'expected_accuracy': '~3-4% (random baseline)',
            'total_articles': len(encrypted_pairs),
            'splits': {k: len(v) for k, v in splits.items()}
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved dataset to: {output_path}/")
        print("\n" + "="*70)
        print("⚠️  NEGATIVE CONTROL DATASET CREATED")
        print("="*70)
        print(f"Cipher: {cipher_type.upper()}")
        print(f"Articles: {len(encrypted_pairs)}")
        print(f"Expected model accuracy: ~3-5% (random)")
        print("\nPurpose: Verify models learn patterns, not overfit")
        print("If models achieve >10% accuracy on this → PROBLEM!")
        print("="*70 + "\n")
    
    def load_articles(self, input_dir: Path) -> List[Dict]:
        """Load articles from directory"""
        articles = []
        
        # Try loading from JSON first (cipher datasets)
        json_files = list(input_dir.glob('**/train.json'))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                for item in data:
                    articles.append({
                        'title': item.get('title', ''),
                        'text': item.get('plain', item.get('text', ''))
                    })
            return articles
        
        # Try loading from text files
        txt_files = list(input_dir.glob('**/*.txt'))
        for txt_file in txt_files:
            if 'article' in txt_file.name.lower():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    articles.append({
                        'title': txt_file.stem,
                        'text': text
                    })
        
        return articles
    
    def split_dataset(self, pairs: List[Dict], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Dict:
        """Split into train/val/test"""
        random.shuffle(pairs)
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': pairs[:train_end],
            'val': pairs[train_end:val_end],
            'test': pairs[val_end:]
        }


def generate_all_negative_controls(
    corpus_dirs: List[str],
    output_base: str = 'data/cipher_datasets_negative_control'
):
    """Generate AES and DES datasets for all corpora"""
    
    generator = SecureCipherGenerator()
    
    print("\n" + "="*70)
    print(" "*15 + "GENERATING ALL NEGATIVE CONTROLS")
    print("="*70)
    print(f"Output base: {output_base}")
    print(f"Corpora: {len(corpus_dirs)}")
    print("="*70 + "\n")
    
    for corpus_dir in corpus_dirs:
        corpus_name = Path(corpus_dir).parts[-1]
        output_dir = f"{output_base}/{corpus_name}"
        
        print(f"\n{'='*70}")
        print(f"Processing: {corpus_name}")
        print(f"{'='*70}")
        
        # Generate AES
        try:
            generator.generate_secure_cipher_dataset(
                input_dir=corpus_dir,
                output_dir=output_dir,
                cipher_type='aes',
                use_fixed_key=True
            )
        except Exception as e:
            print(f"❌ AES failed: {e}")
        
        # Generate DES
        try:
            generator.generate_secure_cipher_dataset(
                input_dir=corpus_dir,
                output_dir=output_dir,
                cipher_type='des',
                use_fixed_key=True
            )
        except Exception as e:
            print(f"❌ DES failed: {e}")
    
    print("\n" + "="*70)
    print(" "*15 + "ALL NEGATIVE CONTROLS GENERATED")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AES/DES negative control datasets')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input corpus directory')
    parser.add_argument('--output-dir', type=str, 
                       default='data/cipher_datasets_negative_control',
                       help='Output directory')
    parser.add_argument('--cipher', type=str, choices=['aes', 'des', 'both'],
                       default='both', help='Which cipher to generate')
    parser.add_argument('--fixed-key', action='store_true', default=True,
                       help='Use fixed key (recommended)')
    
    args = parser.parse_args()
    
    # Install pycryptodome if needed
    try:
        from Crypto.Cipher import AES, DES
    except ImportError:
        print("Installing pycryptodome...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pycryptodome', '--break-system-packages'])
        from Crypto.Cipher import AES, DES
    
    generator = SecureCipherGenerator()
    
    if args.cipher in ['aes', 'both']:
        generator.generate_secure_cipher_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cipher_type='aes',
            use_fixed_key=args.fixed_key
        )
    
    if args.cipher in ['des', 'both']:
        generator.generate_secure_cipher_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cipher_type='des',
            use_fixed_key=args.fixed_key
        )