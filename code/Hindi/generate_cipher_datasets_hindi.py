"""
Complete Cipher Dataset Generation - All 6 Ciphers (Hindi/Devanagari)
Includes: Caesar, Atbash, Affine, Vigenere, Substitution_Fixed, Substitution_Random
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse

class CipherSuiteHindi:
    """All classical cipher implementations for Hindi/Devanagari"""
    
    def __init__(self):
        # Devanagari consonants and vowels
        # Hindi consonants (व्यंजन)
        self.consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        # Hindi vowels (स्वर) 
        self.vowels = 'अआइईउऊऋएऐओऔ'
        # Combining vowel marks (मात्राएँ)
        self.matras = 'ािीुूृेैोौ'
        # Additional characters
        self.additional = 'ंःँ'
        
        # Complete Hindi alphabet for cipher operations
        self.alphabet = self.consonants + self.vowels
        self.all_chars = self.consonants + self.vowels + self.matras + self.additional
        
        print(f"✓ Hindi alphabet size: {len(self.alphabet)} characters")
        print(f"  Consonants: {len(self.consonants)}")
        print(f"  Vowels: {len(self.vowels)}")
    
    def is_hindi_char(self, char):
        """Check if character is a Hindi character that should be encrypted"""
        return char in self.alphabet
    
    def caesar_cipher(self, text, shift=3):
        """Caesar cipher with fixed shift for Devanagari"""
        result = []
        for char in text:
            if self.is_hindi_char(char):
                old_idx = self.alphabet.index(char)
                new_idx = (old_idx + shift) % len(self.alphabet)
                result.append(self.alphabet[new_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def atbash_cipher(self, text):
        """Atbash: reverses alphabet for Devanagari"""
        result = []
        for char in text:
            if self.is_hindi_char(char):
                old_idx = self.alphabet.index(char)
                new_idx = len(self.alphabet) - 1 - old_idx
                result.append(self.alphabet[new_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def affine_cipher(self, text, a=5, b=8):
        """Affine: E(x) = (ax + b) mod n for Devanagari"""
        result = []
        n = len(self.alphabet)
        for char in text:
            if self.is_hindi_char(char):
                x = self.alphabet.index(char)
                encrypted_idx = (a * x + b) % n
                result.append(self.alphabet[encrypted_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def vigenere_cipher(self, text, keyword="गुप्त"):
        """Vigenere: repeating keyword cipher for Devanagari"""
        result = []
        # Filter keyword to only Hindi characters
        keyword_filtered = ''.join([c for c in keyword if self.is_hindi_char(c)])
        if not keyword_filtered:
            keyword_filtered = "गुप्त"  # Default: "secret" in Hindi
        
        keyword_idx = 0
        n = len(self.alphabet)
        
        for char in text:
            if self.is_hindi_char(char):
                shift = self.alphabet.index(keyword_filtered[keyword_idx % len(keyword_filtered)])
                old_idx = self.alphabet.index(char)
                new_idx = (old_idx + shift) % n
                result.append(self.alphabet[new_idx])
                keyword_idx += 1
            else:
                result.append(char)
        return ''.join(result)
    
    def substitution_cipher(self, text, key=None):
        """Substitution with optional fixed key for Devanagari"""
        if key is None:
            key_list = list(self.alphabet)
            random.shuffle(key_list)
            key = ''.join(key_list)
        
        result = []
        for char in text:
            if self.is_hindi_char(char):
                old_idx = self.alphabet.index(char)
                result.append(key[old_idx])
            else:
                result.append(char)
        return ''.join(result), key


class CipherDatasetGeneratorHindi:
    """Generate all 6 cipher datasets for Hindi"""
    
    def __init__(self, corpus_dir, output_dir):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cipher = CipherSuiteHindi()
        
        # Generate fixed key for substitution_fixed
        key_list = list(self.cipher.alphabet)
        random.seed(42)
        random.shuffle(key_list)
        self.fixed_substitution_key = ''.join(key_list)
        print(f"\n✓ Generated fixed substitution key (first 20 chars): {self.fixed_substitution_key[:20]}...\n")
    
    def load_articles(self):
        """Load all articles from corpus"""
        print(f"Loading Hindi articles from {self.corpus_dir}...")
        metadata_path = self.corpus_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"❌ Error: {metadata_path} not found!")
            return []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"Found {len(metadata)} articles in metadata")
        articles = []
        
        for item in tqdm(metadata, desc="Loading articles"):
            article_id = item['id']
            filename = f"article_{article_id:04d}.txt"
            filepath = self.corpus_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    articles.append({
                        'id': article_id,
                        'title': item['title'],
                        'text': text
                    })
        
        print(f"✓ Loaded {len(articles)} articles\n")
        return articles
    
    def preprocess_text(self, text, max_length=5000):
        """Clean and truncate text"""
        lines = text.split('\n')
        if lines and lines[0].startswith("Title:"):
            text = '\n'.join(lines[2:])
        # Don't lowercase for Hindi - preserve Devanagari as is
        if len(text) > max_length:
            text = text[:max_length]
        return text.strip()
    
    def create_cipher_pairs(self, articles, cipher_type, **params):
        """Create encrypted pairs for specified cipher"""
        pairs = []
        cipher_params = {}
        
        print(f"Generating {cipher_type.upper().replace('_', ' ')} cipher pairs...")
        
        for article in tqdm(articles, desc=f"Encrypting"):
            plaintext = self.preprocess_text(article['text'])
            
            if len(plaintext) < 50:
                skipped += 1
                continue
            
            # Apply appropriate cipher
            if cipher_type == 'caesar':
                ciphertext = self.cipher.caesar_cipher(plaintext, params.get('shift', 3))
                cipher_params[article['id']] = {'shift': params.get('shift', 3)}
            
            elif cipher_type == 'atbash':
                ciphertext = self.cipher.atbash_cipher(plaintext)
                cipher_params[article['id']] = {'type': 'atbash'}
            
            elif cipher_type == 'affine':
                a = params.get('a', 5)
                b = params.get('b', 8)
                ciphertext = self.cipher.affine_cipher(plaintext, a, b)
                cipher_params[article['id']] = {'a': a, 'b': b}
            
            elif cipher_type == 'vigenere':
                keyword = params.get('keyword', 'गुप्त')  # "secret" in Hindi
                ciphertext = self.cipher.vigenere_cipher(plaintext, keyword)
                cipher_params[article['id']] = {'keyword': keyword}
            
            elif cipher_type == 'substitution_fixed':
                ciphertext, _ = self.cipher.substitution_cipher(plaintext, key=self.fixed_substitution_key)
                cipher_params[article['id']] = {'key': self.fixed_substitution_key, 'type': 'fixed'}
            
            elif cipher_type == 'substitution_random':
                ciphertext, key = self.cipher.substitution_cipher(plaintext, key=None)
                cipher_params[article['id']] = {'key': key, 'type': 'random'}
            
            else:
                raise ValueError(f"Unknown cipher: {cipher_type}")
            
            pairs.append({
                'id': article['id'],
                'title': article['title'],
                'plaintext': plaintext,
                'ciphertext': ciphertext,
                'length': len(plaintext)
            })
        
        return pairs, cipher_params
    
    def split_dataset(self, pairs, train_ratio=0.7, val_ratio=0.15):
        """Split into train/val/test"""
        random.shuffle(pairs)
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': pairs[:train_end],
            'val': pairs[train_end:val_end],
            'test': pairs[val_end:]
        }
        
        print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
        return splits
    
    def save_dataset(self, cipher_type, splits, cipher_params):
        """Save dataset files including raw text samples"""
        cipher_dir = self.output_dir / cipher_type
        cipher_dir.mkdir(parents=True, exist_ok=True)
        
        # Create raw_texts directory
        raw_dir = cipher_dir / "raw_texts"
        raw_dir.mkdir(exist_ok=True)
        
        # Save JSON splits
        for split_name, split_data in splits.items():
            output_file = cipher_dir / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            
            # Create raw text directory for this split
            split_raw_dir = raw_dir / split_name
            split_raw_dir.mkdir(exist_ok=True)
            
            # Save raw text files (all of them, not just samples)
            for item in split_data:
                article_id = item['id']
                
                # Save plaintext
                plain_file = split_raw_dir / f"article_{article_id:04d}_plain.txt"
                with open(plain_file, 'w', encoding='utf-8') as f:
                    f.write(f"Article ID: {article_id}\n")
                    f.write(f"Title: {item['title']}\n")
                    f.write(f"Length: {item['length']} characters\n")
                    f.write("\n" + "="*60 + "\n\n")
                    f.write(item['plaintext'])
                
                # Save ciphertext
                cipher_file = split_raw_dir / f"article_{article_id:04d}_encrypted.txt"
                with open(cipher_file, 'w', encoding='utf-8') as f:
                    f.write(f"Article ID: {article_id}\n")
                    f.write(f"Title: {item['title']}\n")
                    f.write(f"Cipher: {cipher_type.upper()}\n")
                    f.write(f"Length: {item['length']} characters\n")
                    f.write("\n" + "="*60 + "\n\n")
                    f.write(item['ciphertext'])
        
        # Save cipher params
        with open(cipher_dir / "cipher_params.json", 'w', encoding='utf-8') as f:
            json.dump(cipher_params, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats = {
            'cipher_type': cipher_type,
            'language': 'hindi',
            'script': 'devanagari',
            'total_articles': sum(len(s) for s in splits.values()),
            'total_characters': sum(p['length'] for s in splits.values() for p in s),
            'splits': {k: len(v) for k, v in splits.items()}
        }
        
        with open(cipher_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved JSON files to: {cipher_dir}/")
        print(f"  ✓ Saved {sum(len(s) for s in splits.values())*2} raw text files to: {raw_dir}/")
        return stats
    
    def generate_all(self, caesar_shift=3, affine_a=5, affine_b=8, vigenere_keyword="गुप्त"):
        """Generate all 6 cipher datasets"""
        print("\n" + "="*70)
        print(" "*10 + "GENERATING ALL 6 CIPHER DATASETS (HINDI/DEVANAGARI)")
        print("="*70 + "\n")
        
        articles = self.load_articles()
        if not articles:
            return
        
        all_stats = {}
        
        # Define all 6 ciphers
        ciphers = [
            ('caesar', {'shift': caesar_shift}),
            ('atbash', {}),
            ('affine', {'a': affine_a, 'b': affine_b}),
            ('vigenere', {'keyword': vigenere_keyword}),
            ('substitution_fixed', {}),
            ('substitution_random', {})
        ]
        
        # Generate each cipher
        for idx, (cipher_type, params) in enumerate(ciphers, 1):
            print(f"\n[{idx}/6] {'-'*65}")
            print(f"    {cipher_type.upper().replace('_', ' ')}")
            print(f"{'-'*70}")
            
            pairs, cipher_params = self.create_cipher_pairs(articles, cipher_type, **params)
            splits = self.split_dataset(pairs)
            stats = self.save_dataset(cipher_type, splits, cipher_params)
            all_stats[cipher_type] = stats
        
        # Save overall stats
        with open(self.output_dir / "overall_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*70)
        print(" "*20 + "GENERATION COMPLETE!")
        print("="*70)
        print(f"\n{'Cipher':<25} {'Articles':<12} {'Characters':>12}")
        print("-"*70)
        for cipher_type, stats in all_stats.items():
            name = cipher_type.replace('_', ' ').title()
            print(f"{name:<25} {stats['total_articles']:<12} {stats['total_characters']:>12,}")
        print("="*70)
        
        print("\n" + "🎯 "*35)
        print("\n✅ ALL 6 HINDI DATASETS READY FOR TRAINING!")
        print("\n📊 Expected Accuracy Ranking:")
        print("   1. Caesar               → 99%+   (easiest)")
        print("   2. Atbash               → 99%+")
        print("   3. Substitution Fixed   → 95%+   (same key for all)")
        print("   4. Affine               → 90-95%")
        print("   5. Vigenère             → 85-95%")
        print("   6. Substitution Random  → 45-60% (hardest)")
        print("\n💡 KEY INSIGHT: Compare Substitution Fixed vs Random!")
        print("   This ~40% accuracy gap shows models learn cipher mappings,")
        print("   not general decryption strategies!")
        print("\n🔤 SCRIPT INFO:")
        print(f"   Devanagari alphabet size: {len(self.cipher.alphabet)} characters")
        print(f"   Vigenere keyword: {vigenere_keyword} (गुप्त = 'secret' in Hindi)")
        print("\n" + "🎯 "*35 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate all 6 cipher datasets for Hindi')
    parser.add_argument('--input', type=str, default='data/wikipedia_hindi',
                       help='Input directory with Hindi articles')
    parser.add_argument('--output', type=str, default='data/cipher_datasets_hindi_1000',
                       help='Output directory')
    parser.add_argument('--caesar-shift', type=int, default=3)
    parser.add_argument('--affine-a', type=int, default=5)
    parser.add_argument('--affine-b', type=int, default=8)
    parser.add_argument('--vigenere-key', type=str, default='गुप्त',
                       help='Vigenere keyword in Devanagari (default: गुप्त = secret)')
    
    args = parser.parse_args()
    
    generator = CipherDatasetGeneratorHindi(args.input, args.output)
    generator.generate_all(
        caesar_shift=args.caesar_shift,
        affine_a=args.affine_a,
        affine_b=args.affine_b,
        vigenere_keyword=args.vigenere_key
    )


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm
    
    main()