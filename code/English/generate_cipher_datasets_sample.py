"""
Cipher Dataset Generation
Step 0, Part 2: Create encrypted training data from Wikipedia corpus
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import pickle

class CipherSuite:
    """Classical cipher implementations"""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def caesar_cipher(self, text, shift=3):
        """Caesar cipher with configurable shift"""
        result = []
        text = text.lower()
        
        for char in text:
            if char in self.alphabet:
                old_idx = self.alphabet.index(char)
                new_idx = (old_idx + shift) % 26
                result.append(self.alphabet[new_idx])
            else:
                result.append(char)
        
        return ''.join(result)
    
    def substitution_cipher(self, text, key=None):
        """Simple substitution cipher with random or fixed key"""
        if key is None:
            # Generate random substitution key
            key_list = list(self.alphabet)
            random.shuffle(key_list)
            key = ''.join(key_list)
        
        result = []
        text = text.lower()
        
        for char in text:
            if char in self.alphabet:
                old_idx = self.alphabet.index(char)
                result.append(key[old_idx])
            else:
                result.append(char)
        
        return ''.join(result), key


class CipherDatasetGenerator:
    """Generate cipher datasets from Wikipedia corpus"""
    
    def __init__(self, corpus_dir="data/wikipedia_sample", output_dir="data/cipher_datasets"):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cipher = CipherSuite()
    
    def load_articles(self):
        """Load all Wikipedia articles"""
        print("Loading Wikipedia articles...")
        articles = []
        
        # Load metadata
        metadata_path = self.corpus_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load each article
        for item in metadata:
            article_id = item['id']
            filename = f"article_{article_id:04d}.txt"
            filepath = self.corpus_dir / filename
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                articles.append({
                    'id': article_id,
                    'title': item['title'],
                    'text': text
                })
        
        print(f"✓ Loaded {len(articles)} articles")
        return articles
    
    def preprocess_text(self, text, max_length=5000):
        """
        Preprocess text for cipher encryption
        - Remove title line
        - Clean text
        - Truncate to max_length
        """
        lines = text.split('\n')
        # Skip first line (title) if it starts with "Title:"
        if lines and lines[0].startswith("Title:"):
            text = '\n'.join(lines[2:])  # Skip title and empty line
        
        # Keep only alphanumeric and basic punctuation
        # Convert to lowercase, keep spaces
        text = text.lower()
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def create_cipher_pairs(self, articles, cipher_type='caesar', caesar_shift=3):
        """
        Create (plaintext, ciphertext) pairs
        """
        pairs = []
        cipher_params = {}
        
        print(f"\nGenerating {cipher_type.upper()} cipher pairs...")
        
        for article in tqdm(articles):
            plaintext = self.preprocess_text(article['text'])
            
            if len(plaintext) < 100:  # Skip very short texts
                continue
            
            if cipher_type == 'caesar':
                ciphertext = self.cipher.caesar_cipher(plaintext, caesar_shift)
                cipher_params[article['id']] = {'shift': caesar_shift}
            
            elif cipher_type == 'substitution':
                ciphertext, key = self.cipher.substitution_cipher(plaintext)
                cipher_params[article['id']] = {'key': key}
            
            pairs.append({
                'id': article['id'],
                'title': article['title'],
                'plaintext': plaintext,
                'ciphertext': ciphertext,
                'length': len(plaintext)
            })
        
        return pairs, cipher_params
    
    def split_dataset(self, pairs, train_ratio=0.7, val_ratio=0.15):
        """
        Split dataset into train/val/test
        """
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': pairs[:train_end],
            'val': pairs[train_end:val_end],
            'test': pairs[val_end:]
        }
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(splits['train'])} articles")
        print(f"  Val:   {len(splits['val'])} articles")
        print(f"  Test:  {len(splits['test'])} articles")
        
        return splits
    
    def save_dataset(self, cipher_type, splits, cipher_params):
        """
        Save dataset to disk
        """
        cipher_dir = self.output_dir / cipher_type
        cipher_dir.mkdir(parents=True, exist_ok=True)
        
        # Create raw text directory for inspection
        raw_dir = cipher_dir / "raw_texts"
        raw_dir.mkdir(exist_ok=True)
        
        # Save splits
        for split_name, split_data in splits.items():
            output_file = cipher_dir / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved {split_name}.json ({len(split_data)} samples)")
            
            # Save raw text files for inspection
            split_raw_dir = raw_dir / split_name
            split_raw_dir.mkdir(exist_ok=True)
            
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
        
        print(f"  ✓ Saved raw text files in {raw_dir}/")
        print(f"    (Check these files to inspect encryption visually)")
        
        # Save cipher parameters
        params_file = cipher_dir / "cipher_params.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(cipher_params, f, indent=2)
        
        # Save statistics
        total_chars_plain = sum(p['length'] for split in splits.values() for p in split)
        stats = {
            'cipher_type': cipher_type,
            'total_articles': sum(len(split) for split in splits.values()),
            'total_characters': total_chars_plain,
            'splits': {k: len(v) for k, v in splits.items()}
        }
        
        stats_file = cipher_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Dataset saved to: {cipher_dir}")
        return stats
    
    def generate_all_datasets(self, caesar_shift=3):
        """
        Generate datasets for both cipher types
        """
        print("\n" + "="*60)
        print("CIPHER DATASET GENERATION")
        print("="*60)
        
        # Load articles
        articles = self.load_articles()
        
        all_stats = {}
        
        # Generate Caesar cipher dataset
        print("\n--- CAESAR CIPHER ---")
        caesar_pairs, caesar_params = self.create_cipher_pairs(
            articles, 
            cipher_type='caesar', 
            caesar_shift=caesar_shift
        )
        caesar_splits = self.split_dataset(caesar_pairs)
        caesar_stats = self.save_dataset('caesar', caesar_splits, caesar_params)
        all_stats['caesar'] = caesar_stats
        
        # Generate Substitution cipher dataset
        print("\n--- SUBSTITUTION CIPHER ---")
        sub_pairs, sub_params = self.create_cipher_pairs(
            articles, 
            cipher_type='substitution'
        )
        sub_splits = self.split_dataset(sub_pairs)
        sub_stats = self.save_dataset('substitution', sub_splits, sub_params)
        all_stats['substitution'] = sub_stats
        
        # Save overall statistics
        overall_stats_file = self.output_dir / "overall_statistics.json"
        with open(overall_stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2)
        
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"\nDatasets saved in: {self.output_dir}")
        print("\nGenerated datasets:")
        for cipher_type, stats in all_stats.items():
            print(f"  • {cipher_type.upper()}: {stats['total_articles']} articles, {stats['total_characters']:,} characters")


def main():
    """Main execution"""
    generator = CipherDatasetGenerator(
        corpus_dir="data/wikipedia_sample",
        output_dir="data/cipher_datasets"
    )
    
    generator.generate_all_datasets(caesar_shift=3)
    
    print("\n✅ Ready for model training!")
    print("\nNext steps:")
    print("  1. Inspect datasets in 'data/cipher_datasets/'")
    print("  2. Proceed to model implementation")


if __name__ == "__main__":
    # Install tqdm if needed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm
    
    main()