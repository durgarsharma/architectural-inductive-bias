"""
Complete Cipher Dataset Generation - All 6 Ciphers (GREEK)
Includes: Caesar, Atbash, Affine, Vigenere, Substitution_Fixed, Substitution_Random
For Greek alphabet (24 letters)
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import unicodedata

class GreekCipherSuite:
    """All classical cipher implementations for Greek alphabet"""
    
    def __init__(self):
        # Greek lowercase alphabet (24 letters, no final sigma variant for simplicity)
        self.alphabet = 'αβγδεζηθικλμνξοπρστυφχψω'
        self.alphabet_size = len(self.alphabet)
        print(f"Greek alphabet: {self.alphabet} ({self.alphabet_size} letters)")
    
    def normalize_greek_text(self, text):
        """Normalize Greek text - remove accents, convert to lowercase"""
        # Remove combining diacritical marks
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Convert to lowercase
        text = text.lower()
        # Replace final sigma (ς) with regular sigma (σ)
        text = text.replace('ς', 'σ')
        return text
    
    def caesar_cipher(self, text, shift=3):
        """Caesar cipher with fixed shift for Greek"""
        result = []
        text = self.normalize_greek_text(text)
        for char in text:
            if char in self.alphabet:
                old_idx = self.alphabet.index(char)
                new_idx = (old_idx + shift) % self.alphabet_size
                result.append(self.alphabet[new_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def atbash_cipher(self, text):
        """Atbash: reverses alphabet (α->ω, β->ψ)"""
        result = []
        text = self.normalize_greek_text(text)
        for char in text:
            if char in self.alphabet:
                old_idx = self.alphabet.index(char)
                new_idx = (self.alphabet_size - 1) - old_idx
                result.append(self.alphabet[new_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def affine_cipher(self, text, a=5, b=8):
        """Affine: E(x) = (ax + b) mod 24"""
        result = []
        text = self.normalize_greek_text(text)
        for char in text:
            if char in self.alphabet:
                x = self.alphabet.index(char)
                encrypted_idx = (a * x + b) % self.alphabet_size
                result.append(self.alphabet[encrypted_idx])
            else:
                result.append(char)
        return ''.join(result)
    
    def vigenere_cipher(self, text, keyword="ΚΡΥΠΤΟ"):
        """Vigenere: repeating keyword cipher for Greek"""
        result = []
        text = self.normalize_greek_text(text)
        keyword = self.normalize_greek_text(keyword)
        keyword_idx = 0
        for char in text:
            if char in self.alphabet:
                shift = self.alphabet.index(keyword[keyword_idx % len(keyword)])
                old_idx = self.alphabet.index(char)
                new_idx = (old_idx + shift) % self.alphabet_size
                result.append(self.alphabet[new_idx])
                keyword_idx += 1
            else:
                result.append(char)
        return ''.join(result)
    
    def substitution_cipher(self, text, key=None):
        """Substitution with optional fixed key"""
        if key is None:
            key_list = list(self.alphabet)
            random.shuffle(key_list)
            key = ''.join(key_list)
        
        result = []
        text = self.normalize_greek_text(text)
        for char in text:
            if char in self.alphabet:
                old_idx = self.alphabet.index(char)
                result.append(key[old_idx])
            else:
                result.append(char)
        return ''.join(result), key


class GreekWikipediaDownloader:
    """Download Greek Wikipedia articles"""
    
    def __init__(self, output_dir, num_articles=1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_articles = num_articles
    
    def download_articles(self):
        """Download articles from Greek Wikipedia"""
        try:
            import wikipediaapi
        except ImportError:
            print("Installing Wikipedia-API...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'wikipedia-api', '--break-system-packages'])
            import wikipediaapi
        
        print(f"\n{'='*70}")
        print(f"Downloading {self.num_articles} articles from Greek Wikipedia (el.wikipedia.org)")
        print(f"{'='*70}\n")
        
        # Initialize Greek Wikipedia
        wiki = wikipediaapi.Wikipedia(
            language='el',
            user_agent='GreekCipherResearch/1.0'
        )
        
        articles = []
        metadata = []
        
        # Get random articles
        import requests
        
        with tqdm(total=self.num_articles, desc="Downloading articles") as pbar:
            while len(articles) < self.num_articles:
                try:
                    # Get random page titles
                    response = requests.get(
                        'https://el.wikipedia.org/w/api.php',
                        params={
                            'action': 'query',
                            'format': 'json',
                            'list': 'random',
                            'rnnamespace': 0,
                            'rnlimit': 10
                        }
                    )
                    
                    if response.status_code != 200:
                        continue
                    
                    data = response.json()
                    page_titles = [item['title'] for item in data['query']['random']]
                    
                    for title in page_titles:
                        if len(articles) >= self.num_articles:
                            break
                        
                        page = wiki.page(title)
                        
                        if page.exists() and len(page.text) > 500:
                            article_id = len(articles) + 1
                            
                            # Save article
                            filename = f"article_{article_id:04d}.txt"
                            filepath = self.output_dir / filename
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(f"Title: {page.title}\n")
                                f.write(f"URL: {page.fullurl}\n\n")
                                f.write(page.text)
                            
                            articles.append({
                                'id': article_id,
                                'title': page.title,
                                'url': page.fullurl,
                                'length': len(page.text)
                            })
                            
                            metadata.append({
                                'id': article_id,
                                'title': page.title,
                                'url': page.fullurl,
                                'length': len(page.text),
                                'filename': filename
                            })
                            
                            pbar.update(1)
                
                except Exception as e:
                    continue
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Downloaded {len(articles)} Greek articles")
        print(f"✓ Saved to: {self.output_dir}/")
        return articles


class GreekCipherDatasetGenerator:
    """Generate all 6 cipher datasets for Greek"""
    
    def __init__(self, corpus_dir, output_dir):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cipher = GreekCipherSuite()
        
        # Generate fixed key for substitution_fixed
        key_list = list(self.cipher.alphabet)
        random.seed(42)
        random.shuffle(key_list)
        self.fixed_substitution_key = ''.join(key_list)
        print(f"\n✓ Generated fixed substitution key: {self.fixed_substitution_key}\n")
    
    def load_articles(self):
        """Load all articles from corpus"""
        print(f"Loading articles from {self.corpus_dir}...")
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
        text = self.cipher.normalize_greek_text(text)
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
            
            if len(plaintext) < 100:
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
                keyword = params.get('keyword', 'ΚΡΥΠΤΟ')  # Greek "CRYPTO"
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
            
            # Save raw text files (all of them)
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
            'language': 'greek',
            'alphabet_size': self.cipher.alphabet_size,
            'total_articles': sum(len(s) for s in splits.values()),
            'total_characters': sum(p['length'] for s in splits.values() for p in s),
            'splits': {k: len(v) for k, v in splits.items()}
        }
        
        with open(cipher_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Saved JSON files to: {cipher_dir}/")
        print(f"  ✓ Saved {sum(len(s) for s in splits.values())*2} raw text files to: {raw_dir}/")
        return stats
    
    def generate_all(self, caesar_shift=3, affine_a=5, affine_b=8, vigenere_keyword="ΚΡΥΠΤΟ"):
        """Generate all 6 cipher datasets"""
        print("\n" + "="*70)
        print(" "*10 + "GENERATING ALL 6 CIPHER DATASETS (GREEK)")
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
            json.dump(all_stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print(" "*15 + "GENERATION COMPLETE! (GREEK)")
        print("="*70)
        print(f"\n{'Cipher':<25} {'Articles':<12} {'Characters':>12}")
        print("-"*70)
        for cipher_type, stats in all_stats.items():
            name = cipher_type.replace('_', ' ').title()
            print(f"{name:<25} {stats['total_articles']:<12} {stats['total_characters']:>12,}")
        print("="*70)
        
        print("\n" + "🎯 "*35)
        print("\n✅ ALL 6 GREEK DATASETS READY FOR TRAINING!")
        print(f"\n📊 Greek Alphabet: {self.cipher.alphabet}")
        print(f"   Alphabet size: {self.cipher.alphabet_size} letters (vs 26 for English)")
        print("\n💡 KEY COMPARISON WITH ENGLISH:")
        print("   • Smaller alphabet (24 vs 26) may affect pattern learning")
        print("   • Different character frequency distribution")
        print("   • Same cipher difficulty ranking expected")
        print("\n" + "🎯 "*35 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Greek Wikipedia cipher datasets')
    parser.add_argument('--download', action='store_true',
                       help='Download articles from Greek Wikipedia first')
    parser.add_argument('--num-articles', type=int, default=1000,
                       help='Number of articles to download')
    parser.add_argument('--input', type=str, default='data/wikipedia_greek',
                       help='Input directory with articles')
    parser.add_argument('--output', type=str, default='data/cipher_datasets_greek_1000',
                       help='Output directory')
    parser.add_argument('--caesar-shift', type=int, default=3)
    parser.add_argument('--affine-a', type=int, default=5)
    parser.add_argument('--affine-b', type=int, default=8)
    parser.add_argument('--vigenere-key', type=str, default='ΚΡΥΠΤΟ')
    
    args = parser.parse_args()
    
    # Download articles if requested
    if args.download:
        downloader = GreekWikipediaDownloader(args.input, args.num_articles)
        downloader.download_articles()
    
    # Generate cipher datasets
    generator = GreekCipherDatasetGenerator(args.input, args.output)
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
        subprocess.check_call(['pip', 'install', 'tqdm', '--break-system-packages'])
        from tqdm import tqdm
    
    main()