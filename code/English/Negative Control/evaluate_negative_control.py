"""
Comprehensive Evaluation Script for MORPH Cipher Models
Includes: Character accuracy, Word accuracy, Edit distance, Confusion matrix, Distributions
Works with MLP, LSTM, CNN, Transformer
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys

# Import models
sys.path.append('models')
from mlp_cipher_model import MLPCipherModel
from lstm_cipher_model import LSTMCipherModel
from cnn_cipher_model import CNNCipherModel
from transformer_cipher_model import TransformerCipherModel
from data_loader_cipher import get_data_loaders


class CipherEvaluator:
    """Comprehensive evaluation for cipher decryption models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz '
        
        # Character mappings
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx_to_char = {i: c for i, c in enumerate(self.alphabet)}
    
    def indices_to_text(self, indices):
        """Convert indices to text"""
        return ''.join([self.idx_to_char.get(idx, ' ') for idx in indices])
    
    def calculate_character_accuracy(self, pred_text, target_text):
        """Character-level accuracy"""
        if len(pred_text) != len(target_text):
            min_len = min(len(pred_text), len(target_text))
            pred_text = pred_text[:min_len]
            target_text = target_text[:min_len]
        
        correct = sum(p == t for p, t in zip(pred_text, target_text))
        return correct / len(target_text) if len(target_text) > 0 else 0.0
    
    def calculate_word_accuracy(self, pred_text, target_text):
        """Word-level accuracy"""
        pred_words = pred_text.split()
        target_words = target_text.split()
        
        if len(pred_words) != len(target_words):
            min_len = min(len(pred_words), len(target_words))
            pred_words = pred_words[:min_len]
            target_words = target_words[:min_len]
        
        correct = sum(p == t for p, t in zip(pred_words, target_words))
        return correct / len(target_words) if len(target_words) > 0 else 0.0
    
    def calculate_edit_distance(self, pred_text, target_text):
        """Normalized edit distance (using simple implementation)"""
        # Simple Levenshtein distance
        if len(pred_text) == 0 or len(target_text) == 0:
            return 0.0
        
        # Create distance matrix
        m, n = len(pred_text), len(target_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_text[i-1] == target_text[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        distance = dp[m][n]
        max_len = max(len(pred_text), len(target_text))
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    
    def build_confusion_matrix(self, pred_text, target_text):
        """Build character confusion matrix"""
        confusion = {}
        for pred_char, target_char in zip(pred_text, target_text):
            if target_char in self.alphabet:
                if target_char not in confusion:
                    confusion[target_char] = Counter()
                confusion[target_char][pred_char] += 1
        return confusion
    
    def evaluate_dataset(self, dataloader, max_samples=None):
        """Comprehensive evaluation"""
        all_results = []
        all_char_accuracies = []
        all_word_accuracies = []
        all_edit_distances = []
        global_confusion = {char: Counter() for char in self.alphabet if char != ' '}
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                lengths = batch['lengths']
                
                # Forward pass
                logits = self.model(cipher)
                predictions = torch.argmax(logits, dim=-1)
                
                # Process each sample in batch
                for i, length in enumerate(lengths):
                    pred = predictions[i, :length]
                    targ = plain[i, :length]
                    ciph = cipher[i, :length]
                    
                    # Convert to text
                    pred_text = self.indices_to_text(pred.cpu().tolist())
                    target_text = self.indices_to_text(targ.cpu().tolist())
                    cipher_text = self.indices_to_text(ciph.cpu().tolist())
                    
                    # Calculate metrics
                    char_acc = self.calculate_character_accuracy(pred_text, target_text)
                    word_acc = self.calculate_word_accuracy(pred_text, target_text)
                    edit_dist = self.calculate_edit_distance(pred_text, target_text)
                    
                    # Update confusion matrix
                    confusion = self.build_confusion_matrix(pred_text, target_text)
                    for char, counts in confusion.items():
                        if char != ' ':
                            global_confusion[char].update(counts)
                    
                    # Store metrics
                    all_char_accuracies.append(char_acc)
                    all_word_accuracies.append(word_acc)
                    all_edit_distances.append(edit_dist)
                    
                    # Store sample details
                    all_results.append({
                        'sample_id': sample_count,
                        'cipher': cipher_text[:200],
                        'predicted': pred_text[:200],
                        'target': target_text[:200],
                        'char_accuracy': char_acc,
                        'word_accuracy': word_acc,
                        'edit_distance': edit_dist,
                        'length': length.item()
                    })
                    
                    sample_count += 1
                    
                    if max_samples and sample_count >= max_samples:
                        break
                
                if max_samples and sample_count >= max_samples:
                    break
        
        return {
            'results': all_results,
            'char_accuracies': all_char_accuracies,
            'word_accuracies': all_word_accuracies,
            'edit_distances': all_edit_distances,
            'confusion_matrix': global_confusion,
            'summary': {
                'avg_char_accuracy': np.mean(all_char_accuracies),
                'avg_word_accuracy': np.mean(all_word_accuracies),
                'avg_edit_distance': np.mean(all_edit_distances),
                'std_char_accuracy': np.std(all_char_accuracies),
                'min_char_accuracy': np.min(all_char_accuracies),
                'max_char_accuracy': np.max(all_char_accuracies),
                'total_samples': len(all_results)
            }
        }
    
    def plot_confusion_matrix(self, confusion, save_path, top_k=15):
        """Plot top-k most confused characters"""
        confusions_list = []
        for target_char, pred_counts in confusion.items():
            for pred_char, count in pred_counts.items():
                if pred_char != target_char and pred_char in self.alphabet:
                    confusions_list.append((target_char, pred_char, count))
        
        confusions_list.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confusions_list[:top_k]
        
        if not top_confusions:
            print("  ✓ No confusions found (perfect accuracy!)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        labels = [f"{t}→{p}" for t, p, _ in top_confusions]
        counts = [c for _, _, c in top_confusions]
        
        bars = ax.barh(labels, counts, color='coral')
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_k} Character Confusions')
        ax.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {int(count)}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion matrix to {save_path}")
    
    def plot_distributions(self, results, save_path):
        """Plot accuracy distributions"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Character accuracy
        axes[0].hist(results['char_accuracies'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Character Accuracy')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Character Accuracy Distribution')
        axes[0].axvline(results['summary']['avg_char_accuracy'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {results['summary']['avg_char_accuracy']:.3f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Word accuracy
        axes[1].hist(results['word_accuracies'], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Word Accuracy')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Word Accuracy Distribution')
        axes[1].axvline(results['summary']['avg_word_accuracy'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {results['summary']['avg_word_accuracy']:.3f}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Edit distance
        axes[2].hist(results['edit_distances'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Edit Distance Similarity')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Edit Distance Distribution')
        axes[2].axvline(results['summary']['avg_edit_distance'], color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {results['summary']['avg_edit_distance']:.3f}")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved distributions to {save_path}")


def load_model(model_type, model_path, device='cpu'):
    """Load trained model"""
    
    if model_type == 'mlp':
        model = MLPCipherModel(
            vocab_size=27,
            embedding_dim=128,
            hidden_dims=[512, 256, 128]
        )
    elif model_type == 'lstm':
        model = LSTMCipherModel(
            vocab_size=27,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            bidirectional=True
        )
    elif model_type == 'cnn':
        model = CNNCipherModel(
            vocab_size=27,
            embedding_dim=128,
            num_filters=256,
            kernel_sizes=[3, 5, 7]
        )
    elif model_type == 'transformer':
        model = TransformerCipherModel(
            vocab_size=27,
            d_model=256,
            nhead=8,
            num_layers=4
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_model(model_type, cipher_type, data_dir='data/cipher_datasets_linguistic_250/MORPH', 
                   results_base='results/linguistic/MORPH', device='cpu'):
    """Comprehensive evaluation for one model-cipher combination"""
    
    model_name = f"{model_type}_{cipher_type}"
    results_dir = Path(results_base) / model_name
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_type.upper()} on {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    # Check if model exists
    model_path = results_dir / 'best_model.pt'
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train first: python train_cipher_model.py --model {model_type} --cipher {cipher_type}")
        return None
    
    # Load config
    config_path = results_dir / 'config.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(model_type, model_path, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(
        data_dir=data_dir,
        cipher_type=cipher_type,
        batch_size=16,
        max_length=512
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    evaluator = CipherEvaluator(model, device)
    results = evaluator.evaluate_dataset(test_loader)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_type.upper()} - {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    if checkpoint:
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_acc', 0)
        print(f"\n🏋️  Training Info:")
        print(f"  Model Type:          {model_type.upper()}")
        print(f"  Epochs Trained:      {epoch}")
        print(f"  Best Val Accuracy:   {val_acc*100:>6.2f}%")
        print(f"  Total Parameters:    {total_params:,}")
    
    print(f"\n📊 Test Set Metrics:")
    print(f"  Character Accuracy:  {results['summary']['avg_char_accuracy']*100:>6.2f}%")
    print(f"  Word Accuracy:       {results['summary']['avg_word_accuracy']*100:>6.2f}%")
    print(f"  Edit Distance Sim:   {results['summary']['avg_edit_distance']*100:>6.2f}%")
    
    print(f"\n📈 Statistics:")
    print(f"  Std Dev (char):      {results['summary']['std_char_accuracy']*100:>6.2f}%")
    print(f"  Min Accuracy:        {results['summary']['min_char_accuracy']*100:>6.2f}%")
    print(f"  Max Accuracy:        {results['summary']['max_char_accuracy']*100:>6.2f}%")
    print(f"  Total Samples:       {results['summary']['total_samples']}")
    
    # Save results
    print(f"\nSaving results...")
    
    # Detailed results
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'summary': results['summary'],
            'config': config,
            'model_info': {
                'type': model_type,
                'parameters': total_params,
                'epoch': checkpoint.get('epoch', 'unknown') if checkpoint else 'unknown'
            },
            'per_sample': results['results'][:100]  # Save first 100 samples
        }, f, indent=2)
    
    # Test summary
    with open(results_dir / 'test_summary.json', 'w') as f:
        json.dump({
            'model_type': model_type,
            'cipher_type': cipher_type,
            **results['summary']
        }, f, indent=2)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        results_dir / 'confusion_matrix.png'
    )
    
    # Plot distributions
    evaluator.plot_distributions(results, results_dir / 'accuracy_distributions.png')
    
    print(f"\n✓ All results saved to {results_dir}/")
    print(f"{'='*80}\n")
    
    return results


def evaluate_all_ciphers(model_type, data_dir='data/cipher_datasets_linguistic_250/MORPH',
                         results_base='results/linguistic/MORPH'):
    """Evaluate one model on all 6 ciphers"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ciphers = ['caesar', 'atbash', 'affine', 'vigenere',
               'substitution_fixed', 'substitution_random']
    
    all_results = {}
    
    for cipher in ciphers:
        result = evaluate_model(model_type, cipher, data_dir, results_base, device)
        if result:
            all_results[cipher] = result['summary']
    
    if not all_results:
        print("\n❌ No results to compare. Train models first!")
        return
    
    # Comparison table
    print("\n" + "="*100)
    print(f"{model_type.upper()} - ALL CIPHERS COMPARISON")
    print("="*100)
    print(f"\n{'Cipher':<25} {'Char Acc':<12} {'Word Acc':<12} {'Edit Sim':<12} {'Std Dev':<10}")
    print("-"*100)
    
    for cipher, summary in all_results.items():
        display_name = cipher.replace('_', ' ').title()
        print(f"{display_name:<25} "
              f"{summary['avg_char_accuracy']*100:>10.2f}% "
              f"{summary['avg_word_accuracy']*100:>10.2f}% "
              f"{summary['avg_edit_distance']*100:>10.2f}% "
              f"{summary['std_char_accuracy']*100:>8.2f}%")
    
    print("="*100)
    
    # Save comparison
    comparison_path = Path(results_base) / f'{model_type}_all_ciphers_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cipher_names = [c.replace('_', '\n').title() for c in all_results.keys()]
    char_accs = [s['avg_char_accuracy']*100 for s in all_results.values()]
    word_accs = [s['avg_word_accuracy']*100 for s in all_results.values()]
    
    x = np.arange(len(cipher_names))
    width = 0.35
    
    ax.bar(x - width/2, char_accs, width, label='Character Accuracy', color='steelblue')
    ax.bar(x + width/2, word_accs, width, label='Word Accuracy', color='coral')
    
    ax.set_xlabel('Cipher Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model_type.upper()} Performance Across All Ciphers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cipher_names, rotation=0, ha='center')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    comparison_plot = Path(results_base) / f'{model_type}_all_ciphers_comparison.png'
    plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison to {comparison_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive evaluation for MORPH cipher models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp', 'lstm', 'cnn', 'transformer'],
                       help='Model architecture')
    parser.add_argument('--cipher', type=str, default='all',
                       choices=['caesar', 'atbash', 'affine', 'vigenere',
                               'substitution_fixed', 'substitution_random',
                               'aes', 'des', 'all'],
                       help='Cipher type (or "all" for all ciphers)')
    parser.add_argument('--data-dir', type=str,
                       default='data/cipher_datasets_linguistic_250/MORPH',
                       help='Data directory')
    parser.add_argument('--results-base', type=str,
                       default='results/linguistic/MORPH',
                       help='Results base directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.cipher == 'all':
        evaluate_all_ciphers(args.model, args.data_dir, args.results_base)
    else:
        evaluate_model(args.model, args.cipher, args.data_dir, args.results_base, device)