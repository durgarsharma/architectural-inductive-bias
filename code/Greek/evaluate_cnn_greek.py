"""
Comprehensive Greek CNN Evaluation - All 6 Ciphers
Includes: Character accuracy, Word accuracy, Edit distance, Per-article breakdown, Confusion matrix
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import Levenshtein

from models.cnn_cipher_model_greek import GreekCharCNNDecryptor, GreekDeepCharCNNDecryptor, count_parameters
from models.data_loader_greek import create_dataloaders


class GreekCNNEvaluator:
    """Comprehensive evaluation for Greek CNN models"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
        self.greek_alphabet = 'αβγδεζηθικλμνξοπρστυφχψω'
    
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
        """Normalized Levenshtein distance"""
        distance = Levenshtein.distance(pred_text, target_text)
        max_len = max(len(pred_text), len(target_text))
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    
    def build_confusion_matrix(self, pred_text, target_text):
        """Build character confusion matrix for Greek alphabet"""
        confusion = {}
        for pred_char, target_char in zip(pred_text, target_text):
            if target_char in self.greek_alphabet:
                if target_char not in confusion:
                    confusion[target_char] = Counter()
                confusion[target_char][pred_char] += 1
        return confusion
    
    def evaluate_dataset(self, dataloader):
        """Comprehensive evaluation"""
        all_results = []
        all_char_accuracies = []
        all_word_accuracies = []
        all_edit_distances = []
        global_confusion = {char: Counter() for char in self.greek_alphabet}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                lengths = batch['lengths']
                
                outputs = self.model(cipher)
                predictions = outputs.argmax(dim=-1)
                
                for i, length in enumerate(lengths):
                    pred = predictions[i, :length]
                    targ = plain[i, :length]
                    
                    pred_text = self.vocab.decode(pred.cpu().tolist())
                    target_text = self.vocab.decode(targ.cpu().tolist())
                    cipher_text = self.vocab.decode(cipher[i, :length].cpu().tolist())
                    
                    char_acc = self.calculate_character_accuracy(pred_text, target_text)
                    word_acc = self.calculate_word_accuracy(pred_text, target_text)
                    edit_dist = self.calculate_edit_distance(pred_text, target_text)
                    
                    confusion = self.build_confusion_matrix(pred_text, target_text)
                    for char, counts in confusion.items():
                        global_confusion[char].update(counts)
                    
                    all_char_accuracies.append(char_acc)
                    all_word_accuracies.append(word_acc)
                    all_edit_distances.append(edit_dist)
                    
                    all_results.append({
                        'title': batch['titles'][i],
                        'cipher': cipher_text[:200],
                        'predicted': pred_text[:200],
                        'target': target_text[:200],
                        'char_accuracy': char_acc,
                        'word_accuracy': word_acc,
                        'edit_distance': edit_dist,
                        'length': length.item()
                    })
        
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
                'max_char_accuracy': np.max(all_char_accuracies)
            }
        }
    
    def plot_confusion_matrix(self, confusion, save_path, top_k=10):
        """Plot top-k most confused Greek characters"""
        confusions_list = []
        for target_char, pred_counts in confusion.items():
            for pred_char, count in pred_counts.items():
                if pred_char != target_char and pred_char in self.greek_alphabet:
                    confusions_list.append((target_char, pred_char, count))
        
        confusions_list.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confusions_list[:top_k]
        
        if not top_confusions:
            print("  No confusions found (perfect accuracy!)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = [f"{t}→{p}" for t, p, _ in top_confusions]
        counts = [c for _, _, c in top_confusions]
        
        bars = ax.barh(labels, counts, color='coral')
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_k} Greek Character Confusions (CNN)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {int(count)}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion matrix to {save_path}")


def load_model(model_path, vocab_size, use_deep=False, device='cpu'):
    """Load trained Greek CNN model"""
    if use_deep:
        model = GreekDeepCharCNNDecryptor(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=256,
            num_layers=6,
            kernel_size=3,
            dropout=0.3
        )
    else:
        model = GreekCharCNNDecryptor(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=256,
            kernel_sizes=[3, 5, 7],
            num_conv_layers=3,
            dropout=0.3
        )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_cipher(cipher_type, use_deep=False, data_dir='data/cipher_datasets_greek_1000', device='cpu'):
    """Comprehensive evaluation for one Greek cipher"""
    
    model_name = "cnn_deep_greek" if use_deep else "cnn_greek"
    model_display = "Greek Deep Residual CNN" if use_deep else "Greek Multi-Kernel CNN"
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_display.upper()}: {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    # Load training metrics
    metrics_path = Path('results') / f'{model_name}_{cipher_type}' / 'metrics.json'
    training_info = {}
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            training_info = json.load(f)
    
    print("Loading Greek test data...")
    _, _, test_loader, vocab = create_dataloaders(
        cipher_type=cipher_type,
        data_dir=data_dir,
        batch_size=16,
        max_length=512
    )
    
    model_path = Path('results') / f'{model_name}_{cipher_type}' / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train first: python train_cnn_greek.py --cipher {cipher_type}" + 
              (" --use-deep" if use_deep else ""))
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, vocab.vocab_size, use_deep, device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    evaluator = GreekCNNEvaluator(model, vocab, device)
    results = evaluator.evaluate_dataset(test_loader)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE RESULTS - {model_display.upper()} - {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    # Show training info
    if training_info:
        epochs_trained = training_info.get('epochs_trained', len(training_info.get('train_losses', [])))
        best_val_acc = training_info.get('best_val_accuracy', 0)
        print(f"\n🏋️  Training Info:")
        print(f"  Model Type:          {model_display}")
        print(f"  Language:            Greek (Ελληνικά)")
        print(f"  Alphabet Size:       24 letters")
        print(f"  Vocab Size:          {training_info.get('vocab_size', 37)}")
        print(f"  Epochs Trained:      {epochs_trained}")
        print(f"  Best Val Accuracy:   {best_val_acc*100:>6.2f}%")
        if 'train_fraction' in training_info and training_info['train_fraction'] < 1.0:
            print(f"  Training Data Used:  {training_info['train_fraction']*100:.0f}%")
        if 'learning_rate' in training_info:
            print(f"  Learning Rate:       {training_info['learning_rate']}")
    
    print(f"\n📊 Test Set Metrics:")
    print(f"  Character Accuracy:  {results['summary']['avg_char_accuracy']*100:>6.2f}%")
    print(f"  Word Accuracy:       {results['summary']['avg_word_accuracy']*100:>6.2f}%")
    print(f"  Edit Distance Sim:   {results['summary']['avg_edit_distance']*100:>6.2f}%")
    print(f"\n📈 Statistics:")
    print(f"  Std Dev (char):      {results['summary']['std_char_accuracy']*100:>6.2f}%")
    print(f"  Min Accuracy:        {results['summary']['min_char_accuracy']*100:>6.2f}%")
    print(f"  Max Accuracy:        {results['summary']['max_char_accuracy']*100:>6.2f}%")
    print(f"  Total Samples:       {len(results['results'])}")
    
    # Save results
    results_dir = Path('results') / f'{model_name}_{cipher_type}'
    
    with open(results_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': results['summary'],
            'training_info': training_info,
            'per_article': results['results'],
            'language': 'greek'
        }, f, indent=2, ensure_ascii=False)
    
    with open(results_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            **results['summary'],
            'model_type': model_display,
            'language': 'greek',
            'training_epochs': training_info.get('epochs_trained', 'unknown')
        }, f, indent=2, ensure_ascii=False)
    
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        results_dir / 'confusion_matrix.png'
    )
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(results['char_accuracies'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Character Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Greek Character Accuracy Distribution')
    axes[0].axvline(results['summary']['avg_char_accuracy'], color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    axes[1].hist(results['word_accuracies'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Word Accuracy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Greek Word Accuracy Distribution')
    axes[1].axvline(results['summary']['avg_word_accuracy'], color='red', linestyle='--', label='Mean')
    axes[1].legend()
    
    axes[2].hist(results['edit_distances'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Edit Distance Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Greek Edit Distance Distribution')
    axes[2].axvline(results['summary']['avg_edit_distance'], color='red', linestyle='--', label='Mean')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved detailed results to {results_dir}/")
    
    return results


def evaluate_all_ciphers(use_deep=False, data_dir='data/cipher_datasets_greek_1000'):
    """Evaluate all 6 Greek ciphers"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ciphers = ['caesar', 'atbash', 'affine', 'vigenere',
               'substitution_fixed', 'substitution_random']
    
    all_results = {}
    model_name = "cnn_deep_greek" if use_deep else "cnn_greek"
    model_display = "Greek Deep Residual CNN" if use_deep else "Greek Multi-Kernel CNN"
    
    for cipher in ciphers:
        result = evaluate_cipher(cipher, use_deep, data_dir, device)
        if result:
            all_results[cipher] = result['summary']
    
    if not all_results:
        print("\n❌ No results to compare. Train models first!")
        return
    
    # Comparison table
    print("\n" + "="*100)
    print(f"COMPREHENSIVE COMPARISON - ALL {model_display.upper()}S")
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
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f'all_{model_name}_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cipher_names = [c.replace('_', ' ').title() for c in all_results.keys()]
    char_accs = [s['avg_char_accuracy']*100 for s in all_results.values()]
    word_accs = [s['avg_word_accuracy']*100 for s in all_results.values()]
    
    x = np.arange(len(cipher_names))
    width = 0.35
    
    ax.bar(x - width/2, char_accs, width, label='Character Accuracy', color='steelblue')
    ax.bar(x + width/2, word_accs, width, label='Word Accuracy', color='coral')
    
    ax.set_xlabel('Cipher Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_display} Performance Across All Ciphers', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cipher_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'all_{model_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison to results/all_{model_name}_comparison.png")


def compare_cnn_variants(cipher_type='caesar', data_dir='data/cipher_datasets_greek_1000'):
    """Compare Multi-Kernel vs Deep Residual CNN for Greek"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"COMPARING GREEK CNN VARIANTS: {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    results_multi = evaluate_cipher(cipher_type, use_deep=False, data_dir=data_dir, device=device)
    results_deep = evaluate_cipher(cipher_type, use_deep=True, data_dir=data_dir, device=device)
    
    if not results_multi or not results_deep:
        print("\n❌ Could not compare - one or both models not trained")
        return
    
    # Comparison table
    print("\n" + "="*80)
    print(f"SIDE-BY-SIDE COMPARISON: {cipher_type.upper().replace('_', ' ')}")
    print("="*80)
    print(f"\n{'Metric':<30} {'Multi-Kernel CNN':<20} {'Deep Residual CNN':<20} {'Difference':<15}")
    print("-"*80)
    
    metrics = [
        ('Character Accuracy', 'avg_char_accuracy'),
        ('Word Accuracy', 'avg_word_accuracy'),
        ('Edit Distance Sim', 'avg_edit_distance'),
        ('Std Dev (char)', 'std_char_accuracy')
    ]
    
    for metric_name, metric_key in metrics:
        multi_val = results_multi['summary'][metric_key] * 100
        deep_val = results_deep['summary'][metric_key] * 100
        diff = deep_val - multi_val
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        
        print(f"{metric_name:<30} {multi_val:>8.2f}%       {deep_val:>8.2f}%       {diff_str:>12}")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Greek CNN evaluation')
    parser.add_argument('--cipher', type=str, default='all',
                       choices=['caesar', 'atbash', 'affine', 'vigenere',
                               'substitution_fixed', 'substitution_random', 'all'],
                       help='Which cipher(s) to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/cipher_datasets_greek_1000',
                       help='Data directory')
    parser.add_argument('--use-deep', action='store_true',
                       help='Evaluate deep residual CNN instead of multi-kernel CNN')
    parser.add_argument('--compare-variants', action='store_true',
                       help='Compare multi-kernel vs deep CNN (requires --cipher to be specific)')
    
    args = parser.parse_args()
    
    # Install Levenshtein if needed
    try:
        import Levenshtein
    except ImportError:
        print("Installing python-Levenshtein...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'python-Levenshtein', '--break-system-packages'])
        import Levenshtein
    
    if args.compare_variants:
        if args.cipher == 'all':
            print("❌ --compare-variants requires a specific cipher, not 'all'")
            print("   Example: python evaluate_cnn_greek.py --cipher caesar --compare-variants")
        else:
            compare_cnn_variants(args.cipher, args.data_dir)
    elif args.cipher == 'all':
        evaluate_all_ciphers(args.use_deep, args.data_dir)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate_cipher(args.cipher, args.use_deep, args.data_dir, device)