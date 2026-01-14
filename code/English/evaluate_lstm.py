"""
Model Evaluation Script
Evaluate trained LSTM models on test set
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.lstm_cipher_model import SimpleLSTMDecryptor
from models.data_loader import create_dataloaders


class ModelEvaluator:
    """Evaluate cipher decryption models"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
    
    def calculate_accuracy(self, outputs, targets, lengths):
        """Calculate character-level accuracy"""
        predictions = outputs.argmax(dim=-1)
        
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i, :length]
            targ = targets[i, :length]
            correct += (pred == targ).sum().item()
            total += length.item()
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_dataset(self, dataloader):
        """Evaluate on entire dataset"""
        total_accuracy = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        all_accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                lengths = batch['lengths']
                
                # Forward pass
                outputs = self.model(cipher)
                predictions = outputs.argmax(dim=-1)
                
                # Calculate accuracy per sample
                for i, length in enumerate(lengths):
                    pred = predictions[i, :length]
                    targ = plain[i, :length]
                    
                    acc = (pred == targ).sum().item() / length.item()
                    all_accuracies.append(acc)
                    
                    # Store sample predictions
                    pred_text = self.vocab.decode(pred.cpu().tolist())
                    targ_text = self.vocab.decode(targ.cpu().tolist())
                    cipher_text = self.vocab.decode(cipher[i, :length].cpu().tolist())
                    
                    all_predictions.append({
                        'title': batch['titles'][i],
                        'cipher': cipher_text,
                        'predicted': pred_text,
                        'target': targ_text,
                        'accuracy': acc
                    })
                
                batch_acc = self.calculate_accuracy(outputs, plain, lengths)
                total_accuracy += batch_acc * len(lengths)
                total_samples += len(lengths)
        
        avg_accuracy = total_accuracy / total_samples
        
        return {
            'average_accuracy': avg_accuracy,
            'per_sample_accuracies': all_accuracies,
            'predictions': all_predictions
        }
    
    def display_sample_predictions(self, results, num_samples=3):
        """Display sample predictions"""
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS")
        print("="*80)
        
        for i, pred in enumerate(results['predictions'][:num_samples]):
            print(f"\n[Sample {i+1}] {pred['title'][:60]}")
            print(f"Accuracy: {pred['accuracy']*100:.2f}%")
            print(f"\nEncrypted: {pred['cipher'][:100]}...")
            print(f"Predicted: {pred['predicted'][:100]}...")
            print(f"Target:    {pred['target'][:100]}...")
            print("-"*80)


def load_trained_model(model_path, vocab_size, device='cpu'):
    """Load a trained model from checkpoint"""
    model = SimpleLSTMDecryptor(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_cipher(cipher_type='caesar', device='cpu'):
    """Evaluate a trained model on test set"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {cipher_type.upper()} CIPHER MODEL")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading test data...")
    _, _, test_loader, vocab = create_dataloaders(
        cipher_type=cipher_type,
        batch_size=16,
        max_length=512
    )
    
    # Load model
    model_path = Path('results') / f'lstm_{cipher_type}' / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train the model first: python train_lstm.py --cipher {cipher_type}")
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, vocab.vocab_size, device)
    
    # Evaluate
    evaluator = ModelEvaluator(model, vocab, device)
    results = evaluator.evaluate_dataset(test_loader)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS - {cipher_type.upper()}")
    print(f"{'='*80}")
    print(f"Average Accuracy: {results['average_accuracy']*100:.2f}%")
    print(f"Min Accuracy:     {min(results['per_sample_accuracies'])*100:.2f}%")
    print(f"Max Accuracy:     {max(results['per_sample_accuracies'])*100:.2f}%")
    print(f"Std Dev:          {np.std(results['per_sample_accuracies'])*100:.2f}%")
    
    # Display samples
    evaluator.display_sample_predictions(results, num_samples=3)
    
    # Save results
    results_dir = Path('results') / f'lstm_{cipher_type}'
    output_file = results_dir / 'test_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'average_accuracy': results['average_accuracy'],
            'min_accuracy': min(results['per_sample_accuracies']),
            'max_accuracy': max(results['per_sample_accuracies']),
            'std_accuracy': float(np.std(results['per_sample_accuracies'])),
            'num_samples': len(results['per_sample_accuracies'])
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results


def compare_both_ciphers():
    """Compare Caesar and Substitution cipher results"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate both
    caesar_results = evaluate_cipher('caesar', device)
    substitution_results = evaluate_cipher('substitution', device)
    
    if caesar_results is None or substitution_results is None:
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Caesar distribution
    axes[0].hist(caesar_results['per_sample_accuracies'], bins=20, 
                 color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(caesar_results['average_accuracy'], color='red', 
                    linestyle='--', linewidth=2, label='Mean')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Caesar Cipher\nAvg: {caesar_results["average_accuracy"]*100:.2f}%')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Substitution distribution
    axes[1].hist(substitution_results['per_sample_accuracies'], bins=20,
                 color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(substitution_results['average_accuracy'], color='red',
                    linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Substitution Cipher\nAvg: {substitution_results["average_accuracy"]*100:.2f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_histogram.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to results/comparison_histogram.png")
    plt.close()
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<25} {'Caesar':<20} {'Substitution':<20}")
    print("-"*80)
    print(f"{'Average Accuracy':<25} {caesar_results['average_accuracy']*100:>18.2f}% {substitution_results['average_accuracy']*100:>18.2f}%")
    print(f"{'Min Accuracy':<25} {min(caesar_results['per_sample_accuracies'])*100:>18.2f}% {min(substitution_results['per_sample_accuracies'])*100:>18.2f}%")
    print(f"{'Max Accuracy':<25} {max(caesar_results['per_sample_accuracies'])*100:>18.2f}% {max(substitution_results['per_sample_accuracies'])*100:>18.2f}%")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained cipher models')
    parser.add_argument('--cipher', type=str, default='both',
                       choices=['caesar', 'substitution', 'both'],
                       help='Which cipher to evaluate')
    
    args = parser.parse_args()
    
    if args.cipher == 'both':
        compare_both_ciphers()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate_cipher(args.cipher, device)