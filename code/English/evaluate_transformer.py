"""
Quick Transformer Evaluation Script
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.transformer_cipher_model import SimpleTransformerDecryptor
from models.data_loader import create_dataloaders


def evaluate_transformer(cipher_type='caesar'):
    """Evaluate transformer model on test set"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"EVALUATING TRANSFORMER - {cipher_type.upper()} CIPHER")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading test data...")
    _, _, test_loader, vocab = create_dataloaders(
        cipher_type=cipher_type,
        batch_size=16,
        max_length=512
    )
    
    # Load model
    model_path = Path('results') / f'transformer_{cipher_type}' / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"Loading model from {model_path}...")
    
    model = SimpleTransformerDecryptor(
        vocab_size=vocab.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate
    print("Evaluating on test set...")
    total_correct = 0
    total_chars = 0
    all_accuracies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            cipher = batch['cipher'].to(device)
            plain = batch['plain'].to(device)
            lengths = batch['lengths']
            
            # Forward pass
            outputs = model(cipher)
            predictions = outputs.argmax(dim=-1)
            
            # Calculate accuracy per sample
            for i, length in enumerate(lengths):
                pred = predictions[i, :length]
                targ = plain[i, :length]
                
                correct = (pred == targ).sum().item()
                total_correct += correct
                total_chars += length.item()
                
                acc = correct / length.item()
                all_accuracies.append(acc)
    
    avg_accuracy = total_correct / total_chars
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS - TRANSFORMER - {cipher_type.upper()}")
    print(f"{'='*80}")
    print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
    print(f"Min Accuracy:     {min(all_accuracies)*100:.2f}%")
    print(f"Max Accuracy:     {max(all_accuracies)*100:.2f}%")
    print(f"Std Dev:          {np.std(all_accuracies)*100:.2f}%")
    print(f"Total characters: {total_chars:,}")
    print(f"{'='*80}\n")
    
    # Save results
    results_dir = Path('results') / f'transformer_{cipher_type}'
    results = {
        'model_type': 'transformer',
        'cipher_type': cipher_type,
        'test_accuracy': avg_accuracy,
        'min_accuracy': min(all_accuracies),
        'max_accuracy': max(all_accuracies),
        'std_accuracy': float(np.std(all_accuracies)),
        'num_samples': len(all_accuracies),
        'total_characters': total_chars
    }
    
    with open(results_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_dir / 'test_results.json'}\n")
    
    return avg_accuracy


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("TRANSFORMER MODEL EVALUATION")
    print("="*80)
    
    # Evaluate both ciphers
    caesar_acc = evaluate_transformer('caesar')
    sub_acc = evaluate_transformer('substitution')
    
    # Summary
    print("\n" + "="*80)
    print("TRANSFORMER SUMMARY")
    print("="*80)
    print(f"Caesar Cipher:       {caesar_acc*100:.2f}%")
    print(f"Substitution Cipher: {sub_acc*100:.2f}%")
    print("="*80 + "\n")