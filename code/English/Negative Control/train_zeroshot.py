"""
Zero-Shot Learning Training Script
Trains models on 5 ciphers, tests on 6th unseen cipher
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.mlp_cipher_model import MLPDecryptor, ContextualMLPDecryptor
from models.cnn_cipher_model import CharCNNDecryptor, DeepCharCNNDecryptor
from models.transformer_cipher_model import SimpleTransformerDecryptor
from models.lstm_cipher_model import CharLSTMDecryptor  # Fixed: was LSTMDecryptor
from models.multi_cipher_dataloader import create_zeroshot_dataloaders, get_all_zeroshot_configs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ZeroShotTrainer:
    """Trainer for zero-shot cipher learning"""
    
    def __init__(self, model, vocab, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
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
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            cipher = batch['cipher'].to(self.device)
            plain = batch['plain'].to(self.device)
            lengths = batch['lengths']
            
            self.optimizer.zero_grad()
            outputs = self.model(cipher)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = plain.view(-1)
            
            loss = self.criterion(outputs_flat, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            accuracy = self.calculate_accuracy(outputs, plain, lengths)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch in val_loader:
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                lengths = batch['lengths']
                
                outputs = self.model(cipher)
                
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = plain.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                accuracy = self.calculate_accuracy(outputs, plain, lengths)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader, val_loader, num_epochs=20, save_dir='results',
              patience=5, min_delta=0.001):
        """Full training loop with early stopping"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Starting Zero-Shot Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {num_epochs}")
        print(f"Early Stopping: patience={patience}, min_delta={min_delta}")
        print("="*60 + "\n")
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Check for improvement
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_dir / 'best_model.pt')
                print(f"✓ Saved best model (acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Perfect accuracy check
            if val_acc >= 0.9999:
                print(f"\n🎯 Perfect accuracy achieved")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Epochs trained: {epoch+1}/{num_epochs}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return best_val_acc
    
    def plot_training_history(self, save_path, max_epochs=20):
        """Plot training curves"""
        actual_epochs = len(self.train_losses)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, actual_epochs + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', marker='o', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', marker='s', label='Val Loss', linewidth=2)
        
        if actual_epochs < max_epochs:
            ax1.axvline(x=actual_epochs, color='green', linestyle='--', linewidth=2, 
                       label=f'Early Stop (epoch {actual_epochs})')
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.val_accuracies, 'g-', marker='D', label='Val Accuracy', linewidth=2)
        
        best_idx = self.val_accuracies.index(max(self.val_accuracies))
        ax2.plot(best_idx + 1, self.val_accuracies[best_idx], 'r*', markersize=15,
                label=f'Best: {self.val_accuracies[best_idx]:.4f}')
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
        plt.close()


def create_model(model_type, vocab_size, device):
    """Create model based on type"""
    if model_type == 'mlp':
        return MLPDecryptor(vocab_size=vocab_size, embedding_dim=128, 
                           hidden_dims=[512, 256, 128], dropout=0.3)
    elif model_type == 'mlp_context':
        return ContextualMLPDecryptor(vocab_size=vocab_size, embedding_dim=128,
                                     context_size=5, hidden_dims=[512, 256, 128], dropout=0.3)
    elif model_type == 'cnn':
        return CharCNNDecryptor(vocab_size=vocab_size, embedding_dim=128,
                               num_filters=256, kernel_sizes=[3, 5, 7], 
                               num_conv_layers=3, dropout=0.3)
    elif model_type == 'cnn_deep':
        return DeepCharCNNDecryptor(vocab_size=vocab_size, embedding_dim=128,
                                   num_filters=256, num_layers=6, kernel_size=3, dropout=0.3)
    elif model_type == 'transformer':
        return SimpleTransformerDecryptor(vocab_size=vocab_size, d_model=256, nhead=8,
                                         num_layers=4, dim_feedforward=1024, dropout=0.1)
    elif model_type == 'lstm':
        return CharLSTMDecryptor(vocab_size=vocab_size, embedding_dim=128,
                           hidden_dim=256, num_layers=2, dropout=0.3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_single_zeroshot(
    train_ciphers,
    test_cipher,
    model_type='transformer',
    num_epochs=20,
    data_dir='data/cipher_datasets_1000',
    learning_rate=0.001,
    patience=5,
    min_delta=0.001
):
    """Train a single zero-shot configuration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, vocab = create_zeroshot_dataloaders(
        train_ciphers=train_ciphers,
        test_cipher=test_cipher,
        data_dir=data_dir,
        batch_size=16
    )
    
    # Create model
    print(f"\nInitializing {model_type.upper()} model...")
    model = create_model(model_type, vocab.vocab_size, device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = ZeroShotTrainer(model, vocab, device=device, learning_rate=learning_rate)
    
    # Setup results directory
    results_dir = Path('results') / 'zeroshot' / model_type / f'test_{test_cipher}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    best_acc = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_dir=results_dir,
        patience=patience,
        min_delta=min_delta
    )
    
    # Plot training history
    trainer.plot_training_history(
        results_dir / 'training_history.png',
        max_epochs=num_epochs
    )
    
    # Save metrics
    metrics = {
        'model_type': model_type,
        'train_ciphers': train_ciphers,
        'test_cipher': test_cipher,
        'best_val_accuracy': best_acc,
        'epochs_trained': len(trainer.train_losses),
        'learning_rate': learning_rate,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    return trainer, test_loader


def train_all_zeroshot_configs(model_type='transformer', num_epochs=20, **kwargs):
    """Train all 6 zero-shot configurations for a model"""
    
    configs = get_all_zeroshot_configs()
    results_summary = []
    
    print("\n" + "="*80)
    print(f"TRAINING ALL ZERO-SHOT CONFIGURATIONS FOR {model_type.upper()}")
    print("="*80)
    print(f"Total configurations: {len(configs)}")
    print("="*80 + "\n")
    
    for i, (train_ciphers, test_cipher) in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {i}/{len(configs)}")
        print(f"Test Cipher: {test_cipher.upper()}")
        print(f"Train Ciphers: {', '.join([c.upper() for c in train_ciphers])}")
        print(f"{'#'*80}\n")
        
        trainer, test_loader = train_single_zeroshot(
            train_ciphers=train_ciphers,
            test_cipher=test_cipher,
            model_type=model_type,
            num_epochs=num_epochs,
            **kwargs
        )
        
        results_summary.append({
            'test_cipher': test_cipher,
            'train_ciphers': train_ciphers,
            'best_val_accuracy': max(trainer.val_accuracies)
        })
    
    # Save overall summary
    summary_dir = Path('results') / 'zeroshot' / model_type
    with open(summary_dir / 'all_configs_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print(f"ZERO-SHOT TRAINING SUMMARY - {model_type.upper()}")
    print("="*80)
    print(f"\n{'Test Cipher':<25} {'Val Accuracy':<15}")
    print("-"*40)
    
    for result in results_summary:
        print(f"{result['test_cipher']:<25} {result['best_val_accuracy']*100:>10.2f}%")
    
    print("="*80)
    print(f"\n✓ All configurations completed!")
    print(f"✓ Summary saved to {summary_dir}/all_configs_summary.json\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Zero-Shot Learning for Cipher Decryption')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['mlp', 'mlp_context', 'cnn', 'cnn_deep', 'transformer', 'lstm'],
                       help='Model type to train')
    parser.add_argument('--test-cipher', type=str, default=None,
                       help='Specific cipher to test (if None, trains all 6 configs)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Maximum number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data/cipher_datasets_1000',
                       help='Data directory')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement for early stopping')
    
    args = parser.parse_args()
    
    if args.test_cipher:
        # Train single configuration
        all_ciphers = ['caesar', 'atbash', 'affine', 'vigenere',
                      'substitution_fixed', 'substitution_random']
        train_ciphers = [c for c in all_ciphers if c != args.test_cipher]
        
        print(f"\n{'='*60}")
        print(f"Training {args.model.upper()} - Zero-Shot on {args.test_cipher.upper()}")
        print(f"{'='*60}\n")
        
        train_single_zeroshot(
            train_ciphers=train_ciphers,
            test_cipher=args.test_cipher,
            model_type=args.model,
            num_epochs=args.epochs,
            data_dir=args.data_dir,
            learning_rate=args.learning_rate,
            patience=args.patience,
            min_delta=args.min_delta
        )
    else:
        # Train all 6 configurations
        train_all_zeroshot_configs(
            model_type=args.model,
            num_epochs=args.epochs,
            data_dir=args.data_dir,
            learning_rate=args.learning_rate,
            patience=args.patience,
            min_delta=args.min_delta
        )
    
    print("\n✅ Zero-shot training complete!")