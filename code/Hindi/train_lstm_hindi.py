"""
Training Script for LSTM - All 6 Ciphers (Hindi/Devanagari)
Updated to handle: caesar, atbash, affine, vigenere, substitution_fixed, substitution_random
Includes early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.lstm_cipher_model import SimpleLSTMDecryptor, count_parameters
from models.data_loader_hindi import create_dataloaders_hindi


class LSTMTrainerHindi:
    """Trainer class for LSTM cipher decryption (Hindi)"""
    
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
        print("Starting Training (Hindi/Devanagari)")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {num_epochs}")
        print(f"Early Stopping: patience={patience}, min_delta={min_delta}")
        print("="*60 + "\n")
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
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
            
            # Check for improvement (using accuracy as primary metric)
            improved = False
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_val_loss = val_loss
                improved = True
            elif abs(val_acc - best_val_acc) < min_delta and val_loss < best_val_loss - min_delta:
                # If accuracy is similar but loss improved
                best_val_loss = val_loss
                improved = True
            
            if improved:
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pt')
                print(f"✓ Saved best model (acc: {val_acc:.4f}, loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {patience} consecutive epochs")
                print(f"   Best validation accuracy: {best_val_acc:.4f}")
                break
            
            # Perfect accuracy check
            if val_acc >= 0.9999:
                print(f"\n🎯 Perfect accuracy achieved (99.99%+)")
                print(f"   Stopping training early")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Epochs trained: {epoch+1}/{num_epochs}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return best_val_acc
    
    def plot_training_history(self, save_path='results/training_history.png'):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot to {save_path}")
        plt.close()


def main(cipher_type='caesar', num_epochs=20, data_dir='data/cipher_datasets_hindi_1000',
         train_fraction=1.0, learning_rate=0.001, hidden_dim=256,
         patience=5, min_delta=0.001):
    """Main training function with controllable difficulty and early stopping"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading Hindi data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders_hindi(
        cipher_type=cipher_type,
        data_dir=data_dir,
        batch_size=16,
        max_length=512
    )
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Sample characters: {list(vocab.char2idx.keys())[:20]}")
    
    # Optionally reduce training data for more gradual learning
    if train_fraction < 1.0:
        original_size = len(train_loader.dataset)
        subset_size = int(original_size * train_fraction)
        indices = torch.randperm(original_size)[:subset_size]
        
        from torch.utils.data import Subset, DataLoader
        from models.data_loader_hindi import collate_batch_hindi
        
        subset_dataset = Subset(train_loader.dataset, indices)
        train_loader = DataLoader(
            subset_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_batch_hindi,
            num_workers=0
        )
        print(f"📉 Using {subset_size}/{original_size} training samples ({train_fraction*100:.0f}%)")
    
    print("\nInitializing model...")
    model = SimpleLSTMDecryptor(
        vocab_size=vocab.vocab_size,
        embedding_dim=128,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    trainer = LSTMTrainerHindi(model, vocab, device=device, learning_rate=learning_rate)
    
    results_dir = Path('results') / f'lstm_hindi_{cipher_type}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_dir=results_dir,
        patience=patience,
        min_delta=min_delta
    )
    
    trainer.plot_training_history(results_dir / 'training_history.png')
    
    metrics = {
        'cipher_type': cipher_type,
        'language': 'hindi',
        'script': 'devanagari',
        'best_val_accuracy': best_acc,
        'train_fraction': train_fraction,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'epochs_trained': len(trainer.train_losses),
        'early_stopping_patience': patience,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies
    }
    
    with open(results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    return trainer, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM for Hindi cipher decryption')
    parser.add_argument('--cipher', type=str, default='caesar', 
                       choices=['caesar', 'atbash', 'affine', 'vigenere', 
                               'substitution_fixed', 'substitution_random'],
                       help='Cipher type to train on')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Maximum number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data/cipher_datasets_hindi_1000',
                       help='Data directory')
    parser.add_argument('--train-fraction', type=float, default=1.0,
                       help='Fraction of training data to use (0.0-1.0)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement to reset patience')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training LSTM on Hindi {args.cipher.upper().replace('_', ' ')} cipher")
    if args.train_fraction < 1.0:
        print(f"Using {args.train_fraction*100:.0f}% of training data")
    print(f"{'='*60}\n")
    
    trainer, test_loader = main(
        cipher_type=args.cipher, 
        num_epochs=args.epochs,
        data_dir=args.data_dir,
        train_fraction=args.train_fraction,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    print("\n✅ Training complete! Next: Evaluate on test set")