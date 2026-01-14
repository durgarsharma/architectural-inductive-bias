"""
Training Script for Negative Control Experiments (AES/DES)
Uses dedicated data loader for cryptographically secure ciphers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import models
import sys
sys.path.append('models')
from mlp_cipher_model import MLPCipherModel
from lstm_cipher_model import LSTMCipherModel
from cnn_cipher_model import CNNCipherModel
from transformer_cipher_model import TransformerCipherModel
from data_loader_negative_control import get_data_loaders  # Use negative control loader


class CipherTrainer:
    """Trainer for cipher decryption models"""
    
    def __init__(self, model, device, save_dir):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def calculate_accuracy(self, predictions, targets, lengths):
        """Calculate character-level accuracy"""
        correct = 0
        total = 0
        
        for pred, target, length in zip(predictions, targets, lengths):
            pred = pred[:length]
            target = target[:length]
            correct += (pred == target).sum().item()
            total += length.item()
        
        return correct / total if total > 0 else 0
    
    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            cipher = batch['cipher'].to(self.device)
            plain = batch['plain'].to(self.device)
            lengths = batch['lengths']
            
            optimizer.zero_grad()
            logits = self.model(cipher)
            
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                plain.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            
            predictions = torch.argmax(logits, dim=-1)
            accuracy = self.calculate_accuracy(predictions, plain, lengths)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                lengths = batch['lengths']
                
                logits = self.model(cipher)
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    plain.view(-1)
                )
                
                predictions = torch.argmax(logits, dim=-1)
                accuracy = self.calculate_accuracy(predictions, plain, lengths)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate, patience=5):
        """Full training loop"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Check for negative control expectation
            if val_acc < 0.10:
                print(f"  ✓ Validation accuracy < 10% (expected for negative control)")
            else:
                print(f"  ⚠️  WARNING: Val accuracy > 10% (unexpected for secure cipher!)")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', epoch, val_loss, val_acc)
                print(f"  ✓ New best model saved!")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        self.save_checkpoint('final_model.pt', epoch, val_loss, val_acc)
        self.plot_training_curves()
        
        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation accuracy: {val_acc:.4f}")
        if val_acc < 0.10:
            print("✅ PASS: Model performs near-random (as expected for secure cipher)")
        else:
            print("⚠️  WARNING: Model performs better than random (investigate!)")
        print("=" * 70)
    
    def save_checkpoint(self, filename, epoch, val_loss, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        ax2.axhline(y=0.037, color='red', linestyle='--', label='Random Baseline')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy (Negative Control)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train on negative control (AES/DES)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp', 'lstm', 'cnn', 'transformer'])
    parser.add_argument('--cipher', type=str, required=True,
                       choices=['aes', 'des', 'AES', 'DES'])
    parser.add_argument('--cipher-dir', type=str, required=True,
                       help='Path to cipher directory (e.g., data/.../English/AES)')
    parser.add_argument('--save-dir', type=str, required=True,
                       help='Save directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data using negative control loader
    print(f"\nLoading {args.cipher.upper()} cipher data from {args.cipher_dir}...")
    
    from data_loader_negative_control import get_negative_control_loaders
    train_loader, val_loader, test_loader = get_negative_control_loaders(
        cipher_dir=args.cipher_dir,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    
    if args.model == 'mlp':
        model = MLPCipherModel(vocab_size=27, embedding_dim=128, hidden_dims=[512, 256, 128])
    elif args.model == 'lstm':
        model = LSTMCipherModel(vocab_size=27, embedding_dim=128, hidden_dim=256, 
                               num_layers=2, bidirectional=True)
    elif args.model == 'cnn':
        model = CNNCipherModel(vocab_size=27, embedding_dim=128, num_filters=256,
                              kernel_sizes=[3, 5, 7])
    elif args.model == 'transformer':
        model = TransformerCipherModel(vocab_size=27, d_model=256, nhead=8, num_layers=4)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*70)
    print("⚠️  NEGATIVE CONTROL EXPERIMENT")
    print("="*70)
    print(f"Cipher: {args.cipher.upper()} (cryptographically secure)")
    print(f"Expected accuracy: ~3-5% (random baseline)")
    print(f"If accuracy > 10% → Model is overfitting or data issue")
    print("="*70 + "\n")
    
    # Train
    trainer = CipherTrainer(model, device, args.save_dir)
    trainer.train(train_loader, val_loader, args.epochs, args.lr, args.patience)
    
    # Save configuration
    config = vars(args)
    config['total_parameters'] = total_params
    config['device'] = str(device)
    config['experiment_type'] = 'negative_control'
    config['expected_accuracy'] = '3-5%'
    
    with open(Path(args.save_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()