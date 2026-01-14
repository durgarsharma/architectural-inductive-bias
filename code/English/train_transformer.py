"""
Training Script for Transformer Cipher Decryption
Step 2: Train transformer models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.transformer_cipher_model import SimpleTransformerDecryptor, count_parameters
from models.data_loader import create_dataloaders


class TransformerTrainer:
    """Trainer for Transformer cipher decryption"""
    
    def __init__(self, model, vocab, device='cpu', learning_rate=0.0001):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Tracking
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
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(cipher)
            
            # Calculate loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = plain.view(-1)
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
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
                
                # Forward pass
                outputs = self.model(cipher)
                
                # Calculate loss
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = plain.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                # Calculate accuracy
                accuracy = self.calculate_accuracy(outputs, plain, lengths)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader, val_loader, num_epochs=20, save_dir='results'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Starting Transformer Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Epochs: {num_epochs}")
        print("="*60 + "\n")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_dir / 'best_model.pt')
                print(f"✓ Saved best model (acc: {val_acc:.4f})")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return best_val_acc
    
    def plot_training_history(self, save_path='results/training_history.png'):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
        plt.close()


def main(cipher_type='caesar', num_epochs=20):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        cipher_type=cipher_type,
        batch_size=16,
        max_length=512
    )
    
    # Create model
    print("\nInitializing Transformer model...")
    model = SimpleTransformerDecryptor(
        vocab_size=vocab.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = TransformerTrainer(model, vocab, device=device, learning_rate=0.0001)
    
    # Train
    results_dir = Path('results') / f'transformer_{cipher_type}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_dir=results_dir
    )
    
    # Plot training history
    trainer.plot_training_history(results_dir / 'training_history.png')
    
    # Save training metrics
    metrics = {
        'model_type': 'transformer',
        'cipher_type': cipher_type,
        'best_val_accuracy': best_acc,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accuracies': trainer.val_accuracies
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    return trainer, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transformer for cipher decryption')
    parser.add_argument('--cipher', type=str, default='caesar',
                       choices=['caesar', 'substitution'],
                       help='Cipher type to train on')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Training Transformer on {args.cipher.upper()} cipher")
    print(f"{'='*60}\n")
    
    trainer, test_loader = main(cipher_type=args.cipher, num_epochs=args.epochs)
    
    print("\n✅ Transformer training complete!")