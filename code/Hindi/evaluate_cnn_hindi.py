"""
Comprehensive CNN Evaluation - All 6 Ciphers (Hindi/Devanagari)
Includes: Character accuracy, Word accuracy, Edit distance, Per-article breakdown, 
Full Confusion matrix with heatmaps
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import Levenshtein

from models.cnn_cipher_model import CharCNNDecryptor, DeepCharCNNDecryptor, count_parameters
from models.data_loader_hindi import create_dataloaders_hindi

import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib to use Devanagari fonts BEFORE any plotting
matplotlib.rcParams['font.sans-serif'] = ['Devanagari Sangam MN', 'Arial Unicode MS', 'Noto Sans Devanagari', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display




def save_character_mapping(alphabet_list, save_path):
    """
    Save character-to-index mapping as text and JSON files
    """
    mapping_dir = Path(save_path).parent
    
    # Save as readable text file
    txt_path = mapping_dir / 'character_index_mapping.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DEVANAGARI CHARACTER TO INDEX MAPPING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Characters: {len(alphabet_list)}\n\n")
        
        # Organize by character type
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        vowels = 'अआइईउऊऋएऐओऔ'
        
        f.write("CONSONANTS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in consonants:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nVOWELS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in vowels:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nALL CHARACTERS (Sequential):\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            f.write(f"Index {i:2d}: {char}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    # Save as JSON
    json_path = mapping_dir / 'character_index_mapping.json'
    mapping_dict = {
        'total_characters': len(alphabet_list),
        'index_to_char': {i: char for i, char in enumerate(alphabet_list)},
        'char_to_index': {char: i for i, char in enumerate(alphabet_list)},
        'consonants': {i: char for i, char in enumerate(alphabet_list) if char in consonants},
        'vowels': {i: char for i, char in enumerate(alphabet_list) if char in vowels}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved character mapping:")
    print(f"    - {txt_path.name} (human-readable)")
    print(f"    - {json_path.name} (machine-readable)")
    
    return txt_path, json_path




def save_character_mapping(alphabet_list, save_path):
    """
    Save character-to-index mapping as text and JSON files
    """
    mapping_dir = Path(save_path).parent
    
    # Save as readable text file
    txt_path = mapping_dir / 'character_index_mapping.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DEVANAGARI CHARACTER TO INDEX MAPPING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Characters: {len(alphabet_list)}\n\n")
        
        # Organize by character type
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        vowels = 'अआइईउऊऋएऐओऔ'
        
        f.write("CONSONANTS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in consonants:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nVOWELS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in vowels:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nALL CHARACTERS (Sequential):\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            f.write(f"Index {i:2d}: {char}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    # Save as JSON
    json_path = mapping_dir / 'character_index_mapping.json'
    mapping_dict = {
        'total_characters': len(alphabet_list),
        'index_to_char': {i: char for i, char in enumerate(alphabet_list)},
        'char_to_index': {char: i for i, char in enumerate(alphabet_list)},
        'consonants': {i: char for i, char in enumerate(alphabet_list) if char in consonants},
        'vowels': {i: char for i, char in enumerate(alphabet_list) if char in vowels}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved character mapping:")
    print(f"    - {txt_path.name} (human-readable)")
    print(f"    - {json_path.name} (machine-readable)")
    
    return txt_path, json_path




def save_character_mapping(alphabet_list, save_path):
    """
    Save character-to-index mapping as text and JSON files
    """
    mapping_dir = Path(save_path).parent
    
    # Save as readable text file
    txt_path = mapping_dir / 'character_index_mapping.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DEVANAGARI CHARACTER TO INDEX MAPPING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Characters: {len(alphabet_list)}\n\n")
        
        # Organize by character type
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        vowels = 'अआइईउऊऋएऐओऔ'
        
        f.write("CONSONANTS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in consonants:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nVOWELS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in vowels:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nALL CHARACTERS (Sequential):\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            f.write(f"Index {i:2d}: {char}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    # Save as JSON
    json_path = mapping_dir / 'character_index_mapping.json'
    mapping_dict = {
        'total_characters': len(alphabet_list),
        'index_to_char': {i: char for i, char in enumerate(alphabet_list)},
        'char_to_index': {char: i for i, char in enumerate(alphabet_list)},
        'consonants': {i: char for i, char in enumerate(alphabet_list) if char in consonants},
        'vowels': {i: char for i, char in enumerate(alphabet_list) if char in vowels}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved character mapping:")
    print(f"    - {txt_path.name} (human-readable)")
    print(f"    - {json_path.name} (machine-readable)")
    
    return txt_path, json_path




def save_character_mapping(alphabet_list, save_path):
    """
    Save character-to-index mapping as text and JSON files
    """
    mapping_dir = Path(save_path).parent
    
    # Save as readable text file
    txt_path = mapping_dir / 'character_index_mapping.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DEVANAGARI CHARACTER TO INDEX MAPPING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Characters: {len(alphabet_list)}\n\n")
        
        # Organize by character type
        consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        vowels = 'अआइईउऊऋएऐओऔ'
        
        f.write("CONSONANTS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in consonants:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nVOWELS:\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            if char in vowels:
                f.write(f"Index {i:2d}: {char}  (Unicode: U+{ord(char):04X})\n")
        
        f.write("\n\nALL CHARACTERS (Sequential):\n")
        f.write("-" * 60 + "\n")
        for i, char in enumerate(alphabet_list):
            f.write(f"Index {i:2d}: {char}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    # Save as JSON
    json_path = mapping_dir / 'character_index_mapping.json'
    mapping_dict = {
        'total_characters': len(alphabet_list),
        'index_to_char': {i: char for i, char in enumerate(alphabet_list)},
        'char_to_index': {char: i for i, char in enumerate(alphabet_list)},
        'consonants': {i: char for i, char in enumerate(alphabet_list) if char in consonants},
        'vowels': {i: char for i, char in enumerate(alphabet_list) if char in vowels}
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved character mapping:")
    print(f"    - {txt_path.name} (human-readable)")
    print(f"    - {json_path.name} (machine-readable)")
    
    return txt_path, json_path


class CNNEvaluatorHindi:
    """Comprehensive evaluation for CNN models (Hindi)"""
    
    def __init__(self, model, vocab, device='cp'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
        
        # Hindi alphabet
        self.consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        self.vowels = 'अआइईउऊऋएऐओऔ'
        self.alphabet = self.consonants + self.vowels
        self.alphabet_list = list(self.alphabet)
    
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
        """Build character confusion matrix"""
        confusion = {}
        for pred_char, target_char in zip(pred_text, target_text):
            if target_char in self.alphabet:
                if target_char not in confusion:
                    confusion[target_char] = Counter()
                confusion[target_char][pred_char] += 1
        return confusion
    
    def calculate_loss(self, dataloader):
        """Calculate loss on a dataset"""
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                cipher = batch['cipher'].to(self.device)
                plain = batch['plain'].to(self.device)
                
                outputs = self.model(cipher)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = plain.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate_dataset(self, dataloader):
        """Comprehensive evaluation"""
        all_results = []
        all_char_accuracies = []
        all_word_accuracies = []
        all_edit_distances = []
        global_confusion = {char: Counter() for char in self.alphabet}
        
        # Full confusion matrix for heatmap
        confusion_matrix = np.zeros((len(self.alphabet_list), len(self.alphabet_list)))
        
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
                    
                    # Update full confusion matrix
                    for pred_char, target_char in zip(pred_text, target_text):
                        if target_char in self.alphabet_list and pred_char in self.alphabet_list:
                            target_idx = self.alphabet_list.index(target_char)
                            pred_idx = self.alphabet_list.index(pred_char)
                            confusion_matrix[target_idx, pred_idx] += 1
                    
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
        
        # Normalize confusion matrix
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion_matrix_normalized = confusion_matrix / row_sums
        
        return {
            'results': all_results,
            'char_accuracies': all_char_accuracies,
            'word_accuracies': all_word_accuracies,
            'edit_distances': all_edit_distances,
            'confusion_dict': global_confusion,
            'confusion_matrix': confusion_matrix,
            'confusion_matrix_normalized': confusion_matrix_normalized,
            'summary': {
                'avg_char_accuracy': np.mean(all_char_accuracies),
                'avg_word_accuracy': np.mean(all_word_accuracies),
                'avg_edit_distance': np.mean(all_edit_distances),
                'std_char_accuracy': np.std(all_char_accuracies),
                'min_char_accuracy': np.min(all_char_accuracies),
                'max_char_accuracy': np.max(all_char_accuracies),
                'median_char_accuracy': np.median(all_char_accuracies)
            }
        }
    
    def plot_top_confusions(self, confusion_dict, save_path, top_k=15):
        """Plot top-k most confused characters using indices"""
        confusions_list = []
        for target_char, pred_counts in confusion_dict.items():
            for pred_char, count in pred_counts.items():
                if pred_char != target_char and pred_char in self.alphabet:
                    confusions_list.append((target_char, pred_char, count))
        
        confusions_list.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confusions_list[:top_k]
        
        if not top_confusions:
            print("  No confusions found (perfect accuracy!)")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create labels with indices
        labels = []
        confusion_details = []
        for target_char, pred_char, count in top_confusions:
            target_idx = self.alphabet_list.index(target_char)
            pred_idx = self.alphabet_list.index(pred_char)
            labels.append(f"{target_idx}→{pred_idx}")
            confusion_details.append((target_idx, target_char, pred_idx, pred_char, count))
        
        counts = [c for _, _, c in top_confusions]
        
        bars = ax.barh(labels, counts, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top {top_k} Character Confusions (Index Format: True→Predicted)', 
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {int(count)}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved top confusions to {save_path}")
        
        # Save confusion details with mapping
        mapping_path = Path(save_path).parent / 'top_confusions_mapping.txt'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"TOP {top_k} CHARACTER CONFUSIONS - DETAILED MAPPING")
            f.write("=" * 70 + "\n")
            f.write(f"{'True Idx':<10} {'True Char':<12} {'Pred Idx':<10} ")
            f.write(f"{'Pred Char':<12} {'Count':<10}")
            f.write("-" * 70 + "\n")
            for t_idx, t_char, p_idx, p_char, count in confusion_details:
                f.write(f"{t_idx:<10} {t_char:<12} {p_idx:<10} {p_char:<12} {count:<10}")
            f.write("=" * 70 + "\n")
        
        print(f"  ✓ Saved confusion details to {mapping_path.name}")


    def plot_confusion_matrix_heatmap(self, confusion_matrix, save_path, 
                                      normalize=True, top_n=30):
        """Plot confusion matrix heatmap using character indices"""
        char_frequencies = confusion_matrix.sum(axis=1)
        top_indices = np.argsort(char_frequencies)[-top_n:][::-1]
        
        sub_matrix = confusion_matrix[top_indices][:, top_indices]
        
        if normalize:
            row_sums = sub_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            sub_matrix = sub_matrix / row_sums
        
        # Use indices as labels
        top_char_labels = [f"{i}" for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        im = ax.imshow(sub_matrix, cmap='YlOrRd', aspect='auto', vmin=0, 
                       vmax=1 if normalize else None)
        
        ax.set_xticks(np.arange(len(top_char_labels)))
        ax.set_yticks(np.arange(len(top_char_labels)))
        ax.set_xticklabels(top_char_labels, fontsize=9)
        ax.set_yticklabels(top_char_labels, fontsize=9)
        
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Frequency' if normalize else 'Count', 
                       rotation=270, labelpad=20)
        
        ax.set_xlabel('Predicted Character Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Character Index', fontsize=12, fontweight='bold')
        title = f'Confusion Matrix (Top {top_n} Character Indices)'
        if normalize:
            title += ' - Normalized'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(np.arange(len(top_char_labels))-.5, minor=True)
        ax.set_yticks(np.arange(len(top_char_labels))-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion matrix heatmap to {save_path}")
        
        # Save index mapping for confusion matrix
        mapping_path = Path(save_path).parent / 'confusion_matrix_index_mapping.txt'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CONFUSION MATRIX - INDEX TO CHARACTER MAPPING")
            f.write(f"Top {top_n} Most Frequent Characters")
            f.write("=" * 60 + "\n")
            f.write(f"{'Index':<8} {'Character':<12} {'Frequency':<12}")
            f.write("-" * 60 + "\n")
            for idx in top_indices:
                char = self.alphabet_list[idx]
                freq = int(char_frequencies[idx])
                f.write(f"{idx:<8} {char:<12} {freq:<12}")
            f.write("=" * 60 + "\n")
        
        print(f"  ✓ Saved confusion matrix mapping to {mapping_path.name}")


def plot_training_curves(training_info, save_path):
    """Plot comprehensive training curves"""
    if not training_info or 'train_losses' not in training_info:
        print("  ⚠ No training history available")
        return
    
    train_losses = training_info['train_losses']
    val_losses = training_info['val_losses']
    val_accuracies = training_info['val_accuracies']
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Train and Val Loss
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    axes[0, 1].plot(epochs, [acc * 100 for acc in val_accuracies], 'g-^', 
                    label='Val Accuracy', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # Plot 3: Loss Difference
    loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
    axes[1, 0].plot(epochs, loss_diff, 'm-d', linewidth=2, markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Val Loss - Train Loss', fontsize=11)
    axes[1, 0].set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(epochs, 0, loss_diff, where=[d > 0 for d in loss_diff], 
                            color='red', alpha=0.2, label='Overfitting')
    axes[1, 0].fill_between(epochs, 0, loss_diff, where=[d <= 0 for d in loss_diff], 
                            color='green', alpha=0.2, label='Underfitting')
    axes[1, 0].legend(fontsize=10)
    
    # Plot 4: Summary Stats
    axes[1, 1].axis('off')
    
    summary_text = f"""
    Training Summary (Hindi CNN)
    ─────────────────────────────
    Total Epochs: {len(epochs)}
    
    Final Metrics:
    • Train Loss: {train_losses[-1]:.4f}
    • Val Loss: {val_losses[-1]:.4f}
    • Val Accuracy: {val_accuracies[-1]*100:.2f}%
    
    Best Metrics:
    • Min Train Loss: {min(train_losses):.4f}
    • Min Val Loss: {min(val_losses):.4f}
    • Max Val Acc: {max(val_accuracies)*100:.2f}%
    
    Early Stopping:
    • Patience: {training_info.get('early_stopping_patience', 'N/A')}
    • Triggered: {'Yes' if len(epochs) < training_info.get('epochs', 20) else 'No'}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Training Curves and Analysis (Hindi CNN)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved training curves to {save_path}")


def plot_loss_comparison(train_loss, val_loss, test_loss, save_path):
    """Plot comparison of train/val/test losses"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    splits = ['Train', 'Validation', 'Test']
    losses = [train_loss, val_loss, test_loss]
    colors = ['steelblue', 'coral', 'seagreen']
    
    bars = ax.bar(splits, losses, color=colors, edgecolor='black', alpha=0.7, width=0.6)
    
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Comparison Across Splits (Hindi CNN)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved loss comparison to {save_path}")


def load_model(model_path, vocab_size, use_deep=False, device='cpu'):
    """Load trained CNN model"""
    if use_deep:
        model = DeepCharCNNDecryptor(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=256,
            num_layers=6,
            kernel_size=3,
            dropout=0.3
        )
    else:
        model = CharCNNDecryptor(
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


def evaluate_cipher(cipher_type, use_deep=False, data_dir='data/cipher_datasets_hindi_1000', device='cpu'):
    """Comprehensive evaluation for one cipher"""
    
    model_name = "cnn_deep_hindi" if use_deep else "cnn_hindi"
    model_display = "Deep Residual Hindi CNN" if use_deep else "Multi-Kernel Hindi CNN"
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_display.upper()}: {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    # Load training metrics
    metrics_path = Path('results') / f'{model_name}_{cipher_type}' / 'metrics.json'
    training_info = {}
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            training_info = json.load(f)
    
    print("Loading test data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders_hindi(
        cipher_type=cipher_type,
        data_dir=data_dir,
        batch_size=16,
        max_length=512
    )
    
    model_path = Path('results') / f'{model_name}_{cipher_type}' / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train first: python train_cnn_hindi.py --cipher {cipher_type}" + 
              (" --use-deep" if use_deep else ""))
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, vocab.vocab_size, use_deep, device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    evaluator = CNNEvaluatorHindi(model, vocab, device)
    
    # Save complete character mapping
    results_dir = Path('results') / f'lstm_hindi_{cipher_type}'
    results_dir.mkdir(parents=True, exist_ok=True)
    save_character_mapping(evaluator.alphabet_list, results_dir / 'character_mapping.txt')
    
    print("\nEvaluating on test set...")
    test_results = evaluator.evaluate_dataset(test_loader)
    
    print("Calculating losses on all splits...")
    train_loss = evaluator.calculate_loss(train_loader)
    val_loss = evaluator.calculate_loss(val_loader)
    test_loss = evaluator.calculate_loss(test_loader)
    
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
        print(f"  Epochs Trained:      {epochs_trained}")
        print(f"  Best Val Accuracy:   {best_val_acc*100:>6.2f}%")
        if 'early_stopping_patience' in training_info:
            print(f"  Early Stop Patience: {training_info['early_stopping_patience']}")
        if 'train_fraction' in training_info and training_info['train_fraction'] < 1.0:
            print(f"  Training Data Used:  {training_info['train_fraction']*100:.0f}%")
    
    print(f"\n📊 Test Set Metrics:")
    print(f"  Character Accuracy:  {test_results['summary']['avg_char_accuracy']*100:>6.2f}%")
    print(f"  Word Accuracy:       {test_results['summary']['avg_word_accuracy']*100:>6.2f}%")
    print(f"  Edit Distance Sim:   {test_results['summary']['avg_edit_distance']*100:>6.2f}%")
    
    print(f"\n📈 Statistics:")
    print(f"  Mean (char):         {test_results['summary']['avg_char_accuracy']*100:>6.2f}%")
    print(f"  Median (char):       {test_results['summary']['median_char_accuracy']*100:>6.2f}%")
    print(f"  Std Dev (char):      {test_results['summary']['std_char_accuracy']*100:>6.2f}%")
    print(f"  Min Accuracy:        {test_results['summary']['min_char_accuracy']*100:>6.2f}%")
    print(f"  Max Accuracy:        {test_results['summary']['max_char_accuracy']*100:>6.2f}%")
    print(f"  Total Samples:       {len(test_results['results'])}")
    
    print(f"\n📉 Loss Metrics:")
    print(f"  Train Loss:          {train_loss:.4f}")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Test Loss:           {test_loss:.4f}")
    
    # Save results
    results_dir = Path('results') / f'{model_name}_{cipher_type}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': test_results['summary'],
            'training_info': training_info,
            'losses': {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss
            },
            'per_article': test_results['results']
        }, f, indent=2, ensure_ascii=False)
    
    with open(results_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            **test_results['summary'],
            'model_type': model_display,
            'training_epochs': training_info.get('epochs_trained', 'unknown'),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'language': 'hindi',
            'script': 'devanagari'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Generate all plots
    if training_info:
        plot_training_curves(training_info, results_dir / 'training_curves.png')
    
    plot_loss_comparison(train_loss, val_loss, test_loss, 
                        results_dir / 'loss_comparison.png')
    
    evaluator.plot_top_confusions(
        test_results['confusion_dict'],
        results_dir / 'top_confusions.png',
        top_k=15
    )
    
    evaluator.plot_confusion_matrix_heatmap(
        test_results['confusion_matrix'],
        results_dir / 'confusion_matrix_heatmap.png',
        normalize=True,
        top_n=30
    )
    
    evaluator.plot_confusion_matrix_heatmap(
        test_results['confusion_matrix'],
        results_dir / 'confusion_matrix_counts.png',
        normalize=False,
        top_n=30
    )
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(test_results['char_accuracies'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Character Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Character Accuracy Distribution')
    axes[0].axvline(test_results['summary']['avg_char_accuracy'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(test_results['summary']['median_char_accuracy'], color='green', linestyle='--', linewidth=2, label='Median')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(test_results['word_accuracies'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Word Accuracy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Word Accuracy Distribution')
    axes[1].axvline(test_results['summary']['avg_word_accuracy'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(test_results['edit_distances'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Edit Distance Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Edit Distance Distribution')
    axes[2].axvline(test_results['summary']['avg_edit_distance'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved accuracy distributions to accuracy_distributions.png")
    
    print(f"\n{'='*80}")
    print(f"✓ All results saved to {results_dir}/")
    print(f"{'='*80}")
    print("\n📁 Generated Files:")
    print("  JSON Files:")
    print("    • detailed_results.json")
    print("    • test_results.json")
    print("\n  Visualization Files:")
    print("    • training_curves.png")
    print("    • loss_comparison.png")
    print("    • top_confusions.png")
    print("    • confusion_matrix_heatmap.png")
    print("    • confusion_matrix_counts.png")
    print("    • accuracy_distributions.png")
    print(f"{'='*80}\n")
    
    return test_results


def evaluate_all_ciphers(use_deep=False, data_dir='data/cipher_datasets_hindi_1000'):
    """Evaluate all 6 ciphers"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ciphers = ['caesar', 'atbash', 'affine', 'vigenere',
               'substitution_fixed', 'substitution_random']
    
    all_results = {}
    model_name = "cnn_deep_hindi" if use_deep else "cnn_hindi"
    model_display = "Deep Residual Hindi CNN" if use_deep else "Multi-Kernel Hindi CNN"
    
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    cipher_names = [c.replace('_', ' ').title() for c in all_results.keys()]
    char_accs = [s['avg_char_accuracy']*100 for s in all_results.values()]
    word_accs = [s['avg_word_accuracy']*100 for s in all_results.values()]
    edit_sims = [s['avg_edit_distance']*100 for s in all_results.values()]
    std_devs = [s['std_char_accuracy']*100 for s in all_results.values()]
    
    x = np.arange(len(cipher_names))
    width = 0.35
    
    # Plot 1: Character and Word Accuracy
    axes[0, 0].bar(x - width/2, char_accs, width, label='Character Accuracy', 
                   color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, word_accs, width, label='Word Accuracy', 
                   color='coral', edgecolor='black')
    axes[0, 0].set_xlabel('Cipher Type', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title(f'{model_display} - Character vs Word Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Character Accuracy Only
    bars = axes[0, 1].bar(cipher_names, char_accs, color='seagreen', 
                          edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Character Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Character Accuracy by Cipher', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 105])
    
    for bar, acc in zip(bars, char_accs):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Edit Distance
    axes[1, 0].bar(cipher_names, edit_sims, color='purple', 
                   edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Edit Distance Similarity (%)', fontsize=11)
    axes[1, 0].set_title('Edit Distance Similarity', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Standard Deviation
    axes[1, 1].bar(cipher_names, std_devs, color='orange', 
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Standard Deviation (%)', fontsize=11)
    axes[1, 1].set_title('Accuracy Variance (Std Dev)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_display} Performance Across All Ciphers', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(results_dir / f'all_{model_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comprehensive comparison to results/all_{model_name}_comparison.png")
    print(f"✓ Saved comparison data to results/all_{model_name}_comparison.json\n")


def compare_cnn_variants(cipher_type='caesar', data_dir='data/cipher_datasets_hindi_1000'):
    """Compare Multi-Kernel CNN vs Deep Residual CNN for a specific cipher"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"COMPARING HINDI CNN VARIANTS: {cipher_type.upper().replace('_', ' ')}")
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
    
    parser = argparse.ArgumentParser(description='Comprehensive Hindi CNN evaluation')
    parser.add_argument('--cipher', type=str, default='all',
                       choices=['caesar', 'atbash', 'affine', 'vigenere',
                               'substitution_fixed', 'substitution_random', 'all'],
                       help='Which cipher(s) to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/cipher_datasets_hindi_1000',
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
        subprocess.check_call(['pip', 'install', 'python-Levenshtein'])
        import Levenshtein
    
    if args.compare_variants:
        if args.cipher == 'all':
            print("❌ --compare-variants requires a specific cipher, not 'all'")
            print("   Example: python evaluate_cnn_hindi.py --cipher caesar --compare-variants")
        else:
            compare_cnn_variants(args.cipher, args.data_dir)
    elif args.cipher == 'all':
        evaluate_all_ciphers(args.use_deep, args.data_dir)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate_cipher(args.cipher, args.use_deep, args.data_dir, device)