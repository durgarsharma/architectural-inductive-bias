"""
Comprehensive Model Evaluation - All 6 Ciphers (Hindi/Devanagari)
Includes: Character accuracy, Word accuracy, Edit distance, Per-article breakdown, 
Full Confusion Matrix, Training/Validation/Testing visualizations
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

from models.lstm_cipher_model import SimpleLSTMDecryptor
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


class ComprehensiveEvaluatorHindi:
    """Comprehensive evaluation with multiple metrics for Hindi"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
        
        # Hindi alphabet (consonants + vowels)
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
        """Word-level accuracy (exact word matches)"""
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
        """Build character confusion matrix for Hindi"""
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
        """Comprehensive evaluation on entire dataset"""
        all_results = []
        all_char_accuracies = []
        all_word_accuracies = []
        all_edit_distances = []
        global_confusion = {char: Counter() for char in self.alphabet}
        
        # Full confusion matrix (for heatmap)
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
                    
                    # Calculate all metrics
                    char_acc = self.calculate_character_accuracy(pred_text, target_text)
                    word_acc = self.calculate_word_accuracy(pred_text, target_text)
                    edit_dist = self.calculate_edit_distance(pred_text, target_text)
                    
                    # Update confusion matrix
                    confusion = self.build_confusion_matrix(pred_text, target_text)
                    for char, counts in confusion.items():
                        global_confusion[char].update(counts)
                    
                    # Update full confusion matrix for heatmap
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
                        'cipher': cipher_text[:200],  # First 200 chars
                        'predicted': pred_text[:200],
                        'target': target_text[:200],
                        'char_accuracy': char_acc,
                        'word_accuracy': word_acc,
                        'edit_distance': edit_dist,
                        'length': length.item()
                    })
        
        # Normalize confusion matrix (row-wise)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
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
            f.write(f"TOP {top_k} CHARACTER CONFUSIONS - DETAILED MAPPING) ")
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


    def plot_per_character_accuracy(self, confusion_matrix, save_path, top_n=25):
        """Plot per-character accuracy using indices"""
        # Calculate per-character accuracy
        char_accuracies = []
        char_counts = []
        
        for i, char in enumerate(self.alphabet_list):
            total = confusion_matrix[i].sum()
            if total > 0:
                correct = confusion_matrix[i, i]
                accuracy = correct / total
                char_accuracies.append((i, char, accuracy, total))
                char_counts.append(total)
        
        # Sort by frequency and take top N
        char_accuracies.sort(key=lambda x: x[3], reverse=True)
        top_chars = char_accuracies[:top_n]
        
        # Prepare data
        char_indices = [c[0] for c in top_chars]
        char_labels = [f"Idx_{c[0]}" for c in top_chars]
        actual_chars = [c[1] for c in top_chars]
        accs = [c[2] * 100 for c in top_chars]
        counts = [c[3] for c in top_chars]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Accuracy bars
        colors = ['darkgreen' if acc >= 95 else 'orange' if acc >= 80 else 'red' for acc in accs]
        bars1 = ax1.barh(range(len(char_labels)), accs, color=colors, 
                         edgecolor='black', alpha=0.7)
        
        ax1.set_yticks(range(len(char_labels)))
        ax1.set_yticklabels(char_labels, fontsize=9)
        ax1.set_xlabel('Accuracy (%)', fontsize=12)
        ax1.set_title(f'Per-Character Accuracy (Top {top_n} Most Frequent)', 
                      fontsize=14, fontweight='bold')
        ax1.axvline(x=95, color='green', linestyle='--', linewidth=2, 
                    label='95% threshold', alpha=0.7)
        ax1.axvline(x=80, color='orange', linestyle='--', linewidth=2, 
                    label='80% threshold', alpha=0.7)
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        ax1.set_xlim([0, 105])
        
        # Add accuracy values
        for bar, acc in zip(bars1, accs):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.1f}%', va='center', fontsize=8, fontweight='bold')
        
        # Plot 2: Character frequency
        bars2 = ax2.barh(range(len(char_labels)), counts, color='steelblue', 
                         edgecolor='black', alpha=0.7)
        
        ax2.set_yticks(range(len(char_labels)))
        ax2.set_yticklabels(char_labels, fontsize=9)
        ax2.set_xlabel('Frequency (count)', fontsize=12)
        ax2.set_title(f'Character Frequency (Top {top_n})', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add frequency values
        for bar, freq in zip(bars2, counts):
            width = bar.get_width()
            ax2.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(freq)}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved per-character accuracy to {save_path}")
        
        # Save mapping
        mapping_path = Path(save_path).parent / f'top_{top_n}_character_mapping.txt'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"TOP {top_n} CHARACTERS - INDEX TO CHARACTER MAPPING")
            f.write("=" * 70 + "\n")
            f.write(f"{'Index':<8} {'Character':<12} {'Accuracy':<12} {'Frequency':<12}")
            f.write("-" * 70 + "\n")
            for idx, char, acc, freq in zip(char_indices, actual_chars, accs, counts):
                f.write(f"{idx:<8} {char:<12} {acc:>6.2f}%      {int(freq):<12}")
            f.write("=" * 70 + "\n")
        
        print(f"  ✓ Saved top {top_n} character mapping to {mapping_path.name}")


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
    
    # Plot 3: Loss Difference (Overfitting indicator)
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
    
    # Plot 4: Learning Rate (if available) or Summary Stats
    axes[1, 1].axis('off')
    
    # Add summary text
    summary_text = f"""
    Training Summary
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
    
    plt.suptitle('Training Curves and Analysis', fontsize=14, fontweight='bold', y=0.995)
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
    ax.set_title('Loss Comparison Across Splits', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved loss comparison to {save_path}")


def load_model(model_path, vocab_size, device='cpu'):
    """Load trained model"""
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


def evaluate_cipher(cipher_type, data_dir='data/cipher_datasets_hindi_1000', device='cpu'):
    """Comprehensive evaluation for one cipher"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: Hindi {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    # Load training metrics to show training info
    metrics_path = Path('results') / f'lstm_hindi_{cipher_type}' / 'metrics.json'
    training_info = {}
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            training_info = json.load(f)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders_hindi(
        cipher_type=cipher_type,
        data_dir=data_dir,
        batch_size=16,
        max_length=512
    )
    
    # Load model
    model_path = Path('results') / f'lstm_hindi_{cipher_type}' / 'best_model.pt'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train first: python train_lstm_hindi.py --cipher {cipher_type}")
        return None
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, vocab.vocab_size, device)
    
    # Evaluate
    evaluator = ComprehensiveEvaluatorHindi(model, vocab, device)
    
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
    print(f"COMPREHENSIVE RESULTS - Hindi {cipher_type.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    # Show training info if available
    if training_info:
        epochs_trained = training_info.get('epochs_trained', len(training_info.get('train_losses', [])))
        best_val_acc = training_info.get('best_val_accuracy', 0)
        print(f"\n🏋️  Training Info:")
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
    
    # Save detailed results
    results_dir = Path('results') / f'lstm_hindi_{cipher_type}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-article breakdown
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
    
    # Save summary
    with open(results_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            **test_results['summary'],
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
    
    # Plot 1: Training curves
    if training_info:
        plot_training_curves(training_info, results_dir / 'training_curves.png')
    
    # Plot 2: Loss comparison
    plot_loss_comparison(train_loss, val_loss, test_loss, 
                        results_dir / 'loss_comparison.png')
    
    # Plot 3: Top confusions
    evaluator.plot_top_confusions(
        test_results['confusion_dict'],
        results_dir / 'top_confusions.png',
        top_k=15
    )
    
    # Plot 4: Full confusion matrix heatmap (normalized)
    evaluator.plot_confusion_matrix_heatmap(
        test_results['confusion_matrix'],
        results_dir / 'confusion_matrix_heatmap.png',
        normalize=True,
        top_n=30
    )
    
    # Plot 5: Full confusion matrix heatmap (counts)
    evaluator.plot_confusion_matrix_heatmap(
        test_results['confusion_matrix'],
        results_dir / 'confusion_matrix_counts.png',
        normalize=False,
        top_n=30
    )
    
    # Plot 6: Per-character accuracy
    evaluator.plot_per_character_accuracy(
        test_results['confusion_matrix'],
        results_dir / 'per_character_accuracy.png',
        top_n=25
    )
    
    # Plot 7: Accuracy distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(test_results['char_accuracies'], bins=20, color='blue', 
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Character Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Character Accuracy Distribution')
    axes[0].axvline(test_results['summary']['avg_char_accuracy'], 
                   color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(test_results['summary']['median_char_accuracy'], 
                   color='green', linestyle='--', linewidth=2, label='Median')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(test_results['word_accuracies'], bins=20, color='green', 
                 alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Word Accuracy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Word Accuracy Distribution')
    axes[1].axvline(test_results['summary']['avg_word_accuracy'], 
                   color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(test_results['edit_distances'], bins=20, color='orange', 
                 alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Edit Distance Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Edit Distance Distribution')
    axes[2].axvline(test_results['summary']['avg_edit_distance'], 
                   color='red', linestyle='--', linewidth=2, label='Mean')
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
    print("    • detailed_results.json (per-article breakdown)")
    print("    • test_results.json (summary)")
    print("\n  Visualization Files:")
    print("    • training_curves.png (train/val loss and accuracy over epochs)")
    print("    • loss_comparison.png (train/val/test loss comparison)")
    print("    • top_confusions.png (most confused character pairs)")
    print("    • confusion_matrix_heatmap.png (normalized confusion matrix)")
    print("    • confusion_matrix_counts.png (raw count confusion matrix)")
    print("    • per_character_accuracy.png (accuracy per character)")
    print("    • accuracy_distributions.png (char/word/edit distance distributions)")
    print(f"{'='*80}\n")
    
    return test_results


def evaluate_all_ciphers(data_dir='data/cipher_datasets_hindi_1000'):
    """Evaluate all 6 ciphers and create comparison"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ciphers = ['caesar', 'atbash', 'affine', 'vigenere', 
               'substitution_fixed', 'substitution_random']
    
    all_results = {}
    
    for cipher in ciphers:
        result = evaluate_cipher(cipher, data_dir, device)
        if result:
            all_results[cipher] = result['summary']
    
    if not all_results:
        print("\n❌ No results to compare. Train models first!")
        return
    
    # Create comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON - ALL HINDI CIPHERS")
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
    
    with open(results_dir / 'all_hindi_ciphers_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Create comprehensive comparison plot
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
    axes[0, 0].set_title('Character vs Word Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Character Accuracy Only (with values)
    bars = axes[0, 1].bar(cipher_names, char_accs, color='seagreen', 
                          edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Character Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Character Accuracy by Cipher', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticklabels(cipher_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 105])
    
    # Add value labels
    for bar, acc in zip(bars, char_accs):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Edit Distance Similarity
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
    
    plt.suptitle('LSTM Performance Across All Hindi Ciphers', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(results_dir / 'all_hindi_ciphers_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comprehensive comparison to results/all_hindi_ciphers_comparison.png")
    print(f"✓ Saved comparison data to results/all_hindi_ciphers_comparison.json\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of Hindi cipher models')
    parser.add_argument('--cipher', type=str, default='all',
                       choices=['caesar', 'atbash', 'affine', 'vigenere',
                               'substitution_fixed', 'substitution_random', 'all'],
                       help='Which cipher(s) to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/cipher_datasets_hindi_1000',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # Install Levenshtein if needed
    try:
        import Levenshtein
    except ImportError:
        print("Installing python-Levenshtein...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'python-Levenshtein'])
        import Levenshtein
    
    if args.cipher == 'all':
        evaluate_all_ciphers(args.data_dir)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate_cipher(args.cipher, args.data_dir, device)