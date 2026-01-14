"""
Train and Compare: Classical Ciphers vs Negative Controls
==========================================================
Verify that models learn patterns (not overfit) by testing on AES/DES
"""

import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NegativeControlExperiment:
    """Run and analyze negative control experiments"""
    
    def __init__(self):
        self.classical_ciphers = ['caesar', 'atbash', 'vigenere', 'substitution_fixed']
        self.negative_controls = ['AES', 'DES']
        self.models = ['mlp', 'lstm', 'cnn', 'transformer']
    
    def train_on_negative_controls(
        self,
        corpus_name: str = 'English',
        models: list = None,
        epochs: int = 20,
        batch_size: int = 32
    ):
        """Train models on AES and DES"""
        
        if models is None:
            models = self.models
        
        print("\n" + "="*70)
        print(" "*15 + "TRAINING ON NEGATIVE CONTROLS")
        print("="*70)
        print(f"Corpus: {corpus_name}")
        print(f"Models: {models}")
        print(f"Ciphers: {self.negative_controls}")
        print("="*70 + "\n")
        
        results_log = []
        
        for cipher in self.negative_controls:
            for model in models:
                print(f"\n{'='*70}")
                print(f"Training: {model.upper()} on {cipher}")
                print(f"{'='*70}\n")
                
                data_dir = f"data/cipher_datasets_negative_control/{corpus_name}/{cipher}"
                save_dir = f"results/negative_control/{corpus_name}/{cipher}/{model}"
                
                # Check if data exists
                if not Path(data_dir).exists():
                    print(f"⚠️  Data not found: {data_dir}")
                    continue
                
                # Train
                cmd = [
                    'python', 'train_cipher_model.py',
                    '--model', model,
                    '--cipher', cipher.lower(),
                    '--data-dir', data_dir,
                    '--save-dir', save_dir,
                    '--epochs', str(epochs),
                    '--batch-size', str(batch_size)
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    results_log.append({
                        'corpus': corpus_name,
                        'cipher': cipher,
                        'model': model,
                        'status': 'success'
                    })
                    print(f"✅ Completed: {model} on {cipher}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed: {model} on {cipher}")
                    results_log.append({
                        'corpus': corpus_name,
                        'cipher': cipher,
                        'model': model,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Save log
        log_path = Path(f"results/negative_control/{corpus_name}/training_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)
        
        print("\n✅ Negative control training complete!")
    
    def evaluate_negative_controls(
        self,
        corpus_name: str = 'English',
        models: list = None
    ):
        """Evaluate models on AES and DES"""
        
        if models is None:
            models = self.models
        
        print("\n" + "="*70)
        print(" "*15 + "EVALUATING NEGATIVE CONTROLS")
        print("="*70)
        
        for cipher in self.negative_controls:
            for model in models:
                data_dir = f"data/cipher_datasets_negative_control/{corpus_name}/{cipher}"
                results_base = f"results/negative_control/{corpus_name}/{cipher}"
                
                cmd = [
                    'python', 'evaluate_cipher_model.py',
                    '--model', model,
                    '--cipher', cipher.lower(),
                    '--data-dir', data_dir,
                    '--results-base', results_base
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"✅ Evaluated: {model} on {cipher}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Evaluation failed: {model} on {cipher}")
    
    def compare_classical_vs_negative(
        self,
        corpus_name: str = 'English',
        classical_results_dir: str = 'results/linguistic',
        negative_results_dir: str = 'results/negative_control'
    ):
        """Compare performance on classical ciphers vs negative controls"""
        
        print("\n" + "="*70)
        print(" "*10 + "CLASSICAL vs NEGATIVE CONTROL COMPARISON")
        print("="*70)
        
        results = []
        
        # Load classical cipher results
        for cipher in self.classical_ciphers:
            for model in self.models:
                result_file = Path(classical_results_dir) / corpus_name / f"{model}_{cipher}" / 'test_summary.json'
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        results.append({
                            'cipher_type': 'Classical',
                            'cipher': cipher,
                            'model': model,
                            'char_accuracy': data.get('avg_char_accuracy', 0) * 100,
                            'word_accuracy': data.get('avg_word_accuracy', 0) * 100
                        })
        
        # Load negative control results
        for cipher in self.negative_controls:
            for model in self.models:
                result_file = Path(negative_results_dir) / corpus_name / cipher / model / 'test_summary.json'
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        results.append({
                            'cipher_type': 'Negative Control',
                            'cipher': cipher,
                            'model': model,
                            'char_accuracy': data.get('avg_char_accuracy', 0) * 100,
                            'word_accuracy': data.get('avg_word_accuracy', 0) * 100
                        })
        
        if not results:
            print("\n❌ No results found!")
            return
        
        df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print("\n📊 Average Character Accuracy by Cipher Type:")
        print("-"*70)
        summary = df.groupby('cipher_type')['char_accuracy'].agg(['mean', 'std', 'min', 'max'])
        print(summary)
        
        print("\n📊 Per-Cipher Breakdown:")
        print("-"*70)
        cipher_summary = df.groupby(['cipher_type', 'cipher'])['char_accuracy'].mean().unstack(fill_value=0)
        print(cipher_summary)
        
        print("\n📊 Per-Model Breakdown:")
        print("-"*70)
        model_summary = df.groupby(['cipher_type', 'model'])['char_accuracy'].mean().unstack(fill_value=0)
        print(model_summary)
        
        # Statistical test
        classical_acc = df[df['cipher_type'] == 'Classical']['char_accuracy']
        negative_acc = df[df['cipher_type'] == 'Negative Control']['char_accuracy']
        
        print("\n📈 Statistical Comparison:")
        print("-"*70)
        print(f"Classical Ciphers:     {classical_acc.mean():.2f}% ± {classical_acc.std():.2f}%")
        print(f"Negative Controls:     {negative_acc.mean():.2f}% ± {negative_acc.std():.2f}%")
        print(f"Difference:            {(classical_acc.mean() - negative_acc.mean()):.2f}%")
        print(f"Random Baseline:       ~3.7% (1/27 characters)")
        
        # Interpretation
        print("\n🔍 INTERPRETATION:")
        print("="*70)
        
        if negative_acc.mean() < 10:
            print("✅ PASS: Models perform near-random on AES/DES")
            print("   → Models learn cipher patterns (not overfitting)")
            print("   → Neural networks successfully decipher classical ciphers")
        else:
            print("⚠️  WARNING: Models perform better than random on AES/DES")
            print("   → Possible overfitting or data leakage")
            print("   → Results may not be trustworthy")
        
        if classical_acc.mean() > 50:
            print("\n✅ PASS: Models perform well on classical ciphers")
            print("   → Models successfully learn exploitable patterns")
        else:
            print("\n❌ FAIL: Models perform poorly on classical ciphers")
            print("   → Training may need improvement")
        
        print("="*70)
        
        # Save comparison
        output_dir = Path(f'results/negative_control/{corpus_name}/analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / 'comparison_results.csv', index=False)
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump({
                'classical_mean': float(classical_acc.mean()),
                'classical_std': float(classical_acc.std()),
                'negative_mean': float(negative_acc.mean()),
                'negative_std': float(negative_acc.std()),
                'difference': float(classical_acc.mean() - negative_acc.mean()),
                'interpretation': 'PASS' if negative_acc.mean() < 10 else 'WARNING'
            }, f, indent=2)
        
        # Plot comparison
        self.plot_comparison(df, output_dir)
        
        return df
    
    def plot_comparison(self, df, output_dir):
        """Create comparison visualizations"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Box plot by cipher type
        df_classical = df[df['cipher_type'] == 'Classical']
        df_negative = df[df['cipher_type'] == 'Negative Control']
        
        data_to_plot = [
            df_classical['char_accuracy'].values,
            df_negative['char_accuracy'].values
        ]
        
        bp = axes[0].boxplot(data_to_plot, labels=['Classical\nCiphers', 'Negative\nControls'],
                            patch_artist=True, widths=0.5)
        
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[0].axhline(y=3.7, color='red', linestyle='--', linewidth=2, label='Random Baseline')
        axes[0].set_ylabel('Character Accuracy (%)', fontsize=12)
        axes[0].set_title('Classical vs Negative Control Performance', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Per-model comparison
        model_data = df.groupby(['model', 'cipher_type'])['char_accuracy'].mean().unstack()
        
        x = np.arange(len(self.models))
        width = 0.35
        
        axes[1].bar(x - width/2, model_data['Classical'], width, 
                   label='Classical', color='lightgreen', alpha=0.8)
        axes[1].bar(x + width/2, model_data['Negative Control'], width,
                   label='Negative Control', color='lightcoral', alpha=0.8)
        
        axes[1].set_xlabel('Model Architecture', fontsize=12)
        axes[1].set_ylabel('Character Accuracy (%)', fontsize=12)
        axes[1].set_title('Model Performance: Classical vs Negative', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m.upper() for m in self.models])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=3.7, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'classical_vs_negative_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved comparison plot: {output_dir}/classical_vs_negative_comparison.png")


def run_full_negative_control_experiment(corpus_name: str = 'English'):
    """Run complete negative control experiment"""
    
    experiment = NegativeControlExperiment()
    
    print("\n" + "="*80)
    print(" "*20 + "NEGATIVE CONTROL EXPERIMENT")
    print("="*80)
    print(f"Corpus: {corpus_name}")
    print("Purpose: Verify models learn patterns (not overfit)")
    print("="*80 + "\n")
    
    # Step 1: Train on negative controls
    print("\n" + "="*80)
    print("STEP 1: Training on AES/DES")
    print("="*80)
    experiment.train_on_negative_controls(corpus_name, epochs=20)
    
    # Step 2: Evaluate
    print("\n" + "="*80)
    print("STEP 2: Evaluating on AES/DES")
    print("="*80)
    experiment.evaluate_negative_controls(corpus_name)
    
    # Step 3: Compare
    print("\n" + "="*80)
    print("STEP 3: Comparing Classical vs Negative")
    print("="*80)
    df = experiment.compare_classical_vs_negative(corpus_name)
    
    print("\n" + "="*80)
    print(" "*25 + "EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved to: results/negative_control/{corpus_name}/analysis/")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Negative control experiments')
    parser.add_argument('--corpus', type=str, default='English',
                       help='Corpus name')
    parser.add_argument('--action', type=str, 
                       choices=['train', 'evaluate', 'compare', 'full'],
                       default='full',
                       help='Which step to run')
    parser.add_argument('--models', nargs='+',
                       default=['mlp', 'lstm', 'cnn', 'transformer'],
                       help='Models to train/evaluate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    experiment = NegativeControlExperiment()
    
    if args.action == 'train':
        experiment.train_on_negative_controls(
            corpus_name=args.corpus,
            models=args.models,
            epochs=args.epochs
        )
    elif args.action == 'evaluate':
        experiment.evaluate_negative_controls(
            corpus_name=args.corpus,
            models=args.models
        )
    elif args.action == 'compare':
        experiment.compare_classical_vs_negative(corpus_name=args.corpus)
    elif args.action == 'full':
        run_full_negative_control_experiment(corpus_name=args.corpus)