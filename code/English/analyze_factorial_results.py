"""
Factorial Analysis for HYBRID Experiments
Analyzes 2³ factorial design: MORPH × SYNTAX × PHONO
Computes main effects, interactions, and effect sizes
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import pandas as pd


class FactorialAnalyzer:
    """Analyze factorial HYBRID experiments"""
    
    def __init__(self, results_dir='results/linguistic/HYBRID-FACTORIAL'):
        self.results_dir = Path(results_dir)
        
        # Factorial structure
        self.variants = {
            'HYB-1': {'morph': 'isolating', 'syntax': 'SVO', 'phono': 'large'},
            'HYB-2': {'morph': 'isolating', 'syntax': 'SVO', 'phono': 'small'},
            'HYB-3': {'morph': 'isolating', 'syntax': 'free', 'phono': 'large'},
            'HYB-4': {'morph': 'isolating', 'syntax': 'free', 'phono': 'small'},
            'HYB-5': {'morph': 'polysynthetic', 'syntax': 'SVO', 'phono': 'large'},
            'HYB-6': {'morph': 'polysynthetic', 'syntax': 'SVO', 'phono': 'small'},
            'HYB-7': {'morph': 'polysynthetic', 'syntax': 'free', 'phono': 'large'},
            'HYB-8': {'morph': 'polysynthetic', 'syntax': 'free', 'phono': 'small'}
        }
        
        self.models = ['mlp', 'lstm', 'cnn', 'transformer']
    
    def load_results(self, cipher='caesar'):
        """Load test results for all variants and models"""
        
        results = []
        
        print(f"Loading results from {self.results_dir}...")
        
        for variant in self.variants.keys():
            for model in self.models:
                result_file = self.results_dir / variant / f"{model}_{cipher}" / 'test_summary.json'
                
                if not result_file.exists():
                    print(f"⚠️  Missing: {variant} - {model}")
                    continue
                
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                results.append({
                    'variant': variant,
                    'model': model,
                    'morph': self.variants[variant]['morph'],
                    'syntax': self.variants[variant]['syntax'],
                    'phono': self.variants[variant]['phono'],
                    'char_accuracy': data.get('avg_char_accuracy', 0),
                    'word_accuracy': data.get('avg_word_accuracy', 0),
                    'edit_distance': data.get('avg_edit_distance', 0)
                })
        
        df = pd.DataFrame(results)
        print(f"✓ Loaded {len(df)} results\n")
        
        return df
    
    def compute_main_effects(self, df, metric='char_accuracy'):
        """Compute main effects for each factor"""
        
        print("\n" + "="*70)
        print(f"MAIN EFFECTS - {metric.upper()}")
        print("="*70)
        
        effects = {}
        
        # MORPH effect
        isolating = df[df['morph'] == 'isolating'][metric].mean()
        polysynthetic = df[df['morph'] == 'polysynthetic'][metric].mean()
        morph_effect = abs(isolating - polysynthetic)
        
        effects['MORPH'] = {
            'isolating': isolating,
            'polysynthetic': polysynthetic,
            'effect_size': morph_effect,
            'direction': 'isolating > poly' if isolating > polysynthetic else 'poly > isolating'
        }
        
        print(f"\n📊 MORPH Effect:")
        print(f"  Isolating:     {isolating*100:.2f}%")
        print(f"  Polysynthetic: {polysynthetic*100:.2f}%")
        print(f"  Effect size:   {morph_effect*100:.2f}% ({effects['MORPH']['direction']})")
        
        # SYNTAX effect
        svo = df[df['syntax'] == 'SVO'][metric].mean()
        free = df[df['syntax'] == 'free'][metric].mean()
        syntax_effect = abs(svo - free)
        
        effects['SYNTAX'] = {
            'SVO': svo,
            'free': free,
            'effect_size': syntax_effect,
            'direction': 'SVO > free' if svo > free else 'free > SVO'
        }
        
        print(f"\n📊 SYNTAX Effect:")
        print(f"  SVO:           {svo*100:.2f}%")
        print(f"  Free:          {free*100:.2f}%")
        print(f"  Effect size:   {syntax_effect*100:.2f}% ({effects['SYNTAX']['direction']})")
        
        # PHONO effect
        large = df[df['phono'] == 'large'][metric].mean()
        small = df[df['phono'] == 'small'][metric].mean()
        phono_effect = abs(large - small)
        
        effects['PHONO'] = {
            'large': large,
            'small': small,
            'effect_size': phono_effect,
            'direction': 'large > small' if large > small else 'small > large'
        }
        
        print(f"\n📊 PHONO Effect:")
        print(f"  Large:         {large*100:.2f}%")
        print(f"  Small:         {small*100:.2f}%")
        print(f"  Effect size:   {phono_effect*100:.2f}% ({effects['PHONO']['direction']})")
        
        # Rank effects
        ranked = sorted(effects.items(), key=lambda x: x[1]['effect_size'], reverse=True)
        
        print(f"\n🏆 Feature Dominance Ranking:")
        for i, (feature, data) in enumerate(ranked, 1):
            print(f"  {i}. {feature}: {data['effect_size']*100:.2f}% effect")
        
        print("="*70)
        
        return effects
    
    def compute_interactions(self, df, metric='char_accuracy'):
        """Compute two-way and three-way interactions"""
        
        print("\n" + "="*70)
        print(f"INTERACTION EFFECTS - {metric.upper()}")
        print("="*70)
        
        interactions = {}
        
        # MORPH × SYNTAX
        iso_svo = df[(df['morph'] == 'isolating') & (df['syntax'] == 'SVO')][metric].mean()
        iso_free = df[(df['morph'] == 'isolating') & (df['syntax'] == 'free')][metric].mean()
        poly_svo = df[(df['morph'] == 'polysynthetic') & (df['syntax'] == 'SVO')][metric].mean()
        poly_free = df[(df['morph'] == 'polysynthetic') & (df['syntax'] == 'free')][metric].mean()
        
        morph_effect_svo = abs(iso_svo - poly_svo)
        morph_effect_free = abs(iso_free - poly_free)
        morph_syntax_interaction = abs(morph_effect_svo - morph_effect_free)
        
        interactions['MORPH × SYNTAX'] = {
            'interaction_size': morph_syntax_interaction,
            'values': {
                'iso_svo': iso_svo,
                'iso_free': iso_free,
                'poly_svo': poly_svo,
                'poly_free': poly_free
            }
        }
        
        print(f"\n📊 MORPH × SYNTAX Interaction:")
        print(f"  Iso+SVO:  {iso_svo*100:.2f}%")
        print(f"  Iso+Free: {iso_free*100:.2f}%")
        print(f"  Poly+SVO: {poly_svo*100:.2f}%")
        print(f"  Poly+Free: {poly_free*100:.2f}%")
        print(f"  Interaction size: {morph_syntax_interaction*100:.2f}%")
        
        # MORPH × PHONO
        iso_large = df[(df['morph'] == 'isolating') & (df['phono'] == 'large')][metric].mean()
        iso_small = df[(df['morph'] == 'isolating') & (df['phono'] == 'small')][metric].mean()
        poly_large = df[(df['morph'] == 'polysynthetic') & (df['phono'] == 'large')][metric].mean()
        poly_small = df[(df['morph'] == 'polysynthetic') & (df['phono'] == 'small')][metric].mean()
        
        morph_effect_large = abs(iso_large - poly_large)
        morph_effect_small = abs(iso_small - poly_small)
        morph_phono_interaction = abs(morph_effect_large - morph_effect_small)
        
        interactions['MORPH × PHONO'] = {
            'interaction_size': morph_phono_interaction,
            'values': {
                'iso_large': iso_large,
                'iso_small': iso_small,
                'poly_large': poly_large,
                'poly_small': poly_small
            }
        }
        
        print(f"\n📊 MORPH × PHONO Interaction:")
        print(f"  Iso+Large:  {iso_large*100:.2f}%")
        print(f"  Iso+Small:  {iso_small*100:.2f}%")
        print(f"  Poly+Large: {poly_large*100:.2f}%")
        print(f"  Poly+Small: {poly_small*100:.2f}%")
        print(f"  Interaction size: {morph_phono_interaction*100:.2f}%")
        
        # SYNTAX × PHONO
        svo_large = df[(df['syntax'] == 'SVO') & (df['phono'] == 'large')][metric].mean()
        svo_small = df[(df['syntax'] == 'SVO') & (df['phono'] == 'small')][metric].mean()
        free_large = df[(df['syntax'] == 'free') & (df['phono'] == 'large')][metric].mean()
        free_small = df[(df['syntax'] == 'free') & (df['phono'] == 'small')][metric].mean()
        
        syntax_effect_large = abs(svo_large - free_large)
        syntax_effect_small = abs(svo_small - free_small)
        syntax_phono_interaction = abs(syntax_effect_large - syntax_effect_small)
        
        interactions['SYNTAX × PHONO'] = {
            'interaction_size': syntax_phono_interaction,
            'values': {
                'svo_large': svo_large,
                'svo_small': svo_small,
                'free_large': free_large,
                'free_small': free_small
            }
        }
        
        print(f"\n📊 SYNTAX × PHONO Interaction:")
        print(f"  SVO+Large:  {svo_large*100:.2f}%")
        print(f"  SVO+Small:  {svo_small*100:.2f}%")
        print(f"  Free+Large: {free_large*100:.2f}%")
        print(f"  Free+Small: {free_small*100:.2f}%")
        print(f"  Interaction size: {syntax_phono_interaction*100:.2f}%")
        
        print("="*70)
        
        return interactions
    
    def compare_complexity_levels(self, df, metric='char_accuracy'):
        """Compare simplest vs most complex combinations"""
        
        print("\n" + "="*70)
        print(f"COMPLEXITY COMPARISON - {metric.upper()}")
        print("="*70)
        
        # HYB-1: All simple (baseline)
        hyb1 = df[df['variant'] == 'HYB-1'][metric].mean()
        
        # HYB-8: All complex
        hyb8 = df[df['variant'] == 'HYB-8'][metric].mean()
        
        drop = (hyb1 - hyb8)
        percent_drop = (drop / hyb1) * 100 if hyb1 > 0 else 0
        
        print(f"\n🔹 HYB-1 (Baseline - All Simple):")
        print(f"   Isolating + SVO + Large: {hyb1*100:.2f}%")
        
        print(f"\n🔹 HYB-8 (Most Complex):")
        print(f"   Polysynthetic + Free + Small: {hyb8*100:.2f}%")
        
        print(f"\n📉 Complexity Impact:")
        print(f"   Absolute drop: {drop*100:.2f}%")
        print(f"   Relative drop: {percent_drop:.1f}%")
        
        # All variants ranked
        print(f"\n🏆 Complexity Ranking (Easiest → Hardest):")
        variant_scores = df.groupby('variant')[metric].mean().sort_values(ascending=False)
        
        for i, (variant, score) in enumerate(variant_scores.items(), 1):
            config = self.variants[variant]
            print(f"  {i}. {variant}: {score*100:.2f}% "
                  f"({config['morph'][:3]}+{config['syntax'][:3]}+{config['phono'][:3]})")
        
        print("="*70)
        
        return {'hyb1': hyb1, 'hyb8': hyb8, 'drop': drop, 'ranking': variant_scores}
    
    def analyze_model_differences(self, df, metric='char_accuracy'):
        """Analyze if effects differ by model architecture"""
        
        print("\n" + "="*70)
        print(f"MODEL-SPECIFIC ANALYSIS - {metric.upper()}")
        print("="*70)
        
        for model in self.models:
            model_df = df[df['model'] == model]
            
            if len(model_df) == 0:
                continue
            
            print(f"\n🔧 {model.upper()}:")
            
            # Main effects for this model
            iso = model_df[model_df['morph'] == 'isolating'][metric].mean()
            poly = model_df[model_df['morph'] == 'polysynthetic'][metric].mean()
            morph_effect = abs(iso - poly)
            
            svo = model_df[model_df['syntax'] == 'SVO'][metric].mean()
            free = model_df[model_df['syntax'] == 'free'][metric].mean()
            syntax_effect = abs(svo - free)
            
            large = model_df[model_df['phono'] == 'large'][metric].mean()
            small = model_df[model_df['phono'] == 'small'][metric].mean()
            phono_effect = abs(large - small)
            
            print(f"  MORPH effect:  {morph_effect*100:.2f}%")
            print(f"  SYNTAX effect: {syntax_effect*100:.2f}%")
            print(f"  PHONO effect:  {phono_effect*100:.2f}%")
            
            # Which feature dominates for this model?
            effects_dict = {'MORPH': morph_effect, 'SYNTAX': syntax_effect, 'PHONO': phono_effect}
            dominant = max(effects_dict, key=effects_dict.get)
            print(f"  → Dominant feature: {dominant}")
        
        print("="*70)
    
    def plot_main_effects(self, df, metric='char_accuracy', output_dir=None):
        """Plot main effects"""
        
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MORPH
        morph_data = df.groupby('morph')[metric].agg(['mean', 'std'])
        axes[0].bar(morph_data.index, morph_data['mean']*100, yerr=morph_data['std']*100,
                   color=['steelblue', 'coral'], alpha=0.7, capsize=5)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('MORPH Effect', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # SYNTAX
        syntax_data = df.groupby('syntax')[metric].agg(['mean', 'std'])
        axes[1].bar(syntax_data.index, syntax_data['mean']*100, yerr=syntax_data['std']*100,
                   color=['steelblue', 'coral'], alpha=0.7, capsize=5)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('SYNTAX Effect', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # PHONO
        phono_data = df.groupby('phono')[metric].agg(['mean', 'std'])
        axes[2].bar(phono_data.index, phono_data['mean']*100, yerr=phono_data['std']*100,
                   color=['steelblue', 'coral'], alpha=0.7, capsize=5)
        axes[2].set_ylabel('Accuracy (%)', fontsize=12)
        axes[2].set_title('PHONO Effect', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'main_effects_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved main effects plot: {output_dir}/main_effects_{metric}.png")
    
    def plot_interactions(self, df, metric='char_accuracy', output_dir=None):
        """Plot interaction effects"""
        
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        output_dir = Path(output_dir)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MORPH × SYNTAX
        for morph in ['isolating', 'polysynthetic']:
            data = df[df['morph'] == morph].groupby('syntax')[metric].mean() * 100
            axes[0].plot(['SVO', 'Free'], data.values, marker='o', linewidth=2, 
                        markersize=8, label=morph.capitalize())
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_xlabel('Syntax', fontsize=12)
        axes[0].set_title('MORPH × SYNTAX Interaction', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MORPH × PHONO
        for morph in ['isolating', 'polysynthetic']:
            data = df[df['morph'] == morph].groupby('phono')[metric].mean() * 100
            axes[1].plot(['Large', 'Small'], data.values, marker='o', linewidth=2,
                        markersize=8, label=morph.capitalize())
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_xlabel('Phonology', fontsize=12)
        axes[1].set_title('MORPH × PHONO Interaction', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # SYNTAX × PHONO
        for syntax in ['SVO', 'free']:
            data = df[df['syntax'] == syntax].groupby('phono')[metric].mean() * 100
            axes[2].plot(['Large', 'Small'], data.values, marker='o', linewidth=2,
                        markersize=8, label=syntax.upper())
        axes[2].set_ylabel('Accuracy (%)', fontsize=12)
        axes[2].set_xlabel('Phonology', fontsize=12)
        axes[2].set_title('SYNTAX × PHONO Interaction', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'interactions_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved interaction plot: {output_dir}/interactions_{metric}.png")
    
    def plot_complexity_heatmap(self, df, metric='char_accuracy', output_dir=None):
        """Plot heatmap of all 8 variants"""
        
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        output_dir = Path(output_dir)
        
        # Create matrix
        variant_scores = df.groupby('variant')[metric].mean() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by score
        variant_scores = variant_scores.sort_values(ascending=False)
        
        # Create labels with feature info
        labels = []
        for variant in variant_scores.index:
            config = self.variants[variant]
            label = f"{variant}\n{config['morph'][:4]}+{config['syntax'][:4]}+{config['phono'][:4]}"
            labels.append(label)
        
        # Bar plot
        colors = plt.cm.RdYlGn(variant_scores.values / 100)
        bars = ax.barh(range(len(variant_scores)), variant_scores.values, color=colors)
        ax.set_yticks(range(len(variant_scores)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Character Accuracy (%)', fontsize=12)
        ax.set_title('Complexity Ranking: All 8 Variants', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, variant_scores.values)):
            ax.text(score + 0.5, i, f'{score:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'complexity_ranking_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved complexity ranking: {output_dir}/complexity_ranking_{metric}.png")
    
    def generate_full_report(self, cipher='caesar', metric='char_accuracy'):
        """Generate complete factorial analysis"""
        
        print("\n" + "="*70)
        print(" "*15 + "FACTORIAL ANALYSIS REPORT")
        print("="*70)
        print(f"Cipher: {cipher}")
        print(f"Metric: {metric}")
        print("="*70)
        
        # Load data
        df = self.load_results(cipher)
        
        if len(df) == 0:
            print("\n❌ No results found!")
            return
        
        # Main effects
        main_effects = self.compute_main_effects(df, metric)
        
        # Interactions
        interactions = self.compute_interactions(df, metric)
        
        # Complexity comparison
        complexity = self.compare_complexity_levels(df, metric)
        
        # Model-specific analysis
        self.analyze_model_differences(df, metric)
        
        # Plots
        print("\n📊 Generating visualizations...")
        self.plot_main_effects(df, metric)
        self.plot_interactions(df, metric)
        self.plot_complexity_heatmap(df, metric)
        
        # Save summary
        output_dir = self.results_dir / 'analysis'
        summary = {
            'main_effects': main_effects,
            'interactions': {k: {'size': v['interaction_size']} for k, v in interactions.items()},
            'complexity': {
                'baseline_hyb1': complexity['hyb1'],
                'complex_hyb8': complexity['hyb8'],
                'drop': complexity['drop']
            }
        }
        
        with open(output_dir / f'factorial_summary_{metric}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")
        print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze HYBRID factorial results')
    parser.add_argument('--results-dir', type=str,
                       default='results/linguistic/HYBRID-FACTORIAL',
                       help='Results directory')
    parser.add_argument('--cipher', type=str, default='caesar',
                       help='Cipher type')
    parser.add_argument('--metric', type=str, default='char_accuracy',
                       choices=['char_accuracy', 'word_accuracy', 'edit_distance'],
                       help='Metric to analyze')
    
    args = parser.parse_args()
    
    analyzer = FactorialAnalyzer(args.results_dir)
    analyzer.generate_full_report(args.cipher, args.metric)