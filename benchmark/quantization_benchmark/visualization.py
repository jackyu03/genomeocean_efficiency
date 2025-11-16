"""
Visualization Module

Creates comprehensive visualizations for quantization benchmark results
including performance metrics and quality evaluations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import torch


def setup_plot_style():
    """Set up consistent plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_performance_metrics(df: pd.DataFrame, output_dir: Path):
    """Create performance metric visualizations."""
    
    setup_plot_style()
    
    # 1. Throughput comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.boxplot(data=df, x='quantization', y='seqs_per_s', ax=ax1)
    ax1.set_title('Sequences per Second by Quantization Mode', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Quantization Mode')
    ax1.set_ylabel('Sequences/sec')
    ax1.tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='quantization', y='tokens_per_s', ax=ax2)
    ax2.set_title('Tokens per Second by Quantization Mode', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantization Mode')
    ax2.set_ylabel('Tokens/sec')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory usage comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='quantization', y='peak_vram_GB')
    plt.title('Peak VRAM Usage by Quantization Mode', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Mode')
    plt.ylabel('Peak VRAM (GB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Latency comparison
    if 'E2EL_ms' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='quantization', y='E2EL_ms')
        plt.title('End-to-End Latency by Quantization Mode', fontsize=14, fontweight='bold')
        plt.xlabel('Quantization Mode')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Energy efficiency (if available)
    energy_df = df.dropna(subset=['tokens_per_watt'])
    if len(energy_df) > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=energy_df, x='quantization', y='tokens_per_watt')
        plt.title('Energy Efficiency by Quantization Mode', fontsize=14, fontweight='bold')
        plt.xlabel('Quantization Mode')
        plt.ylabel('Tokens per Watt')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Performance vs Memory tradeoff
    plt.figure(figsize=(12, 8))
    for quant_mode in sorted(df['quantization'].unique()):
        mode_data = df[df['quantization'] == quant_mode]
        plt.scatter(mode_data['peak_vram_GB'], mode_data['tokens_per_s'], 
                   label=quant_mode, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Peak VRAM (GB)', fontsize=12)
    plt.ylabel('Tokens per Second', fontsize=12)
    plt.title('Performance vs Memory Usage Tradeoff', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_memory_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance visualizations saved to: {output_dir}")


def plot_quality_metrics(quality_df: pd.DataFrame, output_dir: Path):
    """Create quality metric visualizations."""
    
    setup_plot_style()
    
    # 1. KL Divergence comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    pivot_kl = quality_df.pivot(index='quantization_mode', columns='layer', values='kl_divergence')
    pivot_kl.plot(kind='bar', ax=ax1)
    ax1.set_title('KL Divergence by Quantization Mode', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Quantization Mode')
    ax1.set_ylabel('KL Divergence')
    ax1.legend(title='Layer')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. JS Divergence comparison
    pivot_js = quality_df.pivot(index='quantization_mode', columns='layer', values='js_divergence')
    pivot_js.plot(kind='bar', ax=ax2)
    ax2.set_title('Jensen-Shannon Divergence by Quantization Mode', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantization Mode')
    ax2.set_ylabel('JS Divergence')
    ax2.legend(title='Layer')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'divergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cosine Similarity comparison
    plt.figure(figsize=(10, 6))
    pivot_cos = quality_df.pivot(index='quantization_mode', columns='layer', values='cosine_similarity')
    pivot_cos.plot(kind='bar')
    plt.title('Cosine Similarity by Quantization Mode', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Mode')
    plt.ylabel('Cosine Similarity')
    plt.legend(title='Layer')
    plt.xticks(rotation=45)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect similarity')
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SNR comparison
    plt.figure(figsize=(10, 6))
    pivot_snr = quality_df.pivot(index='quantization_mode', columns='layer', values='snr_db')
    pivot_snr.plot(kind='bar')
    plt.title('Signal-to-Noise Ratio by Quantization Mode', fontsize=14, fontweight='bold')
    plt.xlabel('Quantization Mode')
    plt.ylabel('SNR (dB)')
    plt.legend(title='Layer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'snr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Individual metric comparisons (better than heatmap for different scales)
    for layer in quality_df['layer'].unique():
        layer_data = quality_df[quality_df['layer'] == layer]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # KL Divergence (lower is better)
        ax = axes[0]
        layer_data.plot(x='quantization_mode', y='kl_divergence', kind='bar', ax=ax, color='coral', legend=False)
        ax.set_title('KL Divergence (Lower = Better)', fontweight='bold')
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('KL Divergence')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # JS Divergence (lower is better)
        ax = axes[1]
        layer_data.plot(x='quantization_mode', y='js_divergence', kind='bar', ax=ax, color='salmon', legend=False)
        ax.set_title('JS Divergence (Lower = Better)', fontweight='bold')
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('JS Divergence')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Cosine Similarity (higher is better)
        ax = axes[2]
        layer_data.plot(x='quantization_mode', y='cosine_similarity', kind='bar', ax=ax, color='lightgreen', legend=False)
        ax.set_title('Cosine Similarity (Higher = Better)', fontweight='bold')
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('Cosine Similarity')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # SNR (higher is better)
        ax = axes[3]
        layer_data.plot(x='quantization_mode', y='snr_db', kind='bar', ax=ax, color='skyblue', legend=False)
        ax.set_title('Signal-to-Noise Ratio (Higher = Better)', fontweight='bold')
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('SNR (dB)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Quality Metrics - {layer.replace("_", " ").title()} Layer', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'quality_metrics_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Quality visualizations saved to: {output_dir}")


def plot_distribution_comparison(stats_df: pd.DataFrame, output_dir: Path):
    """Create distribution comparison visualizations."""
    
    setup_plot_style()
    
    # Plot distribution statistics for each layer
    for layer in stats_df['layer'].unique():
        layer_data = stats_df[stats_df['layer'] == layer]
        
        # Separate standard and quantized
        standard_data = layer_data[layer_data['type'] == 'standard_stats']
        quantized_data = layer_data[layer_data['type'] == 'quantized_stats']
        
        # Plot mean and std comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean comparison
        ax = axes[0, 0]
        x = np.arange(len(quantized_data))
        width = 0.35
        ax.bar(x - width/2, standard_data['mean'].values, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, quantized_data['mean'].values, width, label='Quantized', alpha=0.8)
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('Mean')
        ax.set_title('Mean Value Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(quantized_data['quantization_mode'].values, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Std comparison
        ax = axes[0, 1]
        ax.bar(x - width/2, standard_data['std'].values, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, quantized_data['std'].values, width, label='Quantized', alpha=0.8)
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Std Deviation Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(quantized_data['quantization_mode'].values, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Range comparison (min-max)
        ax = axes[1, 0]
        for i, (idx, row) in enumerate(quantized_data.iterrows()):
            std_row = standard_data.iloc[i]
            ax.plot([i, i], [std_row['min'], std_row['max']], 'o-', label='Standard' if i == 0 else '', 
                   color='C0', linewidth=2, markersize=8)
            ax.plot([i, i], [row['min'], row['max']], 's-', label='Quantized' if i == 0 else '', 
                   color='C1', linewidth=2, markersize=8)
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('Value Range')
        ax.set_title('Min-Max Range Comparison')
        ax.set_xticks(range(len(quantized_data)))
        ax.set_xticklabels(quantized_data['quantization_mode'].values, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Skewness and Kurtosis
        ax = axes[1, 1]
        x = np.arange(len(quantized_data))
        width = 0.2
        ax.bar(x - 1.5*width, standard_data['skewness'].values, width, label='Std Skewness', alpha=0.8)
        ax.bar(x - 0.5*width, quantized_data['skewness'].values, width, label='Quant Skewness', alpha=0.8)
        ax.bar(x + 0.5*width, standard_data['kurtosis'].values, width, label='Std Kurtosis', alpha=0.8)
        ax.bar(x + 1.5*width, quantized_data['kurtosis'].values, width, label='Quant Kurtosis', alpha=0.8)
        ax.set_xlabel('Quantization Mode')
        ax.set_ylabel('Value')
        ax.set_title('Skewness and Kurtosis Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(quantized_data['quantization_mode'].values, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution Statistics - {layer.replace("_", " ").title()} Layer', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_stats_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Distribution visualizations saved to: {output_dir}")


def create_visualizations(results_dir: str, output_dir: str, 
                         include_quality: bool = True):
    """
    Main function to create all visualizations.
    
    Args:
        results_dir: Directory containing analysis results
        output_dir: Directory to save visualizations
        include_quality: Whether to include quality evaluation plots
    """
    
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load performance data
    perf_files = list(results_dir.glob("benchmark_*.csv"))
    if perf_files:
        print("\nCreating performance visualizations...")
        dfs = [pd.read_csv(f) for f in perf_files]
        df = pd.concat(dfs, ignore_index=True)
        plot_performance_metrics(df, output_dir)
    
    # Load and plot quality metrics if available
    if include_quality:
        quality_file = results_dir / 'quality_metrics.csv'
        stats_file = results_dir / 'distribution_statistics.csv'
        
        if quality_file.exists():
            print("\nCreating quality metric visualizations...")
            quality_df = pd.read_csv(quality_file)
            plot_quality_metrics(quality_df, output_dir)
        
        if stats_file.exists():
            print("\nCreating distribution comparison visualizations...")
            stats_df = pd.read_csv(stats_file)
            plot_distribution_comparison(stats_df, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
