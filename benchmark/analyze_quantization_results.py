#!/usr/bin/env python3
"""
Analysis script for quantization benchmark results.
Generates summary statistics and comparisons across quantization modes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_benchmark_results(results_dir):
    """Load all benchmark CSV files from results directory."""
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob("benchmark_*.csv"))
    
    if not csv_files:
        raise ValueError(f"No benchmark CSV files found in {results_dir}")
    
    # Load and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} benchmark results from {len(csv_files)} files")
    
    return combined_df

def analyze_quantization_performance(df, output_dir):
    """Analyze performance across quantization modes."""
    print("\n=== Quantization Performance Analysis ===")
    
    # Group by quantization mode
    quant_summary = df.groupby('quantization').agg({
        'seqs_per_s': ['mean', 'std'],
        'tokens_per_s': ['mean', 'std'],
        'peak_vram_GB': ['mean', 'std'],
        'avg_power_W': ['mean', 'std'],
        'tokens_per_watt': ['mean', 'std'],
        'E2EL_ms': ['mean', 'std']
    }).round(3)
    
    print("\nPerformance Summary by Quantization Mode:")
    print(quant_summary)
    
    # Save the quant_summary dataframe
    output_dir = Path(output_dir)
    quant_summary_file = output_dir / 'quant_summary.csv'
    quant_summary.to_csv(quant_summary_file)
    print(f"\nQuant summary saved to: {quant_summary_file}")
    
    return quant_summary

def analyze_memory_efficiency(df):
    """Analyze memory usage across quantization modes."""
    print("\n=== Memory Efficiency Analysis ===")
    
    # Memory usage by quantization mode
    memory_stats = df.groupby('quantization')['peak_vram_GB'].agg(['mean', 'min', 'max', 'std']).round(3)
    print("\nMemory Usage (GB) by Quantization Mode:")
    print(memory_stats)
    
    # Memory reduction compared to standard
    if 'standard' in df['quantization'].values:
        standard_memory = df[df['quantization'] == 'standard']['peak_vram_GB'].mean()
        memory_reduction = {}
        
        for quant_mode in df['quantization'].unique():
            if quant_mode != 'standard':
                mode_memory = df[df['quantization'] == quant_mode]['peak_vram_GB'].mean()
                reduction = (standard_memory - mode_memory) / standard_memory * 100
                memory_reduction[quant_mode] = reduction
        
        print(f"\nMemory Reduction vs Standard (baseline: {standard_memory:.2f} GB):")
        for mode, reduction in memory_reduction.items():
            print(f"{mode:20s}: {reduction:+6.1f}%")
    
    return memory_stats

def analyze_energy_efficiency(df):
    """Analyze energy efficiency across quantization modes."""
    print("\n=== Energy Efficiency Analysis ===")
    
    # Filter out rows with missing energy data
    energy_df = df.dropna(subset=['tokens_per_watt', 'avg_power_W'])
    
    if len(energy_df) == 0:
        print("No energy data available for analysis")
        return None
    
    energy_stats = energy_df.groupby('quantization').agg({
        'tokens_per_watt': ['mean', 'std'],
        'avg_power_W': ['mean', 'std']
    }).round(3)
    
    print("\nEnergy Efficiency by Quantization Mode:")
    print(energy_stats)
    
    return energy_stats

def create_performance_plots(df, output_dir):
    """Create visualization plots for quantization performance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Throughput comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sequences per second
    sns.boxplot(data=df, x='quantization', y='seqs_per_s', ax=ax1)
    ax1.set_title('Sequences per Second by Quantization Mode')
    ax1.set_xlabel('Quantization Mode')
    ax1.set_ylabel('Sequences/sec')
    ax1.tick_params(axis='x', rotation=45)
    
    # Tokens per second
    sns.boxplot(data=df, x='quantization', y='tokens_per_s', ax=ax2)
    ax2.set_title('Tokens per Second by Quantization Mode')
    ax2.set_xlabel('Quantization Mode')
    ax2.set_ylabel('Tokens/sec')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory usage comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='quantization', y='peak_vram_GB')
    plt.title('Peak VRAM Usage by Quantization Mode')
    plt.xlabel('Quantization Mode')
    plt.ylabel('Peak VRAM (GB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Energy efficiency (if data available)
    energy_df = df.dropna(subset=['tokens_per_watt'])
    if len(energy_df) > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=energy_df, x='quantization', y='tokens_per_watt')
        plt.title('Energy Efficiency by Quantization Mode')
        plt.xlabel('Quantization Mode')
        plt.ylabel('Tokens per Watt')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Performance vs Memory tradeoff
    plt.figure(figsize=(10, 8))
    for quant_mode in df['quantization'].unique():
        mode_data = df[df['quantization'] == quant_mode]
        plt.scatter(mode_data['peak_vram_GB'], mode_data['tokens_per_s'], 
                   label=quant_mode, alpha=0.7, s=60)
    
    plt.xlabel('Peak VRAM (GB)')
    plt.ylabel('Tokens per Second')
    plt.title('Performance vs Memory Usage Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_memory_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")

def save_quantization_summary(df, output_dir):
    """Save detailed quantization summary as CSV and JSON."""
    output_dir = Path(output_dir)
    
    # Create detailed summary by quantization mode
    summary_stats = df.groupby('quantization').agg({
        'tokens_per_s': ['mean', 'std', 'min', 'max'],
        'seqs_per_s': ['mean', 'std', 'min', 'max'],
        'peak_vram_GB': ['mean', 'std', 'min', 'max'],
        'E2EL_ms': ['mean', 'std', 'min', 'max'],
        'avg_power_W': ['mean', 'std', 'min', 'max'],
        'tokens_per_watt': ['mean', 'std', 'min', 'max'],
        'batch_size': ['min', 'max']  # Show batch size range
    }).round(3)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    
    # Save as CSV
    csv_file = output_dir / 'quantization_summary.csv'
    summary_stats.to_csv(csv_file)
    
    # Save as JSON for easy programmatic access
    json_file = output_dir / 'quantization_summary.json'
    summary_stats.to_json(json_file, indent=2)
    
    # Create a simplified comparison table
    comparison_df = df.groupby('quantization').agg({
        'tokens_per_s': 'mean',
        'peak_vram_GB': 'mean',
        'tokens_per_watt': 'mean',
        'E2EL_ms': 'mean'
    }).round(2)
    
    # Add memory reduction vs standard
    if 'standard' in comparison_df.index:
        standard_memory = comparison_df.loc['standard', 'peak_vram_GB']
        comparison_df['memory_reduction_pct'] = ((standard_memory - comparison_df['peak_vram_GB']) / standard_memory * 100).round(1)
    
    # Add throughput ratio vs standard
    if 'standard' in comparison_df.index:
        standard_throughput = comparison_df.loc['standard', 'tokens_per_s']
        comparison_df['throughput_ratio'] = (comparison_df['tokens_per_s'] / standard_throughput).round(3)
    
    comparison_file = output_dir / 'quantization_comparison.csv'
    comparison_df.to_csv(comparison_file)
    
    print(f"\nQuantization summary saved to:")
    print(f"  - Detailed: {csv_file}")
    print(f"  - JSON: {json_file}")
    print(f"  - Comparison: {comparison_file}")
    
    return summary_stats, comparison_df

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report."""
    output_dir = Path(output_dir)
    report_file = output_dir / 'quantization_benchmark_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("QUANTIZATION BENCHMARK ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total benchmark runs: {len(df)}\n")
        f.write(f"Quantization modes tested: {', '.join(sorted(df['quantization'].unique()))}\n")
        f.write(f"Models tested: {', '.join(df['model'].unique())}\n")
        f.write(f"Precision modes: {', '.join(df['precision'].unique())}\n")
        f.write(f"Batch sizes tested: {', '.join(map(str, sorted(df['batch_size'].unique())))}\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY BY QUANTIZATION MODE\n")
        f.write("-" * 40 + "\n")
        perf_summary = df.groupby('quantization').agg({
            'tokens_per_s': 'mean',
            'seqs_per_s': 'mean',
            'peak_vram_GB': 'mean',
            'E2EL_ms': 'mean',
            'tokens_per_watt': 'mean'
        }).round(2)
        f.write(perf_summary.to_string())
        f.write("\n\n")
        
        # Memory efficiency analysis
        if 'standard' in df['quantization'].values:
            f.write("MEMORY EFFICIENCY vs STANDARD\n")
            f.write("-" * 30 + "\n")
            standard_memory = df[df['quantization'] == 'standard']['peak_vram_GB'].mean()
            f.write(f"Standard baseline: {standard_memory:.2f} GB\n\n")
            
            for quant_mode in sorted(df['quantization'].unique()):
                if quant_mode != 'standard':
                    mode_memory = df[df['quantization'] == quant_mode]['peak_vram_GB'].mean()
                    reduction = (standard_memory - mode_memory) / standard_memory * 100
                    f.write(f"{quant_mode:15s}: {mode_memory:6.2f} GB ({reduction:+5.1f}%)\n")
            f.write("\n")
        
        # Best performers
        f.write("BEST PERFORMERS\n")
        f.write("-" * 15 + "\n")
        
        if 'tokens_per_s' in df.columns:
            best_throughput = df.loc[df['tokens_per_s'].idxmax()]
            f.write(f"Highest throughput: {best_throughput['quantization']} ")
            f.write(f"({best_throughput['tokens_per_s']:.0f} tokens/s, BS={best_throughput['batch_size']})\n")
        
        if 'peak_vram_GB' in df.columns:
            best_memory = df.loc[df['peak_vram_GB'].idxmin()]
            f.write(f"Lowest memory usage: {best_memory['quantization']} ")
            f.write(f"({best_memory['peak_vram_GB']:.2f} GB, BS={best_memory['batch_size']})\n")
        
        if 'E2EL_ms' in df.columns:
            best_latency = df.loc[df['E2EL_ms'].idxmin()]
            f.write(f"Lowest latency: {best_latency['quantization']} ")
            f.write(f"({best_latency['E2EL_ms']:.1f} ms, BS={best_latency['batch_size']})\n")
        
        if 'tokens_per_watt' in df.columns and not df['tokens_per_watt'].isna().all():
            best_efficiency = df.loc[df['tokens_per_watt'].idxmax()]
            f.write(f"Most energy efficient: {best_efficiency['quantization']} ")
            f.write(f"({best_efficiency['tokens_per_watt']:.2f} tokens/watt, BS={best_efficiency['batch_size']})\n")
    
    print(f"\nSummary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze quantization benchmark results")
    parser.add_argument("--results-dir", type=str, default="./results", 
                       help="Directory containing benchmark CSV files")
    parser.add_argument("--output-dir", type=str, default="./analysis", 
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    # Load results
    try:
        df = load_benchmark_results(args.results_dir)
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analyses
    quant_summary = analyze_quantization_performance(df, args.output_dir)
    analyze_memory_efficiency(df)
    analyze_energy_efficiency(df)
    
    # Save quantization summary
    save_quantization_summary(df, args.output_dir)
    
    # Create visualizations
    create_performance_plots(df, args.output_dir)
    
    # Generate summary report
    generate_summary_report(df, args.output_dir)
    
    print(f"\n[OK] Analysis complete! Results saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())