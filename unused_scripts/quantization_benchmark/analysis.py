"""
Performance Analysis Module

Analyzes benchmark results for performance metrics like throughput,
memory usage, latency, and energy efficiency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_benchmark_results(results_dir: str) -> pd.DataFrame:
    """Load all benchmark CSV files from results directory."""
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob("performance_*.csv"))
    
    if not csv_files:
        raise ValueError(f"No benchmark CSV files found in {results_dir}")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} benchmark results from {len(csv_files)} files")
    
    return combined_df


def compute_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance summary statistics by quantization mode."""
    
    # Check if we have multiple samples per quantization mode
    samples_per_mode = df.groupby('quantization').size()
    has_multiple_samples = (samples_per_mode > 1).any()
    
    if has_multiple_samples:
        # Include std when we have multiple samples
        metrics = {
            'seqs_per_s': ['mean', 'std', 'min', 'max'],
            'tokens_per_s': ['mean', 'std', 'min', 'max'],
            "static_vram_GB": ['mean', 'std', 'min', 'max'],
            "peak_vram_GB": ['mean', 'std', 'min', 'max'],
            'E2EL_ms': ['mean', 'std', 'min', 'max'],
            'avg_power_W': ['mean', 'std', 'min', 'max'],
            'tokens_per_watt': ['mean', 'std', 'min', 'max']

        }
    else:
        # Skip std when we only have 1 sample per mode
        metrics = {
            'seqs_per_s': ['mean'],
            'tokens_per_s': ['mean'],
            "static_vram_GB": ['mean'],
            'peak_vram_GB': ['mean'],
            'E2EL_ms': ['mean'],
            'avg_power_W': ['mean'],
            'tokens_per_watt': ['mean']
        }
    
    # Filter metrics that exist in dataframe
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    summary = df.groupby('quantization').agg(available_metrics).round(3)
    
    return summary


# def compute_memory_efficiency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
#     """Analyze memory efficiency across quantization modes."""
    
#     memory_stats = df.groupby('quantization')['peak_vram_GB'].agg(
#         ['mean', 'std', 'min', 'max']
#     ).round(3)
    
#     # Calculate memory reduction vs standard
#     memory_reduction = {}
#     if 'standard' in df['quantization'].values:
#         standard_memory = df[df['quantization'] == 'standard']['peak_vram_GB'].mean()
        
#         for quant_mode in df['quantization'].unique():
#             if quant_mode != 'standard':
#                 mode_memory = df[df['quantization'] == quant_mode]['peak_vram_GB'].mean()
#                 # Negative means reduction, positive means increase
#                 reduction_pct = -((mode_memory - standard_memory) / standard_memory * 100)
#                 memory_reduction[quant_mode] = {
#                     'memory_gb': mode_memory,
#                     'reduction_pct': reduction_pct
#                 }
    
#     return memory_stats, memory_reduction

def compute_memory_efficiency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Analyze both static and peak memory efficiency across quantization modes."""
    
    metrics_to_agg = {
        'static_vram_GB': ['mean', 'std', 'min', 'max'],
        'peak_vram_GB': ['mean', 'std', 'min', 'max']
    }
    
    available_metrics = {k: v for k, v in metrics_to_agg.items() if k in df.columns}
    
    memory_stats = df.groupby('quantization').agg(available_metrics).round(3)
    
    memory_reduction = {}
    if 'standard' in df['quantization'].values:
    
        std_df = df[df['quantization'] == 'standard']
        std_static = std_df['static_vram_GB'].mean() if 'static_vram_GB' in df.columns else None
        std_peak = std_df['peak_vram_GB'].mean()
        
        for quant_mode in df['quantization'].unique():
            if quant_mode == 'standard':
                continue
                
            mode_df = df[df['quantization'] == quant_mode]
            
            mode_peak = mode_df['peak_vram_GB'].mean()
            peak_reduction_pct = -((mode_peak - std_peak) / std_peak * 100)
            
            static_reduction_pct = None
            mode_static = None
            if std_static and 'static_vram_GB' in mode_df.columns:
                mode_static = mode_df['static_vram_GB'].mean()
                static_reduction_pct = -((mode_static - std_static) / std_static * 100)
            
            memory_reduction[quant_mode] = {
                'peak_memory_gb': round(mode_peak, 3),
                'peak_reduction_pct': round(peak_reduction_pct, 2),
                'static_memory_gb': round(mode_static, 3) if mode_static else None,
                'static_reduction_pct': round(static_reduction_pct, 2) if static_reduction_pct else None
            }
    
    return memory_stats, memory_reduction


def compute_throughput_comparison(df: pd.DataFrame) -> Dict:
    """Compare throughput across quantization modes."""
    
    throughput_stats = {}
    
    if 'standard' in df['quantization'].values:
        standard_throughput = df[df['quantization'] == 'standard']['tokens_per_s'].mean()
        
        for quant_mode in df['quantization'].unique():
            mode_throughput = df[df['quantization'] == quant_mode]['tokens_per_s'].mean()
            ratio = mode_throughput / standard_throughput
            
            throughput_stats[quant_mode] = {
                'tokens_per_s': mode_throughput,
                'ratio_vs_standard': ratio,
                'slowdown_pct': (1 - ratio) * 100
            }
    
    return throughput_stats


def compute_energy_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze energy efficiency across quantization modes."""
    
    energy_df = df.dropna(subset=['tokens_per_watt', 'avg_power_W'])
    
    if len(energy_df) == 0:
        return None
    
    energy_stats = energy_df.groupby('quantization').agg({
        'tokens_per_watt': ['mean', 'std', 'min', 'max'],
        'avg_power_W': ['mean', 'std', 'min', 'max'],
        'energy_kWh': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    return energy_stats


def identify_best_performers(df: pd.DataFrame) -> Dict:
    """Identify best performing configurations for each metric."""
    
    best_performers = {}
    
    metrics = {
        'highest_throughput': ('tokens_per_s', 'max'),
        'lowest_peak_memory': ('peak_vram_GB', 'min'),
        'lowest_static_memory': ('static_vram_GB', 'min'),
        'lowest_latency': ('E2EL_ms', 'min'),
        'most_energy_efficient': ('tokens_per_watt', 'max')
    }
    
    for name, (metric, agg_func) in metrics.items():
        if metric in df.columns and not df[metric].isna().all():
            if agg_func == 'max':
                best_row = df.loc[df[metric].idxmax()]
            else:
                best_row = df.loc[df[metric].idxmin()]
            
            best_performers[name] = {
                'quantization': best_row['quantization'],
                'batch_size': best_row['batch_size'],
                'value': best_row[metric]
            }
    
    return best_performers


def save_analysis_results(output_dir: str, 
                          performance_summary: pd.DataFrame,
                          memory_stats: pd.DataFrame,
                          memory_reduction: Dict,
                          throughput_comparison: Dict,
                          energy_stats: pd.DataFrame,
                          best_performers: Dict):
    """Save all analysis results to files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save performance summary (the main quant_summary)
    performance_summary.to_csv(output_dir / 'quant_summary.csv')
    
    # Save memory analysis
    memory_stats.to_csv(output_dir / 'memory_analysis.csv')
    
    # Save throughput comparison
    if throughput_comparison:
        throughput_df = pd.DataFrame(throughput_comparison).T
        throughput_df.to_csv(output_dir / 'throughput_comparison.csv')
    
    # Save energy analysis
    if energy_stats is not None:
        energy_stats.to_csv(output_dir / 'energy_analysis.csv')
    
    # Save best performers summary
    best_df = pd.DataFrame(best_performers).T
    best_df.to_csv(output_dir / 'best_performers.csv')
    
    # Generate text report
    report_file = output_dir / 'performance_report.txt'
    with open(report_file, 'w') as f:
        f.write("QUANTIZATION PERFORMANCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(performance_summary.to_string())
        f.write("\n\n")
        
        f.write("MEMORY EFFICIENCY\n")
        f.write("-" * 40 + "\n")
        f.write(memory_stats.to_string())
        f.write("\n\n")
        
        if memory_reduction:
            f.write("Memory Reduction vs Standard:\n")
            for mode, stats in memory_reduction.items():
                f.write(f"  {mode:15s}: {stats['memory_gb']:.2f} GB ({stats['reduction_pct']:+.1f}%)\n")
            f.write("\n")
        
        f.write("BEST PERFORMERS\n")
        f.write("-" * 40 + "\n")
        for metric, info in best_performers.items():
            f.write(f"{metric}: {info['quantization']} (BS={info['batch_size']}, value={info['value']:.2f})\n")
    
    print(f"\nAnalysis results saved to: {output_dir}")
    print(f"  - quant_summary.csv (main performance summary)")
    print(f"  - memory_analysis.csv")
    print(f"  - throughput_comparison.csv")
    print(f"  - energy_analysis.csv")
    print(f"  - best_performers.csv")
    print(f"  - performance_report.txt")


def analyze_performance(results_dir: str, output_dir: str) -> Dict:
    """
    Main function to analyze quantization benchmark performance.
    
    Args:
        results_dir: Directory containing benchmark CSV files
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing all analysis results
    """
    
    print("\n" + "=" * 60)
    print("QUANTIZATION PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_benchmark_results(results_dir)
    
    # Compute analyses
    print("\nComputing performance metrics...")
    performance_summary = compute_performance_summary(df)
    
    print("Analyzing memory efficiency...")
    memory_stats, memory_reduction = compute_memory_efficiency(df)
    
    print("Comparing throughput...")
    throughput_comparison = compute_throughput_comparison(df)
    
    print("Analyzing energy efficiency...")
    energy_stats = compute_energy_efficiency(df)
    
    print("Identifying best performers...")
    best_performers = identify_best_performers(df)
    
    # Save results
    save_analysis_results(
        output_dir,
        performance_summary,
        memory_stats,
        memory_reduction,
        throughput_comparison,
        energy_stats,
        best_performers
    )
    
    return {
        'dataframe': df,
        'performance_summary': performance_summary,
        'memory_stats': memory_stats,
        'memory_reduction': memory_reduction,
        'throughput_comparison': throughput_comparison,
        'energy_stats': energy_stats,
        'best_performers': best_performers
    }
