#!/usr/bin/env python3
"""
Complete Quantization Analysis Script

Runs both performance analysis and quality evaluation, then creates
comprehensive visualizations.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantization_benchmark.analysis import analyze_performance
from quantization_benchmark.quality_eval import evaluate_quantization_quality
from quantization_benchmark.visualization import create_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="Complete quantization benchmark analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing benchmark CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/analysis",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality evaluation (only do performance analysis)"
    )
    parser.add_argument(
        "--test-sequences-file",
        type=str,
        default=None,
        help="File containing test sequences for quality evaluation (one per line)"
    )
    parser.add_argument(
        "--num-test-sequences",
        type=int,
        default=100,
        help="Number of test sequences to use for quality evaluation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DOEJGI/GenomeOcean-100M",
        help="Model name for quality evaluation"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUANTIZATION BENCHMARK - COMPLETE ANALYSIS")
    print("=" * 70)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip quality eval: {args.skip_quality}")
    
    # Step 1: Performance Analysis
    print("\n" + "=" * 70)
    print("STEP 1: PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    try:
        perf_results = analyze_performance(args.results_dir, args.output_dir)
        print("\n[OK] Performance analysis completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: Quality Evaluation (optional)
    if not args.skip_quality:
        print("\n" + "=" * 70)
        print("STEP 2: QUALITY EVALUATION")
        print("=" * 70)
        
        # Load test sequences
        if args.test_sequences_file and Path(args.test_sequences_file).exists():
            with open(args.test_sequences_file, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
            test_sequences = test_sequences[:args.num_test_sequences]
            print(f"Loaded {len(test_sequences)} test sequences from file")
        else:
            # Use sequences from benchmark data
            print("Loading test sequences from benchmark data...")
            import pandas as pd
            csv_files = list(Path(args.results_dir).glob("*.csv"))
            if csv_files:
                # Try to find the original dataset
                dataset_file = Path("dataset/arc53_2000_seq_50k.csv")
                if dataset_file.exists():
                    df = pd.read_csv(dataset_file)
                    if 'seq' in df.columns:
                        test_sequences = df['seq'].head(args.num_test_sequences).tolist()
                        print(f"Loaded {len(test_sequences)} sequences from dataset")
                    else:
                        print("[WARNING] No 'seq' column in dataset, skipping quality evaluation")
                        test_sequences = None
                else:
                    print("[WARNING] Dataset file not found, skipping quality evaluation")
                    test_sequences = None
            else:
                print("[WARNING] No benchmark data found, skipping quality evaluation")
                test_sequences = None
        
        if test_sequences:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16
                
                # Get quantization modes from performance results
                quant_modes = perf_results['dataframe']['quantization'].unique().tolist()
                
                print(f"Evaluating quantization modes: {quant_modes}")
                print(f"Using {len(test_sequences)} test sequences")
                print(f"Device: {device}")
                
                from model_to_benchmark import load_model, get_max_length
                from transformers import AutoConfig, AutoTokenizer
                
                # Get max_len from model config
                config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
                max_len = get_max_length(config, tokenizer)
                
                print(f"Using max_length: {max_len}")
                
                quality_results = evaluate_quantization_quality(
                    model_loader_func=load_model,
                    quantization_modes=quant_modes,
                    test_sequences=test_sequences,
                    model_name=args.model_name,
                    device=device,
                    dtype=dtype,
                    output_dir=args.output_dir,
                    max_len=max_len
                )
                
                print("\n[OK] Quality evaluation completed successfully")
                
            except Exception as e:
                print(f"\n[WARNING] Quality evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with visualization...")
    else:
        print("\n[SKIP] Quality evaluation skipped")
    
    # Step 3: Create Visualizations
    print("\n" + "=" * 70)
    print("STEP 3: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    try:
        create_visualizations(
            results_dir=args.output_dir,
            output_dir=args.output_dir,
            include_quality=not args.skip_quality
        )
        print("\n[OK] Visualizations created successfully")
    except Exception as e:
        print(f"\n[ERROR] Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  Performance Analysis:")
    print("    - quant_summary.csv (main performance metrics)")
    print("    - memory_analysis.csv")
    print("    - throughput_comparison.csv")
    print("    - energy_analysis.csv")
    print("    - best_performers.csv")
    print("    - performance_report.txt")
    
    if not args.skip_quality:
        print("\n  Quality Evaluation:")
        print("    - quality_metrics.csv (KL divergence, cosine similarity, etc.)")
        print("    - distribution_statistics.csv")
    
    print("\n  Visualizations:")
    print("    - throughput_comparison.png")
    print("    - memory_comparison.png")
    print("    - latency_comparison.png")
    print("    - energy_efficiency.png")
    print("    - performance_memory_tradeoff.png")
    
    if not args.skip_quality:
        print("    - divergence_comparison.png")
        print("    - cosine_similarity.png")
        print("    - snr_comparison.png")
        print("    - quality_heatmap_*.png")
        print("    - distribution_stats_*.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
