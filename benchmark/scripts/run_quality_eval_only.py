#!/usr/bin/env python3
"""
Standalone Quality Evaluation Script

Runs only the quality evaluation without performance benchmarking.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantization_benchmark.quality_eval import evaluate_quantization_quality
from quantization_benchmark.visualization import plot_quality_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run quality evaluation for quantized models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DOEJGI/GenomeOcean-100M",
        help="Model name to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/quality_eval",
        help="Directory to save quality evaluation results"
    )
    parser.add_argument(
        "--test-sequences-file",
        type=str,
        default=None,
        help="File containing test sequences (one per line)"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=100,
        help="Number of test sequences to use"
    )
    parser.add_argument(
        "--quantization-modes",
        type=str,
        nargs="+",
        default=["standard", "int8", "4bit_int4", "4bit_nf4", "4bit_fp4", "4bit_nf4_double"],
        help="Quantization modes to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: use model's max_position_embeddings)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUANTIZATION QUALITY EVALUATION")
    print("=" * 70)
    print(f"\nModel: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantization modes: {args.quantization_modes}")
    print(f"Device: {args.device}")
    print(f"Number of sequences: {args.num_sequences}")
    
    # Load test sequences
    if args.test_sequences_file and Path(args.test_sequences_file).exists():
        print(f"\nLoading sequences from: {args.test_sequences_file}")
        with open(args.test_sequences_file, 'r') as f:
            test_sequences = [line.strip() for line in f if line.strip()]
        test_sequences = test_sequences[:args.num_sequences]
        print(f"Loaded {len(test_sequences)} test sequences")
    else:
        # Try to load from default dataset
        dataset_file = Path("dataset/arc53_2000_seq_50k.csv")
        if dataset_file.exists():
            print(f"\nLoading sequences from: {dataset_file}")
            import pandas as pd
            df = pd.read_csv(dataset_file)
            if 'seq' in df.columns:
                test_sequences = df['seq'].head(args.num_sequences).tolist()
                print(f"Loaded {len(test_sequences)} sequences from dataset")
            else:
                print("[ERROR] No 'seq' column in dataset file")
                return 1
        else:
            print(f"[ERROR] No test sequences provided and default dataset not found")
            print(f"Please provide --test-sequences-file or ensure {dataset_file} exists")
            return 1
    
    # Setup
    import torch
    from model_to_benchmark import load_model, get_max_length
    from transformers import AutoConfig, AutoTokenizer
    
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    dtype = torch.float16
    
    # Get max_len from model config if not specified
    if args.max_length is None:
        print("\nLoading model config to determine max_length...")
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        max_len = get_max_length(config, tokenizer)
    else:
        max_len = args.max_length
    
    print(f"Using max_length: {max_len}")
    
    # Run quality evaluation
    try:
        quality_results = evaluate_quantization_quality(
            model_loader_func=load_model,
            quantization_modes=args.quantization_modes,
            test_sequences=test_sequences,
            model_name=args.model_name,
            device=device,
            dtype=dtype,
            output_dir=args.output_dir,
            max_len=max_len
        )
        
        print("\n[OK] Quality evaluation completed successfully")
        
        # Create visualizations
        print("\nCreating visualizations...")
        output_dir = Path(args.output_dir)
        
        # Check if quality metrics file exists
        quality_csv = output_dir / "quality_metrics.csv"
        if quality_csv.exists():
            plot_quality_metrics(str(output_dir), str(output_dir))
            print("[OK] Visualizations created")
        
    except Exception as e:
        print(f"\n[ERROR] Quality evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("QUALITY EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - quality_metrics.csv (KL divergence, cosine similarity, etc.)")
    print("  - distribution_statistics.csv")
    print("  - divergence_comparison.png")
    print("  - cosine_similarity.png")
    print("  - snr_comparison.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
