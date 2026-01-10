#!/usr/bin/env python3
"""
run_full_benchmark.py

Unified runner for GenomeOcean efficiency benchmarks.
Runs both:
1. Quantization Quality Evaluation (KL, Cosine, etc.)
2. Metagenomics Binning Evaluation (DBSCAN Clustering)
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
import json
from pathlib import Path
from datetime import datetime

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantization_benchmark.quality_eval import evaluate_quantization_quality
from binning_benchmark.utils import prepare_binning_dataset
from binning_benchmark.eval import run_binning_eval
from model_to_benchmark import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("go_full_bench")

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean Full Benchmark Suite")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--precision", type=str, default="float16", help="Precision")
    parser.add_argument("--quant-modes", type=str, nargs="+", default=["standard", "8bit", "4bit_nf4"], 
                       help="Quantization modes to test")
    parser.add_argument("--outdir", type=str, default="./results_full", help="Base output directory")
    parser.add_argument("--n-binning-species", type=int, default=50, help="Num species for binning task")
    parser.add_argument("--max-len", type=int, default=5000, help="Max sequence length (bp) for analysis")
    
    args = parser.parse_args()
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Results will be saved to: {outdir}")
    
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # 1. Run Quantization Quality Benchmark
    # This uses the full dataset or a subset? usually a subset for quality to save time
    # We'll randomly select some sequences for quality eval if not too many
    log.info("--- Phase 1: Quantization Quality Evaluation ---")
    
    # Select a subset for quality eval to avoid OOM/huge time
    quality_subset_size = 500
    if len(df) > quality_subset_size:
        quality_seqs = df["seq"].sample(n=quality_subset_size, random_state=42).tolist()
    else:
        quality_seqs = df["seq"].tolist()
        
    dtype = getattr(torch, args.precision)
    
    # We need to wrap load_model to match signature expected by evaluate_quantization_quality if needed?
    # Actually evaluate_quantization_quality calls load_model internally using os.environ['QUANT_MODE']
    # which is a bit hacky but that's how the existing code works (we saw in view_file).
    # wait, quality_eval.py imports load_model from model_to_benchmark.
    
    quality_results = evaluate_quantization_quality(
        model_loader_func=None, # It imports load_model directly inside
        quantization_modes=args.quant_modes,
        test_sequences=quality_seqs,
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        output_dir=str(outdir / "quality"),
        max_len=args.max_len
    )
    
    # 2. Run Binning Benchmark
    log.info("\n--- Phase 2: Binning Evaluation ---")
    
    # Prepare specific dataset for binning (selecting N species)
    binning_df = prepare_binning_dataset(df, n_species=args.n_binning_species)
    binning_seqs = binning_df["seq"].tolist()
    binning_ids = binning_df["genome_id"].tolist()
    
    binning_results = []
    
    for mode in args.quant_modes:
        log.info(f"Running Binning for mode: {mode}")
        os.environ["QUANT_MODE"] = mode
        
        try:
            model, tokenizer, config = load_model(args.model, args.device, dtype)
            model.eval()
            
            # Extract embeddings
            # We can reuse extract_layer_outputs from quality_eval or simple loop
            from quantization_benchmark.quality_eval import extract_layer_outputs
            
            log.info("Extracting embeddings...")
            outputs = extract_layer_outputs(model, tokenizer, binning_seqs, args.device, args.max_len, layers=['last'])
            embeddings = outputs['last'].numpy()
            
            del model
            torch.cuda.empty_cache()
            
            # Run Eval
            metrics = run_binning_eval(embeddings, binning_ids, method="dbscan")
            metrics["quantization"] = mode
            binning_results.append(metrics)
            
        except Exception as e:
            log.error(f"Binning failed for {mode}: {e}")
            
    # Save Binning Results
    binning_results_df = pd.DataFrame(binning_results)
    binning_csv = outdir / "binning_metrics.csv"
    binning_results_df.to_csv(binning_csv, index=False)
    log.info(f"Saved binning metrics to {binning_csv}")
    
    log.info("Full benchmark completed.")

if __name__ == "__main__":
    main()
