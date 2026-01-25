"""
run_generation_benchmark.py

Benchmark for evaluating GENOME generation capabilities (Perplexity & Accuracy).
Evaluates 50 genomes independently to verify biological syntax.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model_loader import load_model, get_max_length
from generation_benchmark.metrics import compute_metrics_sliding_window
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("gen_bench")

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean Generation Quality Benchmark")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision")
    parser.add_argument("--outdir", type=str, default="./results_gen", help="Base output dir")
    parser.add_argument("--quant-modes", type=str, nargs="+", default=["standard", "8bit"], help="Modes")
    parser.add_argument("--context-len", type=int, default=None, help="Max context length (None = auto-detect)")
    parser.add_argument("--stride", type=int, default=None, help="Stride (None = context // 2)")
    parser.add_argument("--n-genomes", type=int, default=50, help="Num genomes to eval (default 50)")
    parser.add_argument("--tokens-per-genome", type=int, default=102400, help="Target tokens per genome (~500k bp)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--loader", type=str, default="native", choices=["native", "bitsandbytes"], help="Model Loader Type")
    parser.add_argument("--verbose", action="store_true", help="Show detailed sliding window progress")
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load & Prepare Data
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Select Genomes
    unique_genomes = df["genome_id"].unique()
    if len(unique_genomes) > args.n_genomes:
        selected_genome_ids = np.random.choice(unique_genomes, size=args.n_genomes, replace=False)
    else:
        selected_genome_ids = unique_genomes
        
    log.info(f"Selected {len(selected_genome_ids)} genomes for independent evaluation.")
    
    # Pre-process: Concatenate fragments per genome to reach target token count
    genome_data = [] # List of (genome_id, full_sequence_text)
    
    log.info("Preparing genome sequences...")
    for gid in tqdm(selected_genome_ids, desc="Preparing Data"):
        g_rows = df[df["genome_id"] == gid]["seq"].tolist()
        
        full_seq = ""
        for frag in g_rows:
            full_seq += frag
            if len(full_seq) > (args.tokens_per_genome * 6): 
                break
        
        genome_data.append((gid, full_seq))
        
    # 2. Tokenize & Benchmark Loop
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    log.info(f"Tokenizer: {type(tokenizer).__name__}")
    log.info(f"Vocab Size: {tokenizer.vocab_size}")
    
    modes = args.quant_modes 
    # Logic: If mode is 'fp8' or 'bf16', treat as native. If '8bit' or '4bit*', treat as bitsandbytes.
    # The --loader arg sets the *default*, but mixed usage in one list requires dynamic switching.
    
    results = []
    
    dtype = getattr(torch, args.precision)
    
    for mode in modes:
        log.info(f"\n=== Evaluating Mode: {mode} ===")
        os.environ["QUANT_MODE"] = mode
        
        # Determine Loader for this specific mode
        if mode in ["8bit", "4bit_nf4", "4bit_fp4"]:
            current_loader = "bitsandbytes"
        else:
            current_loader = "native" # bf16, fp16, fp8
            
        log.info(f"Loader: {current_loader}")
        
        try:
            model, _, _ = load_model(
                args.model, args.device, dtype, model_type="causal", loader_type=current_loader
            )
            
            # Setup stride/context for this model
            if args.context_len is None:
                ctx = getattr(model.config, "max_position_embeddings", 2048)
                if not isinstance(ctx, int) or ctx > 100000: ctx = 2048
            else:
                ctx = args.context_len
            
            stride = args.stride if args.stride is not None else ctx // 2
            log.info(f"Ctx: {ctx}, Stride: {stride}")
            
            # Iterate Genomes
            genome_pbar = tqdm(enumerate(genome_data), total=len(genome_data), desc=f"Eval {mode}")
            for i, (gid, seq_text) in genome_pbar:
                genome_pbar.set_description(f"Eval {mode} | Genome: {gid} ({i+1}/{len(genome_data)})")
                
                # Tokenize this genome's sequence
                # Truncate to target length
                encodings = tokenizer(seq_text, return_tensors="pt", max_length=args.tokens_per_genome, truncation=True)
                input_ids = encodings.input_ids
                
                if input_ids.size(1) < ctx:
                    log.warning(f"Genome {gid} is too short ({input_ids.size(1)} tokens). Skipping.")
                    continue
                
                # Compute Metrics (Single Pass)
                metrics = compute_metrics_sliding_window(
                    model, input_ids, stride=stride, context_len=ctx, 
                    verbose=args.verbose, device=args.device
                )
                
                results.append({
                    "genome_id": gid,
                    "quantization": mode,
                    "loader": current_loader,
                    "model": args.model,
                    "perplexity": metrics['perplexity'],
                    "neg_log_likelihood": metrics['neg_log_likelihood'],
                    "accuracy": metrics['accuracy'],
                    "total_tokens": metrics['total_tokens']
                })
                
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            log.error(f"Failed mode {mode}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Report Results
    if results:
        # Save detailed CSV
        df_res = pd.DataFrame(results)
        save_path = outdir / "generation_metrics_by_genome.csv"
        df_res.to_csv(save_path, index=False)
        log.info(f"Detailed results saved to {save_path}")
        
        # Calculate Summary Stats
        summary_lines = ["=== Benchmark Summary ===\n"]
        for mode in modes:
            subset = df_res[df_res["quantization"] == mode]
            if len(subset) == 0: continue
            
            mean_ppl = subset["perplexity"].mean()
            std_ppl = subset["perplexity"].std()
            
            mean_nll = subset["neg_log_likelihood"].mean()
            std_nll = subset["neg_log_likelihood"].std()
            
            mean_acc = subset["accuracy"].mean()
            std_acc = subset["accuracy"].std()
            
            line = (f"Mode: {mode}\n"
                    f"  Perplexity: {mean_ppl:.4f} ± {std_ppl:.4f}\n"
                    f"  NLL:        {mean_nll:.4f} ± {std_nll:.4f}\n"
                    f"  Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}\n"
                    f"  (N={len(subset)} genomes)\n")
            summary_lines.append(line)
            print(line)
            
        # Save Summary TXT
        summary_path = outdir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.writelines(summary_lines)
        log.info(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()
