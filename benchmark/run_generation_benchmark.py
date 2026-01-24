"""
run_generation_benchmark.py

Benchmark for evaluating GENOME generation capabilities (Perplexity & Accuracy).
Focuses on how quantization affects the model's ability to predict the next base.
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

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model_loader import load_model, get_max_length
from generation_benchmark.metrics import compute_perplexity, compute_accuracy
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
    parser.add_argument("--stride", type=int, default=512, help="Stride for PPL calculation")
    parser.add_argument("--n-genomes", type=int, default=None, help="Num genomes to sample")
    parser.add_argument("--n-fragments", type=int, default=None, help="Num fragments per genome")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Subsampling Logic
    if args.n_genomes is not None:
        unique_genomes = df["genome_id"].unique()
        if len(unique_genomes) > args.n_genomes:
            selected = np.random.choice(unique_genomes, size=args.n_genomes, replace=False)
            df = df[df["genome_id"].isin(selected)]
            log.info(f"Subsampled to {args.n_genomes} genomes.")
            
    if args.n_fragments is not None:
        log.info(f"Subsampling to {args.n_fragments} fragments per genome...")
        df = df.groupby("genome_id", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), args.n_fragments), random_state=args.seed)
        )
        
    sequences = df["seq"].tolist()
    log.info(f"Total sequences: {len(sequences)}")
    
    # 2. Tokenize Data
    log.info("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # For Perplexity: We typically concat everything into one long stream (or process seqs independently)
    # Merging all sequences with a separator could be good, or just simply processing list.
    # To follow standard LM evaluation, we concat them.
    
    # However, these are distinct genomic fragments. Concatenating them might introduce artifacts at boundaries?
    # Actually, PPL logic usually handles long text.
    # Let's simple concatenate with an EOS token if available, or just space.
    # For now, let's concat with nothing, assuming they are just DNA streams.
    full_text = "".join(sequences)
    
    encodings = tokenizer(full_text, return_tensors="pt")
    # For Accuracy, we might want the individual batches to handle padding properly?
    # Or just use the same sliding window on the full stream for accuracy too.
    # Let's reuse the encodings.
    
    input_ids_stream = encodings.input_ids
    log.info(f"Total Tokens in Stream: {input_ids_stream.numel()}")
    
    # 3. Benchmark Loop
    # Only standard and 8bit for now
    modes = ["standard", "8bit"]
    results = []
    
    dtype = getattr(torch, args.precision)
    
    for mode in modes:
        log.info(f"\n=== Evaluating Mode: {mode} ===")
        os.environ["QUANT_MODE"] = mode
        
        try:
            # Load Causal LM
            model, _, _ = load_model(args.model, args.device, dtype, model_type="causal")
            
            # A. Perplexity
            log.info("Calculating Perplexity...")
            ppl, nll = compute_perplexity(model, input_ids_stream, stride=args.stride, device=args.device)
            log.info(f"Perplexity: {ppl:.4f} | NLL: {nll:.4f}")
            
            # B. Accuracy
            # For Accuracy, calculating on the WHOLE stream using the same sliding window logic inside metric would be best,
            # or we can just chop it up. PPL function does sliding window.
            # Let's extract accuracy validation.
            # Since `compute_accuracy` takes a list of tensors (sequences), let's reshape our stream or use individual sequences.
            
            # Better: Evaluate accuracy on the original VALIDATION FRAGMENTS individually to avoid boundary issues.
            # Re-tokenize as list
            log.info("Calculating Next-Token Accuracy (on fragments)...")
            encodings_list = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            acc = compute_accuracy(model, encodings_list["input_ids"], device=args.device, batch_size=4)
            log.info(f"Accuracy: {acc:.4f}")
            
            results.append({
                "quantization": mode,
                "model": args.model,
                "perplexity": ppl,
                "neg_log_likelihood": nll,
                "accuracy_next_token": acc
            })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            log.error(f"Failed mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save
    if results:
        df_res = pd.DataFrame(results)
        save_path = outdir / "generation_metrics.csv"
        df_res.to_csv(save_path, index=False)
        log.info(f"Results saved to {save_path}")
        print(df_res)

if __name__ == "__main__":
    main()
