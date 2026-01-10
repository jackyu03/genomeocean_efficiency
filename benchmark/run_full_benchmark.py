"""
run_full_benchmark.py

Unified runner for GenomeOcean efficiency benchmarks.
Runs integrated pipeline for each quantization mode:
1. Performance (Speed/Power) while extracting embeddings.
2. Quality (Cosine/KL) using extracted embeddings.
3. Binning (UMAP/DBSCAN) using extracted embeddings.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
import json
import time
import numpy as np
import threading
from pathlib import Path
from datetime import datetime

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantization_benchmark.quality_eval import evaluate_quantization_quality, extract_layer_outputs, compare_quantization_outputs, save_quality_results
from binning_benchmark.eval import run_binning_eval
from core.model_loader import load_model, get_max_length
from core.metrics import EnergyMeter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("go_full_bench")

def batch_process_and_measure(model, tokenizer, sequences, device, max_len, batch_size=8):
    """
    Runs inference to extract embeddings while measuring performance (Speed + Energy).
    Returns:
        embeddings: np.ndarray (N, hidden_dim)
        perf_metrics: dict (tokens/s, avg_power, etc.)
    """
    model.eval()
    
    # 1. Prepare Batches
    batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
    
    # 2. Setup Metrics
    total_tokens = 0
    total_seqs = 0
    all_embeddings = []
    
    # Energy Meter
    gpu_index = 0
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    
    with EnergyMeter(gpu_index=gpu_index) as em:
        with torch.no_grad():
            for batch in batches:
                # Tokenize
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Count tokens
                n_toks = inputs["input_ids"].numel()
                total_tokens += n_toks
                total_seqs += len(batch)
                
                # Forward Pass
                outputs = model(**inputs, output_hidden_states=True)
                
                # Mean Pool Last Hidden State
                hs = outputs.hidden_states[-1] # (B, L, D)
                mask = inputs["attention_mask"].unsqueeze(-1) # (B, L, 1)
                sum_hs = torch.sum(hs * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = sum_hs / sum_mask # (B, D)
                
                all_embeddings.append(pooled.cpu().numpy())
                
                del inputs, outputs, hs, pooled
                
    if device == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 3. Calculate Metrics
    embeddings = np.concatenate(all_embeddings, axis=0) # (N, D)
    
    seqs_per_s = total_seqs / duration
    tokens_per_s = total_tokens / duration
    
    avg_power = 0.0
    energy_kwh = 0.0
    tokens_per_watt = 0.0
    
    if em.kwh is not None:
        energy_kwh = em.kwh
        avg_power = (energy_kwh * 3_600_000.0) / duration # Joules / sec = Watts
        if avg_power > 0:
            tokens_per_watt = tokens_per_s / avg_power
            
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.max_memory_reserved() / 1e9
        
    perf_metrics = {
        "duration_s": round(duration, 2),
        "total_seqs": total_seqs,
        "total_tokens": total_tokens,
        "seqs_per_s": round(seqs_per_s, 2),
        "tokens_per_s": round(tokens_per_s, 2),
        "avg_power_W": round(avg_power, 2),
        "energy_kWh": round(energy_kwh, 6),
        "tokens_per_watt": round(tokens_per_watt, 2),
        "peak_vram_GB": round(vram_gb, 2)
    }
    
    return embeddings, perf_metrics

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean Full Integrated Benchmark")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="float16", help="Precision")
    parser.add_argument("--quant-modes", type=str, nargs="+", default=["standard", "8bit", "4bit_nf4"], help="Modes")
    parser.add_argument("--outdir", type=str, default="./results_full", help="Base output dir")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max length in TOKENS (not base pairs)")
    parser.add_argument("--n-genomes", type=int, default=None, help="Number of genomes to use (randomly sampled)")
    parser.add_argument("--n-fragments", type=int, default=None, help="Number of fragments per genome (randomly sampled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    # Binning Params
    parser.add_argument("--umap-dim", type=int, default=10)
    parser.add_argument("--dbscan-eps", type=float, default=0.5)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Validate columns
    required_cols = {"genome_id", "fragment_id", "seq"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")

    # Filter by number of genomes if requested
    if args.n_genomes is not None:
        unique_genomes = df["genome_id"].unique()
        if len(unique_genomes) > args.n_genomes:
            # Randomly sample genomes using the seeded generator
            selected_genomes = np.random.choice(unique_genomes, size=args.n_genomes, replace=False)
            df = df[df["genome_id"].isin(selected_genomes)]
            log.info(f"Randomly subsampled to {args.n_genomes} genomes (Seed={args.seed}).")
        else:
            log.info(f"Requested {args.n_genomes} genomes but found {len(unique_genomes)}. Using all.")

    # Filter by number of fragments per genome if requested
    if args.n_fragments is not None:
        # Group by genome_id and sample n_fragments
        log.info(f"Subsampling to {args.n_fragments} fragments per genome...")
        
        def sample_fragments(group):
            if len(group) > args.n_fragments:
                return group.sample(n=args.n_fragments, random_state=args.seed)
            return group

        df = df.groupby("genome_id", group_keys=False).apply(sample_fragments)
        log.info(f"Total sequences after fragment subsampling: {len(df)}")

    sequences = df["seq"].tolist()
    genome_ids = df["genome_id"].tolist()
    
    dtype = getattr(torch, args.precision)
    
    # Storage for results
    perf_results = []
    binning_results = []
    
    # Need to store STANDARD embeddings for quality comparison
    standard_embeddings = None
    quality_comparisons = []
    
    # Ensure 'standard' runs first if present so we have baseline
    modes = args.quant_modes
    if "standard" in modes and modes[0] != "standard":
        modes.remove("standard")
        modes.insert(0, "standard")
        
    # 2. Main Loop Over Modes
    for mode in modes:
        log.info(f"\n=== Processing Mode: {mode} ===")
        os.environ["QUANT_MODE"] = mode
        
        try:
            # A. Load Model
            model, tokenizer, config = load_model(args.model, args.device, dtype)
            
            # B. Run Inference (Perf + Embeddings)
            log.info(f"Running Inference & Performance Measurement (BS={args.batch_size})...")
            embeddings, perf = batch_process_and_measure(
                model, tokenizer, sequences, args.device, args.max_tokens, args.batch_size
            )
            
            # Save Perf
            perf["quantization"] = mode
            perf["model"] = args.model
            perf["batch_size"] = args.batch_size
            perf_results.append(perf)
            
            # Log Perf
            log.info(f"Speed: {perf['tokens_per_s']} tok/s | Power: {perf['avg_power_W']} W | Eff: {perf['tokens_per_watt']} tok/W")

            del model
            torch.cuda.empty_cache()

            # C. Quality Eval (Compare to Standard)
            if mode == "standard":
                standard_embeddings = embeddings
                log.info("Stored standard embeddings for reference.")
            elif standard_embeddings is not None:
                log.info("Computing Quality Metrics vs Standard...")
                # We reuse the logic from quality_eval but adapted for raw numpy arrays
                
                # Wrap in dict format expected by helper
                std_dict = {'last': torch.tensor(standard_embeddings)}
                quant_dict = {'last': torch.tensor(embeddings)}
                
                comp = compare_quantization_outputs(std_dict, quant_dict, mode)
                quality_comparisons.append(comp)
            else:
                log.warning("Skipping Quality Eval (No 'standard' baseline found yet).")
            
            # D. Binning Eval
            log.info("Running Binning Evaluation...")
            mode_plot_dir = outdir / "plots" / mode
            mode_plot_dir.mkdir(parents=True, exist_ok=True)
            
            bin_metrics = run_binning_eval(
                embeddings, 
                genome_ids,
                method="dbscan",
                output_dir=str(mode_plot_dir),
                umap_dim=args.umap_dim,
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples
            )
            bin_metrics["quantization"] = mode
            binning_results.append(bin_metrics)
            
        except Exception as e:
            log.error(f"Failed mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            
    # 3. Save All Results
    # Perf
    pd.DataFrame(perf_results).to_csv(outdir / "performance_metrics.csv", index=False)
    # Binning
    pd.DataFrame(binning_results).to_csv(outdir / "binning_metrics.csv", index=False)
    # Quality
    if quality_comparisons:
        save_quality_results(quality_comparisons, outdir / "quality")
        
    log.info(f"\nAll Done! Results saved to {outdir}")

if __name__ == "__main__":
    main()
