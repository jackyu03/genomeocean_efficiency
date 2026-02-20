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
from tqdm import tqdm

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantization_benchmark.quality_eval import evaluate_quantization_quality, extract_layer_outputs, compare_quantization_outputs, save_quality_results
from binning_benchmark.eval import run_binning_eval
from core.model_loader import load_model, get_max_length, STANDARD_MODE
from core.metrics import EnergyMeter
from transformers import AutoTokenizer
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("go_full_bench")

def batch_process_and_measure(model, input_ids_all, attention_mask_all, device, batch_size=8):
    """
    Runs inference to extract embeddings while measuring performance (Speed + Energy).
    """
    model.eval()
    
    # 2. Setup Metrics
    total_samples = input_ids_all.size(0)
    total_tokens = 0
    total_seqs = 0
    all_embeddings = {'last': [], 'second_last': []}
    
    # Energy Meter
    gpu_index = 0
    base_vram = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        base_vram = torch.cuda.memory_allocated() / 1e9

    with torch.no_grad():
        dummy_ids = input_ids_all[:1].to(device)
        _ = model(dummy_ids)
        del dummy_ids

    start_time = time.perf_counter()
    
    with EnergyMeter(gpu_index=gpu_index) as em:
        with torch.no_grad():
            for i in tqdm(range(0, total_samples, batch_size), desc="Inference", unit="batch"):
                # Slice batch
                batch_input_ids = input_ids_all[i : i + batch_size].to(device)
                batch_attention_mask = attention_mask_all[i : i + batch_size].to(device)
                
                inputs = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask
                }
                
                if total_seqs == 0: 
                     log.info(f"Draft Input shape (Batch 0): {inputs['input_ids'].shape} (B, L)")
                
                # Count tokens (non-padded entities)
                n_toks = batch_attention_mask.sum().item()
                total_tokens += n_toks
                total_seqs += batch_input_ids.size(0)
                
                # Forward Pass
                outputs = model(**inputs, output_hidden_states=True)
                
                # Mean Pool Hidden States
                mask = batch_attention_mask.unsqueeze(-1) # (B, L, 1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

                for layer_idx, layer_name in [(-1, 'last'), (-2, 'second_last')]:
                    hs = outputs.hidden_states[layer_idx].float() # Cast to float32 to prevent overflow
                    sum_hs = torch.sum(hs * mask, dim=1)
                    pooled = sum_hs / sum_mask # (B, D)
                    
                    # Sanitize to prevent Infinity/NaN propagation
                    pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    all_embeddings[layer_name].append(pooled.cpu().numpy())
                
                del inputs, outputs, hs, pooled
    
    peak_vram = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 3. Calculate Metrics
    embeddings = {
        k: np.concatenate(v, axis=0) for k, v in all_embeddings.items()
    }
    
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
            
    perf_metrics = {
        "duration_s": round(duration, 2),
        "total_seqs": total_seqs,
        "total_tokens": total_tokens,
        "seqs_per_s": round(seqs_per_s, 2),
        "tokens_per_s": round(tokens_per_s, 2),
        "avg_power_W": round(avg_power, 2),
        "energy_kWh": round(energy_kwh, 6),
        "tokens_per_watt": round(tokens_per_watt, 2),
        "static_vram_GB": round(base_vram, 2),
        "peak_vram_GB": round(peak_vram, 2)
    }
    
    return embeddings, perf_metrics

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean Full Integrated Benchmark")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision")
    parser.add_argument("--quant-modes", type=str, nargs="+", default=[STANDARD_MODE, "8bit", "4bit_nf4"], help="Modes")
    parser.add_argument("--outdir", type=str, default="./results_full", help="Base output dir")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max length in TOKENS (not base pairs)")
    parser.add_argument("--n-genomes", type=int, default=None, help="Number of genomes to use (randomly sampled)")
    parser.add_argument("--n-fragments", type=int, default=None, help="Number of fragments per genome (randomly sampled)")
    parser.add_argument("--subdivide-fragments", type=int, default=None, help="Length to subdivide long fragments into (bp). Discards remainder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    
    # Binning Params
    parser.add_argument("--umap-dim", type=int, default=10)
    parser.add_argument("--loader", type=str, default="native", choices=["native", "bitsandbytes"], help="Model Loader Type")
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

    # 1.1 Optional Subdivision
    if args.subdivide_fragments is not None:
        chunk_size = args.subdivide_fragments
        log.info(f"Subdividing fragments into chunks of {chunk_size} bp (discarding remainder)...")
        
        new_rows = []
        for idx, row in df.iterrows():
            seq = row['seq']
            gid = row['genome_id']
            seq_len = len(seq)
            
            # Calculate number of full chunks
            num_chunks = seq_len // chunk_size
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                sub_seq = seq[start:end]
                new_rows.append({"genome_id": gid, "seq": sub_seq})
        
        original_count = len(df)
        df = pd.DataFrame(new_rows)
        log.info(f"Subdivision complete. Rows: {original_count} -> {len(df)}")

    
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
        log.info(f"Dataset shape: {df.shape}")
        avg_len_chars = df["seq"].str.len().mean()
        log.info(f"Average sequence length (bps): {avg_len_chars:.2f}")

    sequences = df["seq"].tolist()
    genome_ids = df["genome_id"].tolist()
    
    # Generate Global Color Map for Consistent Plotting
    unique_genomes_sorted = sorted(list(set(genome_ids)))
    palette = sns.color_palette("husl", len(unique_genomes_sorted))
    global_color_map = dict(zip(unique_genomes_sorted, palette))
    log.info(f"Generated global color map for {len(unique_genomes_sorted)} genomes.")
    
    dtype = getattr(torch, args.precision)
    
    # Storage for results
    perf_results = []
    binning_results = []
    
    # Need to store STANDARD embeddings for quality comparison
    standard_embeddings = {}
    quality_comparisons = []
    
    # Ensure 'standard' runs first if present so we have baseline
    # Ensure STANDARD_MODE runs first if present so we have baseline
    # Also resolve any 'standard' alias in input args
    modes = []
    for m in args.quant_modes:
        if m == "standard":
            modes.append(STANDARD_MODE)
        else:
            modes.append(m)
    
    # Remove duplicates while preserving order
    modes = list(dict.fromkeys(modes))
    
    if STANDARD_MODE in modes and modes[0] != STANDARD_MODE:
        modes.remove(STANDARD_MODE)
        modes.insert(0, STANDARD_MODE)
        
    # 1.5 Pre-tokenize Data
    log.info("Pre-tokenizing entire dataset to eliminate CPU bottlenecks...")
    pre_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if pre_tokenizer.pad_token is None:
        if pre_tokenizer.eos_token is not None:
            pre_tokenizer.pad_token = pre_tokenizer.eos_token
        else:
            pre_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
    # Tokenize all sequences at once (or could do in large chunks if needed)
    all_inputs = pre_tokenizer(
        sequences, 
        padding=True, 
        truncation=True, 
        max_length=args.max_tokens, 
        return_tensors="pt"
    )
    
    input_ids_all = all_inputs["input_ids"]
    attention_mask_all = all_inputs["attention_mask"]
    log.info(f"Dataset Tokenized. Shape: {input_ids_all.shape}")
        
    # 2. Main Loop Over Modes
    for mode in modes:
        log.info(f"\n=== Processing Mode: {mode} ===")
        os.environ["QUANT_MODE"] = mode
        
        # Determine Loader for this specific mode
        if mode in ["8bit", "4bit_nf4", "4bit_fp4", "4bit_nf4_double"]:
            current_loader = "bitsandbytes"
        else:
            current_loader = "native" # bf16, fp16, fp8
            
        log.info(f"Loader: {current_loader}")
        
        try:
            # A. Load Model
            model, tokenizer, config = load_model(args.model, args.device, dtype, loader_type=current_loader)
            
            # B. Run Inference (Perf + Embeddings)
            log.info(f"Running Inference & Performance Measurement (BS={args.batch_size})...")
            embeddings, perf = batch_process_and_measure(
                model, input_ids_all, attention_mask_all, args.device, args.batch_size
            )
            
            # Save Perf
            perf["quantization"] = mode
            perf["model"] = args.model
            perf["batch_size"] = args.batch_size
            perf_results.append(perf)
            
            # Log Perf
            log.info(f"Speed: {perf['tokens_per_s']} tok/s | Power: {perf['avg_power_W']} W | Eff: {perf['tokens_per_watt']} tok/W")
            log.info(f"Static vram: {perf['static_vram_GB']} GB | Peak vram: {perf['peak_vram_GB']} GB")

            del model
            torch.cuda.empty_cache()

            # C. Quality Eval (Compare to Standard)
            if mode == STANDARD_MODE:
                standard_embeddings = embeddings
                log.info(f"Stored {STANDARD_MODE} embeddings for reference.")
            elif standard_embeddings:
                log.info(f"Computing Quality Metrics vs {STANDARD_MODE}...")
                # We reuse the logic from quality_eval but adapted for raw numpy arrays
                
                # Wrap in dict format expected by helper
                std_dict = {k: torch.tensor(v) for k, v in standard_embeddings.items()}
                quant_dict = {k: torch.tensor(v) for k, v in embeddings.items()}
                
                comp = compare_quantization_outputs(std_dict, quant_dict, mode)
                quality_comparisons.append(comp)
            else:
                log.warning(f"Skipping Quality Eval (No '{STANDARD_MODE}' baseline found yet).")
            
            # D. Binning Eval
            log.info("Running Binning Evaluation...")
            mode_plot_dir = outdir / "plots" / mode
            mode_plot_dir.mkdir(parents=True, exist_ok=True)
            
            bin_metrics = run_binning_eval(
                embeddings['last'], 
                genome_ids,
                method="dbscan",
                output_dir=str(mode_plot_dir),
                umap_dim=args.umap_dim,
                eps=args.dbscan_eps,
                min_samples=args.dbscan_min_samples,
                model_name=args.model,
                quant_mode=mode,
                color_map=global_color_map
            )
            bin_metrics["quantization"] = mode
            binning_results.append(bin_metrics)
            
        except Exception as e:
            log.error(f"Failed mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            
    # 3. Save All Results
    pd.DataFrame(perf_results).to_csv(outdir / "performance_metrics.csv", index=False)
    pd.DataFrame(binning_results).to_csv(outdir / "binning_metrics.csv", index=False)
    if quality_comparisons:
        quality_dir = outdir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        save_quality_results(quality_comparisons, quality_dir)
        
    log.info(f"\nAll Done! Results saved to {outdir}")

if __name__ == "__main__":
    main()
