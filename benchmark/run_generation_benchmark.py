"""
run_generation_benchmark.py

Benchmark for evaluating GENOME generation capabilities (Perplexity & NLL).
Evaluates genomes using sliding-window chunking via vLLM for high performance.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import AutoTokenizer

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.metrics import EnergyMeter
from core.model_loader import STANDARD_MODE

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("gen_bench_vllm")

def chunk_genome(input_ids, context_len, stride):
    seq_len = len(input_ids)
    chunks = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break
            
        chunk_ids = input_ids[begin_loc:end_loc]
        chunks.append((chunk_ids, trg_len))
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    return chunks

def compute_metrics_vllm(llm, genome_chunks):
    """
    Computes Perplexity (NLL) by evaluating sliding window chunks via vLLM.
    genome_chunks: list of chunks arrays per genome.
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1, 
        prompt_logprobs=1,
        ignore_eos=True
    )
    
    # Flatten chunks
    all_chunks = []
    chunk_meta = [] # (genome_idx, chunk_idx, trg_len)
    
    for gen_idx, chunks in enumerate(genome_chunks):
        for c_idx, (chunk_ids, trg_len) in enumerate(chunks):
            all_chunks.append(chunk_ids)
            chunk_meta.append((gen_idx, c_idx, trg_len))
            
    if len(all_chunks) == 0:
        return []
            
    log.info(f"Submitting {len(all_chunks)} sequence chunks to vLLM engine for scoring...")
    outputs = llm.generate(prompt_token_ids=all_chunks, sampling_params=sampling_params, use_tqdm=True)
    
    genome_nll = {i: 0.0 for i in range(len(genome_chunks))}
    genome_acc = {i: 0 for i in range(len(genome_chunks))}
    genome_tokens = {i: 0 for i in range(len(genome_chunks))}
    
    for output, meta in zip(outputs, chunk_meta):
        gen_idx, c_idx, trg_len = meta
        prompt_logprobs = output.prompt_logprobs[1:] 
        actual_tokens_to_score = output.prompt_token_ids[1:]
        
        if trg_len > 0:
            target_tokens = actual_tokens_to_score[-trg_len:]
            target_logprobs = prompt_logprobs[-trg_len:]
            
            for token_id, logprob_dict in zip(target_tokens, target_logprobs):
                if logprob_dict is None or len(logprob_dict) == 0:
                    continue
                    
                # Perplexity / NLL tracking
                target_logprob = logprob_dict.get(token_id, None)
                if target_logprob is not None:
                    genome_nll[gen_idx] += -target_logprob.logprob
                
                # Accuracy tracking: The top-1 predicted token has the highest logprob in the dict.
                # vLLM's prompt_logprobs=1 inherently includes the top-1 prediction alongside the prompt token.
                highest_prob_token = max(logprob_dict.items(), key=lambda x: x[1].logprob)[0]
                if highest_prob_token == token_id:
                    genome_acc[gen_idx] += 1
                    
                genome_tokens[gen_idx] += 1
                    
    results = []
    
    for gen_idx in range(len(genome_chunks)):
        total_tokens = genome_tokens[gen_idx]
        nll_sum = genome_nll[gen_idx]
        acc_sum = genome_acc[gen_idx]
        
        avg_nll = nll_sum / total_tokens if total_tokens > 0 else 0.0
        accuracy = acc_sum / total_tokens if total_tokens > 0 else 0.0
        
        results.append({
            'perplexity': math.exp(avg_nll) if avg_nll < 50 else float('inf'),
            'neg_log_likelihood': avg_nll,
            'accuracy': accuracy, 
            'total_tokens': total_tokens
        })
        
    return results

def compute_throughput_vllm(llm, prompt_ids, max_new_tokens, batch_size):
    """
    Computes purely raw generation throughput (tokens/sec) and handles Energy logging.
    prompt_ids: list of tokenized genome prompt prefixes.
    """
    from vllm import SamplingParams
    
    # Pure generation, NO logprobs overhead to maximize speed
    sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=max_new_tokens,
        ignore_eos=True
    )
    
    prompts_to_run = prompt_ids
    if len(prompts_to_run) < batch_size:
        # Replicate to fill batch size to properly stress test the GPU
        multiplier = math.ceil(batch_size / max(1, len(prompts_to_run)))
        prompts_to_run = (prompts_to_run * multiplier)[:batch_size]
    
    log.info(f"Warming up vLLM generation engine...")
    _ = llm.generate(prompt_token_ids=[prompts_to_run[0]], sampling_params=sampling_params, use_tqdm=False)

    log.info(f"Running Phase B (Efficiency/Throughput) on {len(prompts_to_run)} parallel sequences...")
    
    import time
    gpu_index = 0
    start_time = time.perf_counter()
    
    with EnergyMeter(gpu_index=gpu_index) as em:
        _ = llm.generate(prompt_token_ids=prompts_to_run, sampling_params=sampling_params, use_tqdm=True)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    total_generated_tokens = len(prompts_to_run) * max_new_tokens
    tps = total_generated_tokens / duration
    
    avg_power = 0.0
    energy_kwh = 0.0
    tokens_per_watt = 0.0
    
    if em.kwh is not None:
        energy_kwh = em.kwh
        avg_power = (energy_kwh * 3_600_000.0) / duration # Joules / sec = Watts
        if avg_power > 0:
            tokens_per_watt = tps / avg_power
            
    return {
        "duration_s": round(duration, 3),
        "total_generated_tokens": total_generated_tokens,
        "throughput_tps": round(tps, 2),
        "avg_power_W": round(avg_power, 2),
        "energy_kWh": round(energy_kwh, 6),
        "tokens_per_watt": round(tokens_per_watt, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean Generation Quality Benchmark (vLLM Accelerated)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--outdir", type=str, default="./results_gen", help="Base output dir")
    parser.add_argument("--quant-modes", type=str, nargs="+", default=["bf16", "fp8"], help="Modes to benchmark (e.g. bf16 fp8)")
    parser.add_argument("--n-genomes", type=int, default=50, help="Num genomes to eval (default 50)")
    parser.add_argument("--tokens-per-genome", type=int, default=102400, help="Target tokens per genome (~500k bp)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (currently fixed to cuda for vLLM)")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision format")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for concurrent generation")
    parser.add_argument("--context-len", type=int, default=1024, help="Max context length window")
    parser.add_argument("--stride", type=int, default=256, help="Window stride length")
    parser.add_argument("--gen-len", type=int, default=1024, help="Number of new tokens to generate in Phase B")

    args = parser.parse_args()
    
    # We allow vLLM to automatically select the best attention backend (FA3 or XFormers)
    # based on the hardware and the specific FP8/head_dim constraints.
    
    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_vllm_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    unique_genomes = df["genome_id"].unique()
    if len(unique_genomes) > args.n_genomes:
        selected_genome_ids = np.random.choice(unique_genomes, size=args.n_genomes, replace=False)
    else:
        selected_genome_ids = unique_genomes
        
    log.info(f"Selected {len(selected_genome_ids)} genomes for evaluation.")
    
    genome_data = []
    
    log.info("Preparing genome sequences...")
    for gid in tqdm(selected_genome_ids, desc="Preparing Data"):
        g_rows = df[df["genome_id"] == gid]["seq"].tolist()
        
        full_seq = ""
        for frag in g_rows:
            full_seq += frag
            if len(full_seq) > (args.tokens_per_genome * 6): 
                break
        
        genome_data.append((gid, full_seq))
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    log.info(f"Tokenizer: {type(tokenizer).__name__}")
    
    all_input_ids = []
    ordered_gids = []
    
    log.info("Tokenizing all genomes...")
    for gid, seq_text in tqdm(genome_data, desc="Tokenizing"):
        encodings = tokenizer(seq_text, return_tensors="pt", max_length=args.tokens_per_genome, truncation=True)
        input_ids = encodings.input_ids[0].tolist()
        if len(input_ids) < args.context_len:
            log.warning(f"Genome {gid} is shorter than context_len ({len(input_ids)} < {args.context_len}). Skipping.")
            continue
        all_input_ids.append(input_ids)
        ordered_gids.append(gid)

    # Convert all genomes into their sliding window chunks right now
    genome_chunks_list = []
    for input_ids in all_input_ids:
        chunks = chunk_genome(input_ids, args.context_len, args.stride)
        genome_chunks_list.append(chunks)
        
    # We find the max chunk length just to set max_model_len
    max_dataset_len = args.context_len
    
    all_quality_results = []
    all_eff_results = []
    
    modes = args.quant_modes
    # Resolve 'standard' to 'bf16'
    modes = [STANDARD_MODE if m == "standard" else m for m in modes]
    
    for mode in modes:
        log.info(f"\n=== Evaluating Mode: {mode} ===")
        log.info("Booting vLLM Engine...")
        
        from vllm import LLM
        vllm_kwargs = {
            "model": args.model,
            "trust_remote_code": True,
            "dtype": args.precision,
            "tensor_parallel_size": 1,
            "max_num_seqs": args.batch_size, 
            "max_model_len": max_dataset_len + args.gen_len + 128,
            "enforce_eager": True
        }
        
        if mode == "fp8":
            vllm_kwargs["kv_cache_dtype"] = "fp8"
            log.info("Mode 'fp8' detected: Assigning fp8 KV Cache to vLLM.")
            
        try:
            llm = LLM(**vllm_kwargs)
        except Exception as e:
            log.error(f"Failed to load vLLM engine for mode {mode}: {e}")
            continue
        
        log.info(f"=== Phase A: Quality Extraction ({mode}) ===")
        import torch
        if args.device == "cuda": torch.cuda.synchronize()
        base_vram = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else 0
        
        results_list = compute_metrics_vllm(llm, genome_chunks_list)
        
        for gid, res in zip(ordered_gids, results_list):
            all_quality_results.append({
                "genome_id": gid,
                "quantization": mode,
                "model": args.model,
                "perplexity": res['perplexity'],
                "neg_log_likelihood": res['neg_log_likelihood'],
                "accuracy": res['accuracy'], 
                "total_tokens": res['total_tokens']
            })
            
        log.info(f"=== Phase B: Pure Throughput & Efficiency ({mode}) ===")
        # Build prompt prefixes for generation
        prompt_prefixes = []
        for ids in all_input_ids:
            prompt_prefixes.append(ids[:args.context_len])
            
        eff_stats = compute_throughput_vllm(llm, prompt_prefixes, args.gen_len, args.batch_size)
        
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0
        
        eff_stats["quantization"] = mode
        eff_stats["model"] = args.model
        eff_stats["batch_size"] = args.batch_size
        eff_stats["static_vram_GB"] = round(base_vram, 2)
        eff_stats["peak_vram_GB"] = round(peak_vram, 2)
        
        all_eff_results.append(eff_stats)
            
        del llm
        torch.cuda.empty_cache()

    if all_quality_results:
        df_qual = pd.DataFrame(all_quality_results)
        save_path_qual = outdir / "generation_quality_metrics.csv"
        df_qual.to_csv(save_path_qual, index=False)
        log.info(f"Detailed quality results saved to {save_path_qual}")
        
    if all_eff_results:
        df_eff = pd.DataFrame(all_eff_results)
        save_path_eff = outdir / "generation_efficiency_metrics.csv"
        df_eff.to_csv(save_path_eff, index=False)
        log.info(f"Detailed efficency results saved to {save_path_eff}")
        
    if all_quality_results and all_eff_results:
        summary_lines = ["=== Unified Generation Benchmark Summary ===\n"]
        
        for mode in modes:
            subset_q = df_qual[df_qual["quantization"] == mode]
            subset_e = df_eff[df_eff["quantization"] == mode]
            if len(subset_q) == 0: continue
            
            mean_ppl = subset_q["perplexity"].mean()
            std_ppl = subset_q["perplexity"].std()
            
            mean_nll = subset_q["neg_log_likelihood"].mean()
            std_nll = subset_q["neg_log_likelihood"].std()
            
            eff_row = subset_e.iloc[0] if not subset_e.empty else {}
            
            line = (f"\nMode: {mode}\n"
                    f"  [Quality - N={len(subset_q)} genomes]\n"
                    f"  Perplexity: {mean_ppl:.4f} ± {std_ppl:.4f}\n"
                    f"  NLL:        {mean_nll:.4f} ± {std_nll:.4f}\n"
                    f"  [Efficiency - Batch: {args.batch_size}]\n"
                    f"  Throughput: {eff_row.get('throughput_tps', 0)} tokens/sec\n"
                    f"  Power:      {eff_row.get('avg_power_W', 0)} W\n"
                    f"  Efficiency: {eff_row.get('tokens_per_watt', 0)} tokens/watt\n"
                    f"  VRAM Peak:  {eff_row.get('peak_vram_GB', 0)} GB\n")
            summary_lines.append(line)
            print(line)
                
        summary_path = outdir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.writelines(summary_lines)
        log.info(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()
