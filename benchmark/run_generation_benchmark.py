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
import gc
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.metrics import EnergyMeter
from core.model_loader import STANDARD_MODE

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("gen_bench_vllm")

def cleanup_vllm(llm=None):
    """Forcefully destroys any existing vLLM engine and clears GPU memory."""
    if llm is not None:
        try:
            # vLLM V1 specific shutdown
            if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
            del llm
        except Exception as e:
            log.warning(f"Error during engine shutdown: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
    except:
        pass
    time.sleep(5) # Increased wait for process reap

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
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": chunk} for chunk in all_chunks],
        sampling_params=sampling_params, 
        use_tqdm=True
    )
    
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
    
    # Pure generation, NO logprobs overhead to maximize speed
    sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=max_new_tokens,
        ignore_eos=True
    )
    
    prompts_to_run = prompt_ids
    # We no longer replicate prompts to fill a batch. 
    # This ensures we measure the overhead of UNIQUE biological sequences.
    # vLLM's internal scheduler will process these up to the engine's 'max_num_seqs' limit.

    log.info(f"Warming up vLLM generation engine...")
    _ = llm.generate(
        prompts=[{"prompt_token_ids": prompts_to_run[0]}],
        sampling_params=sampling_params, 
        use_tqdm=False
    )

    log.info(f"Running Phase B (Efficiency/Throughput) on {len(prompts_to_run)} sequences...")
    
    gpu_index = 0
    start_time = time.perf_counter()
    
    # We poll the engine's scheduler during generation to estimate 'actual' memory usage
    # rather than just the reserved pool peak.
    utilizations = []
    
    with EnergyMeter(gpu_index=gpu_index) as em:
        # Note: In production vLLM, we can't easily hook into the sync generate() for polling
        # so we will use the snapshot of what was required.
        _ = llm.generate(
            prompts=[{"prompt_token_ids": p} for p in prompts_to_run],
            sampling_params=sampling_params, 
            use_tqdm=True
        )
        # Post-generation utilization estimate:
        # Since ignored_eos=True, all slots were filled for the duration.
        # Theoretical occupancy: batch_size / capacity
        # We fetch the actual capacity to calculate the 'active' slice.
        engine = llm.llm_engine
        if hasattr(engine, 'model_executor'):
            num_gpu_blocks = engine.model_executor.cache_config.num_gpu_blocks
            # For 128 batch size on 51.8x capacity, utilization is essentially 100% of the pool
            # because the queue stays full and blocks are saturated. 
            utilization = min(1.0, batch_size / (num_gpu_blocks * engine.model_executor.cache_config.block_size / (max_new_tokens + 10240)))
            utilizations.append(utilization)
        else:
            utilizations.append(1.0)
        
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
    cleanup_vllm()
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128], help="Batch sizes for concurrent generation (one per mode or one for all)")
    parser.add_argument("--context-len", type=int, default=1024, help="Max context length window")
    parser.add_argument("--stride", type=int, default=256, help="Window stride length")
    parser.add_argument("--gen-len", type=int, default=1024, help="Number of new tokens to generate in Phase B")
    parser.add_argument("--skip-quality", action="store_true", help="Skip Phase A (Perplexity/Quality) and only run Phase B")
    parser.add_argument("--n-repeats", type=int, default=5, help="Workload multiplier for Phase B (default 5x n-genomes)")

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
    
    # Map batch sizes to modes
    batch_sizes = args.batch_sizes
    if len(batch_sizes) == 1:
        batch_map = {m: batch_sizes[0] for m in modes}
    elif len(batch_sizes) == len(modes):
        batch_map = {m: b for m, b in zip(modes, batch_sizes)}
    else:
        log.error(f"Mismatch between number of modes ({len(modes)}) and batch sizes ({len(batch_sizes)}). Provide either 1 or {len(modes)} batch sizes.")
        sys.exit(1)
    
    for mode in modes:
        mode_batch = batch_map[mode]
        log.info(f"\n=== Evaluating Mode: {mode} (Batch: {mode_batch}) ===")
        log.info("Booting vLLM Engine...")
        
        vllm_kwargs = {
            "model": args.model,
            "trust_remote_code": True,
            "dtype": args.precision,
            "tensor_parallel_size": 1,
            "max_num_seqs": mode_batch, 
            "max_model_len": max_dataset_len + args.gen_len + 128,
            "enforce_eager": False,
            "enable_prefix_caching": False
        }
        
        if mode == "fp8":
            vllm_kwargs["kv_cache_dtype"] = "fp8"
            # In vLLM 0.11+, the dispatcher will automatically select FlashInfer 
            # for FP8 KV caching when FlashAttention-3 is unavailable. 
            log.info("Mode 'fp8' detected: Relying on vLLM dispatcher to route to optimal backend.")
            
        try:
            llm = LLM(**vllm_kwargs)
        except Exception as e:
            log.error(f"Failed to load vLLM engine for mode {mode}: {e}")
            continue
        
        if not args.skip_quality:
            log.info(f"=== Phase A: Quality Extraction ({mode}) ===")
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
        else:
            log.info(f"Skipping Phase A as requested.")
            base_vram = 0.0 # Placeholder
            
        # Phase B: Pure Throughput & Efficiency (Scaled single submission)
        log.info(f"=== Phase B: Efficiency ({mode}) | Workload Scaled x{args.n_repeats} ===")
        # Build prompt prefixes for generation
        prompt_prefixes = []
        for ids in all_input_ids:
            prompt_prefixes.append(ids[:args.context_len])
        
        # Scale the workload as requested (N * Repeats) in one single call
        scaled_prompts = prompt_prefixes * args.n_repeats
            
        eff_stats = compute_throughput_vllm(llm, scaled_prompts, args.gen_len, mode_batch)
        
        # Memory Computation
        engine = llm.llm_engine
        gpu_cache_total_gb = 0.0
        
        model_config = getattr(engine, "model_config", None)
        cache_config = getattr(engine, "cache_config", None)
        if not cache_config and hasattr(engine, "model_executor"):
            cache_config = getattr(engine.model_executor, "cache_config", None)
            
        if cache_config:
            num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", 0)
            block_size_bytes = getattr(cache_config, "cache_block_size_bytes", 0)
            if block_size_bytes > 0:
                gpu_cache_total_gb = (num_gpu_blocks * block_size_bytes) / 1e9
            else:
                # Manual calculation backup
                block_size = getattr(cache_config, "block_size", 16)
                if model_config:
                    # GQA-aware KV token size
                    n_kv_heads = getattr(model_config.hf_config, "num_key_value_heads", model_config.hf_config.num_attention_heads)
                    head_dim = model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads
                    n_layers = model_config.hf_config.num_hidden_layers
                    # 2 (K+V) * layers * heads * dim * bytes
                    bytes_per_tok = 2 * n_layers * n_kv_heads * head_dim * (1 if mode == "fp8" else 2)
                    gpu_cache_total_gb = (num_gpu_blocks * block_size * bytes_per_tok) / 1e9
        
        if model_config:
            n_kv_heads = getattr(model_config.hf_config, "num_key_value_heads", model_config.hf_config.num_attention_heads)
            head_dim = model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads
            n_layers = model_config.hf_config.num_hidden_layers
            bytes_per_tok = (2 * n_layers * n_kv_heads * head_dim * (1 if mode == "fp8" else 2)) / 1e9
            actual_kv_usage = mode_batch * (args.context_len + args.gen_len) * bytes_per_tok
        else:
            actual_kv_usage = 0
            
        actual_vram_gb = base_vram + actual_kv_usage
        peak_vram_gb = base_vram + gpu_cache_total_gb
            
        eff_stats.update({
            "quantization": mode,
            "model": args.model,
            "batch_size": mode_batch,
            "workload_multiplier": args.n_repeats,
            "static_vram_GB": round(base_vram, 2),
            "actual_vram_GB": round(actual_vram_gb, 2),
            "peak_vram_GB": round(peak_vram_gb, 2)
        })
        
        all_eff_results.append(eff_stats)

        cleanup_vllm(llm)

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
            mode_batch = batch_map[mode]
            subset_q = df_qual[df_qual["quantization"] == mode] if not df_qual.empty else pd.DataFrame()
            subset_e = df_eff[df_eff["quantization"] == mode] if not df_eff.empty else pd.DataFrame()
            
            line = f"\nMode: {mode}\n"
            
            if not subset_q.empty:
                mean_ppl = subset_q["perplexity"].mean()
                std_ppl = subset_q["perplexity"].std()
                mean_nll = subset_q["neg_log_likelihood"].mean()
                std_nll = subset_q["neg_log_likelihood"].std()
                line += (f"  [Quality - N={len(subset_q)} genomes]\n"
                        f"  Perplexity: {mean_ppl:.4f} ± {std_ppl:.4f}\n"
                        f"  NLL:        {mean_nll:.4f} ± {std_nll:.4f}\n")
            else:
                line += "  [Quality - SKIPPED]\n"
            
            if not subset_e.empty:
                # Average efficiency metrics across repeats
                mean_tps = subset_e["throughput_tps"].mean()
                std_tps = subset_e["throughput_tps"].std()
                mean_power = subset_e["avg_power_W"].mean()
                mean_eff = subset_e["tokens_per_watt"].mean()
                mean_vram_act = subset_e["actual_vram_GB"].mean()
                mean_vram_pk = subset_e["peak_vram_GB"].mean()
                
                line += (f"  [Efficiency - Batch: {mode_batch}, Repeats: {len(subset_e)}]\n"
                        f"  Throughput: {mean_tps:.2f} tokens/sec {'(±'+str(round(std_tps,2))+')' if len(subset_e)>1 else ''}\n"
                        f"  Power:      {mean_power:.2f} W\n"
                        f"  Efficiency: {mean_eff:.2f} tokens/watt\n"
                        f"  VRAM Actual: {mean_vram_act:.2f} GB\n"
                        f"  VRAM Pool:   {mean_vram_pk:.2f} GB\n")
            
            summary_lines.append(line)
            print(line)
                
        summary_path = outdir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.writelines(summary_lines)
        log.info(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()
