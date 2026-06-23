#!/usr/bin/env python3
"""
run_tpot_vllm_experiment.py

Evaluates the per-token decoding latency (TPOT) vs context length 
using vLLM's streaming API. This perfectly matches Protocol B by allowing 
true FP8 KV-Cache memory compression (kv_cache_dtype="fp8").
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time
import asyncio
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("tpot_vllm_bench")

import torch

async def consume_stream(gen, context_len):
    step_records = []
    step = 0
    try:
        async for request_output in gen:
            current_time = time.perf_counter()
            
            output_tokens_count = len(request_output.outputs[0].token_ids)
            current_ctx_len = context_len + output_tokens_count
            
            step_records.append({
                "step": step,
                "effective_context_length": current_ctx_len,
                "timestamp": current_time
            })
                
            step += 1
    except Exception as e:
        log.warning(f"Stream interrupted at step {step} (likely OOM). Partial data retained. Error: {str(e)}")
    return step_records

async def run_vllm_experiment(engine, prompt_ids_list, gen_len, batch_size, outdir, profile=False, rep_idx=1):
    sampling_params = SamplingParams(max_tokens=gen_len, temperature=0.0, ignore_eos=True)
    context_len = len(prompt_ids_list[0])
    log.info(f"Starting vLLM generation: Batch Size = {batch_size}, Prefix = {context_len}, Gen = {gen_len}")
    
    start_time = time.perf_counter()
    generators = []
    for i in range(batch_size):
        gen = engine.generate({"prompt_token_ids": prompt_ids_list[i]}, sampling_params, f"req_rep{rep_idx}_{i}")
        generators.append(gen)
        
    tasks = [consume_stream(g, context_len) for g in generators]
    if profile:
        log.info("Starting Torch Profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            all_records = await asyncio.gather(*tasks)
            
        log.info("Saving profiler traces...")
        prof.export_chrome_trace(str(outdir / f"kernel_trace_vllm_rep{rep_idx}.json"))
        with open(outdir / f"profiler_summary_vllm_rep{rep_idx}.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    else:
        all_records = await asyncio.gather(*tasks)
        
    # Calculate System-Level TPOT (Global Batch Latency)
    if not all_records or not all_records[0]:
        return [], 0.0
        
    sys_records = []
    # Find minimum steps achieved to handle cases where sequences dropped out due to OOM
    num_steps = min(len(r) for r in all_records)
    
    for s in range(num_steps):
        max_ts = max(r[s]["timestamp"] for r in all_records)
        
        if s == 0:
            sys_tpot = (max_ts - start_time) * 1000.0
        else:
            prev_max_ts = max(r[s-1]["timestamp"] for r in all_records)
            sys_tpot = (max_ts - prev_max_ts) * 1000.0
            
        sys_records.append({
            "step": s,
            "effective_context_length": all_records[0][s]["effective_context_length"],
            "latency_ms": sys_tpot,
            "timestamp": max_ts  # pass through to allow writing raw data if needed
        })
        
    prefill_latency = sys_records[0]["latency_ms"]
    log.info(f"Average Prefill Latency (Context={context_len}): {prefill_latency:.2f} ms")
    return sys_records, prefill_latency

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean TPOT vs Context Bench (vLLM)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--outdir", type=str, default="./results_tpot_vllm", help="Base output dir")
    parser.add_argument("--prompt-len", type=int, default=1024, help="Seed prefix length")
    parser.add_argument("--gen-len", type=int, default=9216, help="Number of new tokens to generate")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision: bf16 or fp8")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--repeats", type=int, default=5, help="Number of times to repeat the experiment to average")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"run_{args.precision}_bs{args.batch_size}_{timestamp}"
    if args.profile:
        dir_name += "_profile"
    outdir = Path(args.outdir) / dir_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    unique_genomes = df["genome_id"].unique()
    selected_gid = np.random.choice(unique_genomes, size=1)[0]
    
    g_rows = df[df["genome_id"] == selected_gid]["seq"].tolist()
    full_seq = "".join(g_rows)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encodings = tokenizer(full_seq, return_tensors="pt", max_length=args.prompt_len, truncation=True)
    input_ids = encodings.input_ids[0].tolist()
    
    if len(input_ids) < args.prompt_len:
        input_ids += [tokenizer.eos_token_id or 0] * (args.prompt_len - len(input_ids))
        
    prompt_ids_list = [input_ids for _ in range(args.batch_size)]
    
    # Setup vLLM Engine
    kv_cache_dtype = "fp8" if args.precision == "fp8" else "auto"
    log.info(f"Initializing vLLM AsyncEngine. Model: {args.model}, KV-Cache: {kv_cache_dtype}")
    
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=True, # Allows torch.profiler to see the individual attention kernels
        max_model_len=args.prompt_len + args.gen_len + 128
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    loop = asyncio.get_event_loop()
    all_repeats_dfs = []
    prefill_latencies = []
    
    for rep in range(1, args.repeats + 1):
        log.info(f"--- Starting Repeat {rep}/{args.repeats} ---")
        try:
            records, prefill_latency = loop.run_until_complete(
                run_vllm_experiment(engine, prompt_ids_list, args.gen_len, args.batch_size, outdir, args.profile, rep)
            )
        except Exception as e:
            log.warning(f"Engine failed at repeat {rep} (likely OOM). Stopping repeats. Exception: {str(e)}")
            break
        
        if records:
            prefill_latencies.append(prefill_latency)
            df_rep = pd.DataFrame(records)
            df_rep["batch_size"] = args.batch_size
            df_rep["precision"] = args.precision
            df_rep["model"] = args.model
            df_rep["repeat"] = rep
            
            save_path = outdir / f"tpot_context_latency_rep{rep}.csv"
            df_rep.drop(columns=["timestamp"], errors="ignore").to_csv(save_path, index=False)
            all_repeats_dfs.append(df_rep)
            log.info(f"Saved repeat {rep} to {save_path}")
            
    if not all_repeats_dfs:
        log.error("No data collected across any repeats.")
        return
        
    # Calculate Average
    log.info("Calculating averages across all repeats...")
    df_combined = pd.concat(all_repeats_dfs, ignore_index=True)
    
    # Group by step and effective_context_length to compute the mean latency
    df_avg = df_combined.groupby(['step', 'effective_context_length'], as_index=False)['latency_ms'].mean()
    df_avg["batch_size"] = args.batch_size
    df_avg["precision"] = args.precision
    df_avg["model"] = args.model
    
    avg_save_path = outdir / "tpot_context_latency_avg.csv"
    df_avg.to_csv(avg_save_path, index=False)
    log.info(f"Averaged results saved to {avg_save_path}")
    
    import json
    summary_info = {
        "model": args.model,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "kv_cache_dtype": kv_cache_dtype,
        "seed_prompt_length": args.prompt_len,
        "generated_tokens": args.gen_len,
        "avg_prefill_latency_ms": sum(prefill_latencies)/len(prefill_latencies) if prefill_latencies else 0.0,
        "overall_avg_tpot_latency_ms": df_avg["latency_ms"].mean()
    }
    with open(outdir / "experiment_summary.json", "w") as f:
        json.dump(summary_info, f, indent=4)
        
    # Plot Average
    log.info("Generating averaged plot...")
    plt.figure(figsize=(10, 6))
    
    plot_df = df_avg[df_avg['step'] > 1]
    
    plt.plot(plot_df['effective_context_length'], plot_df['latency_ms'], color='tab:red', label='Average Latency')
    plt.xlabel('Effective Context Length (Tokens)')
    plt.ylabel('Latency Per Token (ms)')
    plt.title(f'vLLM TPOT vs Context Length (Averaged over {args.repeats} runs)\n(Model: {args.model}, Batch: {args.batch_size}, KV: {kv_cache_dtype})')
    plt.tight_layout()
    plt.savefig(outdir / "tpot_vs_context_vllm_avg.png", dpi=300)
        
if __name__ == "__main__":
    main()
