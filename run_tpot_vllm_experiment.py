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
    last_time = time.perf_counter()
    async for request_output in gen:
        current_time = time.perf_counter()
        latency_ms = (current_time - last_time) * 1000.0
        
        output_tokens_count = len(request_output.outputs[0].token_ids)
        current_ctx_len = context_len + output_tokens_count
        
        step_records.append({
            "step": step,
            "effective_context_length": current_ctx_len,
            "latency_ms": latency_ms
        })
            
        last_time = current_time
        step += 1
    return step_records

async def run_vllm_experiment(engine, prompt_ids_list, gen_len, batch_size, profile=False):
    sampling_params = SamplingParams(max_tokens=gen_len, temperature=0.0, ignore_eos=True)
    context_len = len(prompt_ids_list[0])
    log.info(f"Starting vLLM generation: Batch Size = {batch_size}, Prefix = {context_len}, Gen = {gen_len}")
    
    generators = []
    for i in range(batch_size):
        gen = engine.generate({"prompt_token_ids": prompt_ids_list[i]}, sampling_params, f"req_{i}")
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
        prof.export_chrome_trace("results_tpot_vllm/kernel_trace_vllm.json")
        with open("results_tpot_vllm/profiler_summary_vllm.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    else:
        all_records = await asyncio.gather(*tasks)
        
    # Average the records across the batch
    if not all_records or not all_records[0]:
        return [], 0.0
        
    avg_records = []
    num_steps = len(all_records[0])
    for s in range(num_steps):
        avg_lat = sum(r[s]["latency_ms"] for r in all_records) / batch_size
        avg_records.append({
            "step": s,
            "effective_context_length": all_records[0][s]["effective_context_length"],
            "latency_ms": avg_lat
        })
        
    prefill_latency = avg_records[0]["latency_ms"]
    log.info(f"Average Prefill Latency (Context={context_len}): {prefill_latency:.2f} ms")
    return avg_records, prefill_latency

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean TPOT vs Context Bench (vLLM)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--outdir", type=str, default="./results_tpot_vllm", help="Base output dir")
    parser.add_argument("--prompt-len", type=int, default=1024, help="Seed prefix length")
    parser.add_argument("--gen-len", type=int, default=9216, help="Number of new tokens to generate")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision: bf16 or fp8")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
    
    # Run
    log.info(f"Starting experiment: Precision = {args.precision} (KV Cache = {kv_cache_dtype})")
    
    loop = asyncio.get_event_loop()
    records, prefill_latency = loop.run_until_complete(
        run_vllm_experiment(engine, prompt_ids_list, args.gen_len, args.batch_size, args.profile)
    )
    
    # Save Results
    if records:
        df_results = pd.DataFrame(records)
        df_results["batch_size"] = args.batch_size
        df_results["precision"] = args.precision
        df_results["model"] = args.model
        
        save_path = outdir / "tpot_context_latency.csv"
        df_results.to_csv(save_path, index=False)
        log.info(f"TPOT vs Context results saved to {save_path}")
        
        import json
        summary_info = {
            "model": args.model,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "kv_cache_dtype": kv_cache_dtype,
            "seed_prompt_length": args.prompt_len,
            "generated_tokens": args.gen_len,
            "prefill_latency_ms": prefill_latency,
            "avg_tpot_latency_ms": df_results["latency_ms"].mean()
        }
        with open(outdir / "experiment_summary.json", "w") as f:
            json.dump(summary_info, f, indent=4)
            
        # Plot
        log.info("Generating plot...")
        plt.figure(figsize=(10, 6))
        
        plot_df = df_results[df_results['step'] > 1]
        
        plt.plot(plot_df['effective_context_length'], plot_df['latency_ms'], color='tab:red', label='Latency')
        plt.xlabel('Effective Context Length (Tokens)')
        plt.ylabel('Latency Per Token (ms)')
        plt.title(f'vLLM TPOT vs Context Length\n(Model: {args.model}, Batch: {args.batch_size}, KV: {kv_cache_dtype})')
        plt.tight_layout()
        plt.savefig(outdir / "tpot_vs_context_vllm.png", dpi=300)
        
if __name__ == "__main__":
    main()
