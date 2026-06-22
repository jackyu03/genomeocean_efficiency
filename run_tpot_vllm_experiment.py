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

async def run_vllm_experiment(engine, prompt_ids, gen_len, batch_size):
    """
    Run vLLM generation and measure latency for every single token yielded.
    """
    sampling_params = SamplingParams(
        max_tokens=gen_len,
        temperature=0.0,
        ignore_eos=True
    )
    
    context_len = len(prompt_ids[0])
    log.info(f"Starting vLLM generation: Batch Size = {batch_size}, Prefix = {context_len}, Gen = {gen_len}")
    
    # We must submit all requests to the engine
    generators = []
    for i in range(batch_size):
        request_id = f"req_{i}"
        # vLLM takes prompt_token_ids
        gen = engine.generate({"prompt_token_ids": prompt_ids[i]}, sampling_params, request_id)
        generators.append(gen)

    # To accurately measure TPOT for a batch, we need to advance all generators concurrently
    # However, for rigorous TPOT tracking vs Context Length, a single sequence is the cleanest.
    # If batch_size > 1, the engine will batch them internally if they arrive together.
    
    step_records = []
    
    if batch_size == 1:
        gen = generators[0]
        step = 0
        
        # Start timing for the very first token (Prefill phase)
        last_time = time.perf_counter()
        
        async for request_output in gen:
            current_time = time.perf_counter()
            latency_ms = (current_time - last_time) * 1000.0
            
            # RequestOutput contains the full generated text/tokens so far
            # The number of output tokens is len(request_output.outputs[0].token_ids)
            output_tokens_count = len(request_output.outputs[0].token_ids)
            current_ctx_len = context_len + output_tokens_count
            
            if step == 0:
                prefill_latency = latency_ms
                log.info(f"Prefill Latency (Context={context_len}): {prefill_latency:.2f} ms")
            
            if step >= 1: # Record decoding steps
                step_records.append({
                    "step": step,
                    "effective_context_length": current_ctx_len,
                    "latency_ms": latency_ms
                })
                
            last_time = current_time
            step += 1
            
            if step % 500 == 0:
                log.info(f"Generated {step} tokens... Current TPOT: {latency_ms:.2f} ms")
                
        return step_records, prefill_latency
    else:
        log.error("Batch sizes > 1 with async streaming requires parallel event loops. For this experiment, please use BS=1.")
        return [], 0.0

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean TPOT vs Context Bench (vLLM)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--outdir", type=str, default="./results_tpot_vllm", help="Base output dir")
    parser.add_argument("--prompt-len", type=int, default=625, help="Seed prefix length")
    parser.add_argument("--gen-len", type=int, default=2000, help="Number of new tokens to generate")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision: bf16 or fp8")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_vllm_{timestamp}"
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
        
    prompt_ids = [input_ids] # Batch size 1
    
    # Setup vLLM Engine
    kv_cache_dtype = "fp8" if args.precision == "fp8" else "auto"
    log.info(f"Initializing vLLM AsyncEngine. Model: {args.model}, KV-Cache: {kv_cache_dtype}")
    
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=True, # Disable CUDA graphs to get true step-by-step latency without capture overhead
        max_model_len=args.prompt_len + args.gen_len + 128
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Run
    log.info(f"Starting experiment: Precision = {args.precision} (KV Cache = {kv_cache_dtype})")
    
    loop = asyncio.get_event_loop()
    records, prefill_latency = loop.run_until_complete(
        run_vllm_experiment(engine, prompt_ids, args.gen_len, batch_size=1)
    )
    
    # Save Results
    if records:
        df_results = pd.DataFrame(records)
        df_results["batch_size"] = 1
        df_results["precision"] = args.precision
        df_results["model"] = args.model
        
        save_path = outdir / "tpot_context_latency.csv"
        df_results.to_csv(save_path, index=False)
        log.info(f"TPOT vs Context results saved to {save_path}")
        
        # Save overarching summary to file
        import json
        summary_info = {
            "model": args.model,
            "precision": args.precision,
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
        plt.plot(df_results['effective_context_length'], df_results['latency_ms'], color='tab:red', label='Latency')
        plt.xlabel('Effective Context Length (Tokens)')
        plt.ylabel('Latency Per Token (ms)')
        plt.title(f'vLLM TPOT vs Context Length\n(Model: {args.model}, KV-Cache: {kv_cache_dtype})')
        plt.tight_layout()
        plt.savefig(outdir / "tpot_vs_context_vllm.png", dpi=300)
        
if __name__ == "__main__":
    main()
