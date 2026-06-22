#!/usr/bin/env python3
"""
run_tpot_context_experiment.py

Evaluates the per-token decoding latency (TPOT) vs context length to measure
the theoretical vs empirical computational overhead of the KV cache.
This isolates the inference speed directly and uses a manual PyTorch generation loop
to precisely track latency at every decoding step.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from assistive_scripts.core.model_loader import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("tpot_context_bench")

def run_experiment(model, tokenizer, prompt_ids, gen_len, profile=False, profile_steps=(100, 110), outdir=None):
    """
    Run autoregressive generation, logging the latency of each individual step.
    prompt_ids: tensor of shape (batch_size, seq_len)
    """
    model.eval()
    
    batch_size, context_len = prompt_ids.shape
    
    # Store latencies per step
    # list of tuples: (step_idx, current_context_len, latency_ms)
    step_records = []
    
    past_key_values = None
    input_ids = prompt_ids
    
    log.info(f"Starting generation: Context prefix = {context_len}, Generation steps = {gen_len}")
    
    # Setup profiler if requested
    prof = None
    if profile:
        log.info(f"Profiling enabled for steps {profile_steps[0]} to {profile_steps[1]}")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False, # True can cause large overhead
        )

    with torch.no_grad():
        for step in tqdm(range(gen_len), desc="Generating Tokens"):
            current_ctx_len = context_len + step
            
            # Start profiling if in range
            if profile and step == profile_steps[0]:
                prof.start()

            # Measure exactly the forward pass and KV cache updating
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Stop profiling if out of range
            if profile and step == profile_steps[1]:
                prof.stop()
            
            latency_ms = (end_time - start_time) * 1000.0
            
            vram_mb = 0.0
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            
            if step == 0:
                prefill_latency = latency_ms
                log.info(f"Prefill Latency (Context={context_len}): {latency_ms:.2f} ms")
            
            # Record latency for ALL steps, but mark prefill explicitly
            if step >= 1: # Start recording from step 1 (first actual decoding step)
                step_records.append({
                    "step": step,
                    "effective_context_length": current_ctx_len,
                    "latency_ms": latency_ms,
                    "vram_mb": vram_mb
                })
            
            # Prepare for next step
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = next_tokens
            
    if profile and prof is not None:
        try:
            prof.stop() # Just in case it didn't stop
        except:
            pass
        
        trace_path = os.path.join(outdir, "kernel_trace.json")
        prof.export_chrome_trace(trace_path)
        log.info(f"Chrome trace exported to {trace_path}")
        
        # Also save the summary to a text file
        prof_summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        print(prof_summary)
        
        prof_summary_path = os.path.join(outdir, "profiler_summary.txt")
        with open(prof_summary_path, "w") as f:
            f.write(prof_summary)
        log.info(f"Profiler summary saved to {prof_summary_path}")

    return step_records, prefill_latency

def main():
    parser = argparse.ArgumentParser(description="GenomeOcean TPOT vs Context Bench")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV (genome_id, seq)")
    parser.add_argument("--model", type=str, required=True, help="HF Model ID")
    parser.add_argument("--outdir", type=str, default="./results_tpot", help="Base output dir")
    parser.add_argument("--n-genomes", type=int, default=1, help="Num genomes to eval (default 1 - we just need a valid prompt)")
    parser.add_argument("--prompt-len", type=int, default=625, help="Seed prefix length")
    parser.add_argument("--gen-len", type=int, default=2000, help="Number of new tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the forward pass")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision (bf16, fp8, etc.)")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler for a narrow window")
    parser.add_argument("--attn-impl", type=str, default=None, help="Attention implementation (e.g. flash_attention_2)")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"run_tpot_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    log.info(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    unique_genomes = df["genome_id"].unique()
    selected_genome_ids = np.random.choice(unique_genomes, size=args.n_genomes, replace=False)
    
    genome_data = []
    for gid in selected_genome_ids:
        g_rows = df[df["genome_id"] == gid]["seq"].tolist()
        full_seq = "".join(g_rows)
        genome_data.append((gid, full_seq))
        
    # Load Model (using custom loader from the repo)
    log.info(f"Loading Model {args.model} with precision {args.precision}")
    # Force quantization via environment variable so the loader picks it up
    os.environ["QUANT_MODE"] = args.precision
    
    # We use AutoModelForCausalLM directly here if we want to enforce attn_impl
    if args.attn_impl:
        log.info(f"Forcing attention implementation: {args.attn_impl}")
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation=args.attn_impl
        )
    else:
        model, tokenizer, config = load_model(
            model_name=args.model,
            device=args.device,
            dtype=torch.bfloat16, # base compute
            model_type="causal",
            loader_type="native"
        )
    
    # Ensure flash attention is used if possible (if using HF transformers >= 4.36)
    # The AutoModel should pick it up automatically if installed, but we can log it.
    log.info(f"Model config _attn_implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
    
    # Tokenize
    all_input_ids = []
    for gid, seq_text in genome_data:
        encodings = tokenizer(seq_text, return_tensors="pt", max_length=args.prompt_len, truncation=True)
        input_ids = encodings.input_ids[0].tolist()
        
        # pad if slightly short
        if len(input_ids) < args.prompt_len:
            pad_len = args.prompt_len - len(input_ids)
            input_ids = input_ids + [tokenizer.eos_token_id or 0] * pad_len
            
        all_input_ids.append(input_ids)

    # We need batch_size sequences. We can just tile the ones we have.
    while len(all_input_ids) < args.batch_size:
        all_input_ids.extend(all_input_ids[:(args.batch_size - len(all_input_ids))])
        
    input_tensor = torch.tensor(all_input_ids[:args.batch_size]).to(args.device)
    
    # Run Experiment
    log.info(f"Starting experiment: Batch Size = {args.batch_size}, Precision = {args.precision}")
    records, prefill_latency = run_experiment(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=input_tensor,
        gen_len=args.gen_len,
        profile=args.profile,
        profile_steps=(args.gen_len // 2, (args.gen_len // 2) + 10), # Profile 10 steps in the middle
        outdir=str(outdir)
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
        
        # Save overarching summary to file
        import json
        summary_info = {
            "model": args.model,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "seed_prompt_length": args.prompt_len,
            "generated_tokens": args.gen_len,
            "prefill_latency_ms": prefill_latency,
            "avg_tpot_latency_ms": df_results["latency_ms"].mean(),
            "peak_vram_mb": df_results["vram_mb"].max()
        }
        with open(outdir / "experiment_summary.json", "w") as f:
            json.dump(summary_info, f, indent=4)
        log.info(f"High-level summary saved to {outdir / 'experiment_summary.json'}")
        
        # Quick summary stats
        log.info("Sample Latencies (First 5):")
        print(df_results.head())
        log.info("Sample Latencies (Last 5):")
        print(df_results.tail())
        
        # Generate Plot
        log.info("Generating plot...")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Effective Context Length (Tokens)')
        ax1.set_ylabel('Latency Per Token (ms)', color=color)
        ax1.plot(df_results['effective_context_length'], df_results['latency_ms'], color=color, label='Latency')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('VRAM Allocated (MB)', color=color)
        ax2.plot(df_results['effective_context_length'], df_results['vram_mb'], color=color, linestyle='--', label='VRAM')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title(f'TPOT & VRAM vs Context Length\n(Model: {args.model}, Batch: {args.batch_size}, Precision: {args.precision})')
        
        plot_path = outdir / "tpot_and_vram_vs_context.png"
        plt.savefig(plot_path, dpi=300)
        log.info(f"Plot saved successfully to {plot_path}")
        
if __name__ == "__main__":
    main()
