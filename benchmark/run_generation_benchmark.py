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
    parser.add_argument("--context-len", type=int, default=1024, help="Max context length window")
    parser.add_argument("--stride", type=int, default=256, help="Window stride length")

    args = parser.parse_args()
    
    # Proactively set the Attention Backend. 
    # FlashInfer natively computes under FP8 but ONLY supports head_dim in [64, 128, 256].
    # GenomeOcean-100M has head_dim=96 (unsupported), while GenomeOcean-4B has head_dim=128 (supported).
    if "fp8" in args.quant_modes:
        if "4B" in args.model:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        else:
            # Fallback to XFORMERS for 100M model or others with non-standard head_dim
            os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
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
    
    all_results = []
    
    modes = args.quant_modes
    # Resolve 'standard' to 'bf16'
    modes = ["bf16" if m == "standard" else m for m in modes]
    
    for mode in modes:
        log.info(f"\n=== Evaluating Mode: {mode} ===")
        log.info("Booting vLLM Engine...")
        
        from vllm import LLM
        vllm_kwargs = {
            "model": args.model,
            "trust_remote_code": True,
            "dtype": args.precision,
            "tensor_parallel_size": 1,
            "max_num_seqs": 128, # Safe batch size for smaller chunks
            "max_model_len": max_dataset_len + 128,
            "enforce_eager": True
        }
        
        if mode == "fp8":
            vllm_kwargs["kv_cache_dtype"] = "fp8"
            # We don't force 'quantization'='fp8' weight quant unless user explicitly set up fp8 checkpoints.
            # vLLM provides massive speedup just with FP8 KV cache + BF16 weights.
            log.info("Mode 'fp8' detected: Assiging fp8 KV Cache to vLLM.")
            
        try:
            llm = LLM(**vllm_kwargs)
        except Exception as e:
            log.error(f"Failed to load vLLM engine for mode {mode}: {e}")
            continue
        
        log.info(f"=== Running Benchmark Extraction ({mode}) ===")
        
        results_list = compute_metrics_vllm(llm, genome_chunks_list)
        
        for gid, res in zip(ordered_gids, results_list):
            all_results.append({
                "genome_id": gid,
                "quantization": mode,
                "model": args.model,
                "perplexity": res['perplexity'],
                "neg_log_likelihood": res['neg_log_likelihood'],
                "accuracy": res['accuracy'], 
                "total_tokens": res['total_tokens']
            })
            
        del llm
        import torch
        torch.cuda.empty_cache()

    if all_results:
        df_res = pd.DataFrame(all_results)
        save_path = outdir / "generation_metrics_by_genome.csv"
        df_res.to_csv(save_path, index=False)
        log.info(f"Detailed results saved to {save_path}")
        
        summary_lines = ["=== Benchmark Summary ===\n"]
        
        for mode in modes:
            subset = df_res[df_res["quantization"] == mode]
            if len(subset) == 0: continue
            
            mean_ppl = subset["perplexity"].mean()
            std_ppl = subset["perplexity"].std()
            
            mean_nll = subset["neg_log_likelihood"].mean()
            std_nll = subset["neg_log_likelihood"].std()
            
            line = (f"Mode: {mode}\n"
                    f"  Perplexity: {mean_ppl:.4f} ± {std_ppl:.4f}\n"
                    f"  NLL:        {mean_nll:.4f} ± {std_nll:.4f}\n"
                    f"  (N={len(subset)} genomes)\n")
            summary_lines.append(line)
            print(line)
                
        summary_path = outdir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.writelines(summary_lines)
        log.info(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    main()
