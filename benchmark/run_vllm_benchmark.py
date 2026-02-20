import argparse
import time
import numpy as np
import os
from vllm import LLM, SamplingParams
def run_benchmark(model_name: str, seq_len: int, batch_size: int, gen_len: int, use_fp8_kv: bool, use_fp8_weights: bool):
    # We allow vLLM to automatically select the best attention backend (FA3 or XFormers)

    print(f"--- vLLM Benchmark | Model: {model_name} ---")
    print(f"Seq Len: {seq_len} | Batch: {batch_size}")
    print(f"FP8 KV Cache: {use_fp8_kv} | FP8 Weights: {use_fp8_weights}")
    
    # Configure vLLM engine
    # If dealing with long sequences, we want to maximize our KV cache memory usage
    kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": batch_size,
        "max_model_len": seq_len + gen_len + 32,  # allow space for generation
        "enforce_eager": True # Fixes CUDA graph broadcast error with XFormers backend > 8192 tokens
    }
    
    if use_fp8_kv:
        kwargs["kv_cache_dtype"] = "fp8"  # <--- THIS fixes the N^2 Attention bottleneck!
        
    if use_fp8_weights:
        kwargs["quantization"] = "fp8"    # Requires pre-quantized FP8 checkpoint (W8A8)
        
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        print(f"\n[Error] Failed to load vLLM engine: {e}")
        if use_fp8_weights:
            print("Note: vLLM's 'quantization=\"fp8\"' expects a pre-quantized FP8 checkpoint.")
            print("If your model isn't pre-quantized, turn off FP8 weights but leave FP8 KV Cache ON!")
        return
        
    # Create dummy integer prompts to bypass tokenizer limits
    vocab_size = 4096 # Safe fallback
    try:
        vocab_size = llm.get_tokenizer().vocab_size
    except:
        pass
        
    # Generate gen_len tokens
    dummy_prompts = [np.random.randint(0, vocab_size, seq_len).tolist() for _ in range(batch_size)]
    
    sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=gen_len,
        ignore_eos=True
    )
    
    print("\nWarming up engine...")
    _ = llm.generate(prompt_token_ids=[dummy_prompts[0]], sampling_params=sampling_params, use_tqdm=False)
    
    print("\nRunning Throughput Benchmark...")
    start_time = time.perf_counter()
    _ = llm.generate(prompt_token_ids=dummy_prompts, sampling_params=sampling_params, use_tqdm=True)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    total_tokens = batch_size * gen_len
    tps = total_tokens / duration
    
    print(f"\n--- Benchmark Results ---")
    print(f"Time Taken: {duration:.3f} s")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Throughput: {tps:.2f} tokens / second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DOEJGI/GenomeOcean-4B")
    parser.add_argument("--seq-len", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-len", type=int, default=1024, help="Number of tokens to generate")
    parser.add_argument("--fp8-kv", action="store_true", help="Enable FP8 KV Cache (solves Attention scaling)")
    parser.add_argument("--fp8-weight", action="store_true", help="Enable FP8 Weights (requires FP8 checkpoint)")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.seq_len, args.batch_size, args.gen_len, args.fp8_kv, args.fp8_weight)
