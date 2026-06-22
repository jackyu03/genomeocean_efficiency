#!/bin/bash

# run_tpot_experiments_for_paper.sh
# 
# This script runs the necessary configurations of the TPOT vs Context 
# experiment to generate the exact data needed for your paper.
# It runs the GenomeOcean-4B model (the one you used for Protocol B) 
# across multiple batch sizes to clearly demonstrate how KV cache memory 
# and computation costs scale.

DATA_CSV="generation_dataset.csv"
MODEL="DOEJGI/GenomeOcean-4B"
PROMPT_LEN=625
GEN_LEN=2000

echo "=== Starting TPOT vs Context Experiments for Paper ==="

# 1. Baseline Single-Sequence (Pure compute curve using vLLM)
echo "Running BF16, Batch Size 1..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision bf16

# 4. FP8 Comparison Runs (To directly measure how FP8 KV-Cache helps in vLLM)
echo "Running FP8, Batch Size 1..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision fp8

# 5. Kernel Profiling Run (To prove H100 Flash Attention utilization to Rob)
# We use the NATIVE PyTorch script here because vLLM's background scheduler 
# makes raw kernel Chrome tracing extremely noisy and hard to read.
echo "Running Profiling Trace (BF16)..."
python run_tpot_context_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --batch-size 1 \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision bf16 \
    --attn-impl flash_attention_2 \
    --profile

echo "=== All experiments complete. Check results_tpot_vllm/ and results_tpot/! ==="
