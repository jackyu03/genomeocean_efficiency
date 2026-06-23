#!/bin/bash

# run_tpot_experiments_for_paper.sh
# 
# This script runs the necessary configurations of the TPOT vs Context 
# experiment to generate the exact data needed for your paper.
# It runs the GenomeOcean-4B model (the one you used for Protocol B) 
# across multiple configurations to clearly demonstrate how KV cache memory 
# and computation costs scale.

DATA_CSV="generation_dataset.csv"
MODEL="DOEJGI/GenomeOcean-4B"
PROMPT_LEN=1024
GEN_LEN=9216

echo "=== Starting TPOT vs Context Experiments for Paper ==="

# 1. Baseline Single-Sequence (Pure compute curve using vLLM)
echo "Running vLLM BF16, Batch Size 1..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision bf16

# 2. FP8 Comparison Run (To directly measure how FP8 KV-Cache helps in vLLM)
echo "Running vLLM FP8, Batch Size 1..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision fp8

# 3. Batched Scaling via vLLM
# (Shows linear KV memory traffic vs quadratic compute)
echo "Running vLLM BF16, Batch Size 16..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --batch-size 16 \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision bf16

echo "Running vLLM BF16, Batch Size 32..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --batch-size 32 \
    --prompt-len $PROMPT_LEN \
    --gen-len $GEN_LEN \
    --precision bf16

# 4. Kernel Profiling Run (via vLLM)
echo "Running Profiling Trace (vLLM BF16)..."
python run_tpot_vllm_experiment.py \
    --csv "$DATA_CSV" \
    --model "$MODEL" \
    --batch-size 1 \
    --prompt-len $PROMPT_LEN \
    --gen-len 100 \
    --precision bf16 \
    --profile

echo "=== All experiments complete. Check results_tpot_vllm/! ==="
