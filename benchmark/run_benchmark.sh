#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 3:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=go_benchmark
#SBATCH --output=./results/logs/go_benchmark_%j.out
#SBATCH --error=./results/logs/go_benchmark_%j.err

# Load your environment and move to benchmark directory
cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate GO

# Optional: show CUDA info
nvidia-smi
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run the benchmark

# Use explicit truncation instead of binning
# For bp truncation:
 python go_benchmark.py \
   --csv dataset/arc53_2000_seq_50k.csv \
   --model DOEJGI/GenomeOcean-100M \
   --device cuda \
   --precision float16 \
   --samples-per-cond 1000 \
   --truncate-bp 1000 2500 5000 10000 \
   --batch-sizes 1 4 8 \
   --warmup 3 \
   --outdir ./results

# For token truncation:
# python go_benchmark.py \
#   --csv dataset/arc53_2000_seq_50k.csv \
#   --model DOEJGI/GenomeOcean-100M \
#   --device cuda \
#   --precision float16 \
#   --samples-per-cond 1000 \
#   --truncate-tokens 512 1024 2048 4096 \
#   --batch-sizes 1 4 8 \
#   --warmup 3 \
#   --outdir ./results

echo "Benchmark completed at $(date)"