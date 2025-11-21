#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=go_benchmark
#SBATCH --output=./results/logs/go_benchmark_%j.out
#SBATCH --error=./results/logs/go_benchmark_%j.err

# Load your environment and move to benchmark directory
cd /global/scratch/users/jingyuhuo2028/genomeocean_efficiency/benchmark/Jingyu_work

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /global/scratch/users/mutianyu2026/conda/envs/GO

# Optional: show CUDA info
nvidia-smi
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run the comparison

python quantization_embedding_comparison.py \
    --model_dir DOEJGI/GenomeOcean-4B \
    --model_max_length 5000 \
    --sequence_file dataset/arc53_2000_seq_50k.csv \
    --batch_size 4 \
    --precision float32 \
    --output_file outputs/quant_embeddings_comparison.csv


echo "Benchmark completed at $(date)"