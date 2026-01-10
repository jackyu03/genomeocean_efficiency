#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 12:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=go_bench
#SBATCH --output=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/benchmark_%j.log
#SBATCH --error=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/benchmark_%j.log

# ==============================================================================
# GenomeOcean Unified Benchmark Runner (HPC)
# ==============================================================================
# Configuration
MODEL_NAME="DOEJGI/GenomeOcean-100M"
DATASET_FILE="dataset/arc53_2000_seq_50k.csv"
OUTPUT_DIR="./results_hpc"
QUANT_MODES="standard 8bit 4bit_nf4"

# Parameters
UMAP_DIM=10
DBSCAN_EPS=0.5
DBSCAN_MIN_SAMPLES=5
MAX_LEN=5000
N_GENOMES=50

cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/ || exit

# Activate environment
source /global/scratch/users/mutianyu2026/conda/etc/profile.d/conda.sh
conda activate GO

echo "Starting Benchmark for $MODEL_NAME..."
echo "Timestamp: $(date)"

python run_full_benchmark.py \
    --csv "$DATASET_FILE" \
    --model "$MODEL_NAME" \
    --outdir "$OUTPUT_DIR" \
    --quant-modes $QUANT_MODES \
    --max-len $MAX_LEN \
    --umap-dim $UMAP_DIM \
    --dbscan-eps $DBSCAN_EPS \
    --dbscan-min-samples $DBSCAN_MIN_SAMPLES \
    --batch-size 8 \
    --n-genomes $N_GENOMES \
    --device cuda

echo "Benchmark finished at $(date)"
