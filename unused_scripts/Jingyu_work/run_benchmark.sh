#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 12:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=go_bench
#SBATCH --output=/global/scratch/users/jingyuhuo2028/genomeocean_efficiency/benchmark/Jingyu_work/results/logs/benchmark_%j.log
#SBATCH --error=/global/scratch/users/jingyuhuo2028/genomeocean_efficiency/benchmark/Jingyu_work/results/logs/benchmark_%j.log

# ==============================================================================
# GenomeOcean Unified Benchmark Runner (HPC)
# ==============================================================================
# Configuration
MODEL_NAME="DOEJGI/GenomeOcean-4B"
DATASET_FILE="./Jingyu_work/dataset/arc53_2000_seq_50k.csv"
OUTPUT_DIR="./Jingyu_work/results"
QUANT_MODES="standard 8bit 4bit_nf4 4bit_fp4 4bit_nf4_double"

# Parameters
UMAP_DIM=10
DBSCAN_EPS=0.5
DBSCAN_MIN_SAMPLES=5
MAX_TOKENS=8196
N_GENOMES=20
N_FRAGMENTS=100
SEED=42

cd /global/scratch/users/jingyuhuo2028/genomeocean_efficiency/benchmark || exit

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
    --max-tokens $MAX_TOKENS \
    --umap-dim $UMAP_DIM \
    --dbscan-eps $DBSCAN_EPS \
    --dbscan-min-samples $DBSCAN_MIN_SAMPLES \
    --batch-size 2 \
    --n-genomes $N_GENOMES \
    --n-fragments $N_FRAGMENTS \
    --seed $SEED \
    --device cuda

echo "Benchmark finished at $(date)"
