#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 12:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=quant_eval
#SBATCH --output=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/quant_eval_%j.log
#SBATCH --error=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/quant_eval_%j.log

# Quality Evaluation Only - Skip Performance Benchmarking
# This script runs only the quality evaluation comparing quantized vs standard models

cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/

set -e

echo "=========================================="
echo "Quality Evaluation Only"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Configuration
MODEL_NAME="DOEJGI/GenomeOcean-100M"
OUTPUT_DIR="./results/quality_eval_$(date +%Y%m%d_%H%M%S)"
DATASET_FILE="dataset/arc53_2000_seq_50k.csv"
NUM_SEQUENCES=500

# Quantization modes to evaluate
QUANT_MODES="standard int8 4bit_int4 4bit_nf4 4bit_fp4 4bit_nf4_double"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Dataset: $DATASET_FILE"
echo "  Num sequences: $NUM_SEQUENCES"
echo "  Quantization modes: $QUANT_MODES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Activate environment (adjust as needed)
# Activate conda environment
source /global/scratch/users/mutianyu2026/conda/etc/profile.d/conda.sh
conda activate GO

# Run quality evaluation
echo "Starting quality evaluation..."
python benchmark/scripts/run_quality_eval_only.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --test-sequences-file "$DATASET_FILE" \
    --num-sequences "$NUM_SEQUENCES" \
    --quantization-modes $QUANT_MODES \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Quality evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - quality_metrics.csv"
    echo "  - distribution_statistics.csv"
    echo "  - divergence_comparison.png"
    echo "  - cosine_similarity.png"
    echo "  - snr_comparison.png"
else
    echo "Quality evaluation failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
