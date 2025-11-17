#!/bin/bash
#SBATCH --job-name=quality_eval
#SBATCH --output=logs/quality_eval_%j.out
#SBATCH --error=logs/quality_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

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
MODEL_NAME="DOEJGI/GenomeOcean-4B"
OUTPUT_DIR="./results/4B/quality_eval_$(date +%Y%m%d_%H%M%S)"
DATASET_FILE="dataset/arc53_2000_seq_50k.csv"
NUM_SEQUENCES=100

# Quantization modes to evaluate
QUANT_MODES="standard int8 int4 nf4 fp4"

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
