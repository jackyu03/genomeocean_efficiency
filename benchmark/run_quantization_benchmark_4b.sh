#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 12:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=quant_benchmark
#SBATCH --output=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/quant_benchmark_%j.log
#SBATCH --error=/global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/results/logs/quant_benchmark_%j.log

# Quantization benchmark script
# Tests all quantization modes: standard, 8bit, 4bit variants

# Load environment and move to benchmark directory
cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/

# Activate conda environment
source /global/scratch/users/mutianyu2026/conda/etc/profile.d/conda.sh
conda activate GO

# Create results and logs directories
mkdir -p ./results/logs

# Show system info
nvidia-smi
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ============================================
# BENCHMARK CONFIGURATION - EDIT THESE
# ============================================

# Dataset and model
CSV_FILE="dataset/arc53_2000_seq_50k.csv"
MODEL="DOEJGI/GenomeOcean-4B"

# Sequence length to test (in base pairs)
TRUNCATE_BP="50000"

# Benchmark parameters
DEVICE="cuda"
PRECISION="float16"
SAMPLES_PER_COND=1000
BATCH_SIZES="4"
WARMUP=3
OUTDIR="./results/4B"

# Quantization modes to test
QUANT_MODES=("standard" "8bit" "4bit_int4" "4bit_nf4" "4bit_fp4" "4bit_nf4_double")

# ============================================

echo "Testing quantization modes: ${QUANT_MODES[@]}"
echo "Truncation target: $TRUNCATE_BP bp"

# Run benchmark for each quantization mode
for QUANT_MODE in "${QUANT_MODES[@]}"; do
    echo ""
    echo "Testing quantization mode: $QUANT_MODE"
    
    # Set environment variable for quantization mode
    export QUANT_MODE=$QUANT_MODE
    
    # Run the benchmark (repeats handled inside go_benchmark.py)
    python go_benchmark.py \
        --csv "$CSV_FILE" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --precision "$PRECISION" \
        --samples-per-cond $SAMPLES_PER_COND \
        --truncate-bp $TRUNCATE_BP \
        --batch-sizes $BATCH_SIZES \
        --warmup $WARMUP \
        --num-repeats 1 \
        --outdir "$OUTDIR"
    
    echo "Completed: $QUANT_MODE"
    
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo ""
echo "Quantization benchmark completed at $(date)"

# Run complete analysis with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ANALYSIS_DIR="$OUTDIR/analysis_$TIMESTAMP"

echo ""
echo "Running complete analysis (performance + quality + visualizations)..."
python scripts/run_complete_analysis.py \
    --results-dir "$OUTDIR" \
    --output-dir "$ANALYSIS_DIR" \
    --model-name "$MODEL" \
    --num-test-sequences 100

echo ""
echo "Analysis complete!"
echo "Results saved to: $ANALYSIS_DIR"