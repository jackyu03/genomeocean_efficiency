#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 12:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=precision_quant_benchmark
#SBATCH --output=./results/logs/precision_quant_benchmark_%j.out
#SBATCH --error=./results/logs/precision_quant_benchmark_%j.err

# Comprehensive benchmark testing different precision modes with quantization
# Tests combinations of precision (float16, bfloat16, float32) with quantization modes

# Load environment and move to benchmark directory
cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate GO

# Create results and logs directories
mkdir -p ./results/logs

# Show system info
echo "=== System Information ==="
nvidia-smi
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================="

# Common benchmark parameters
CSV_FILE="dataset/arc53_2000_seq_50k.csv"
MODEL="DOEJGI/GenomeOcean-100M"
DEVICE="cuda"
SAMPLES_PER_COND=500  # Reduced for comprehensive testing
BATCH_SIZES="1 4 8"
WARMUP=2
OUTDIR="./results"

# Test configurations
PRECISION_MODES=("float16" "bfloat16")  # float32 often too memory intensive with large models
QUANT_MODES=("standard" "8bit" "4bit_nf4" "4bit_nf4_double")
TRUNCATE_BP="2500 5000"  # Focused on mid-range sequences

echo "=== Starting Precision + Quantization Benchmark ==="
echo "Testing precision modes: ${PRECISION_MODES[@]}"
echo "Testing quantization modes: ${QUANT_MODES[@]}"
echo "Truncation targets (bp): $TRUNCATE_BP"
echo "=================================================="

# Run benchmark for each combination
for PRECISION in "${PRECISION_MODES[@]}"; do
    for QUANT_MODE in "${QUANT_MODES[@]}"; do
        echo ""
        echo "=== Testing: Precision=$PRECISION, Quantization=$QUANT_MODE ==="
        echo "Started at: $(date)"
        
        # Set environment variable for quantization mode
        export QUANT_MODE=$QUANT_MODE
        
        # Run the benchmark
        python go_benchmark.py \
            --csv "$CSV_FILE" \
            --model "$MODEL" \
            --device "$DEVICE" \
            --precision "$PRECISION" \
            --samples-per-cond $SAMPLES_PER_COND \
            --truncate-bp $TRUNCATE_BP \
            --batch-sizes $BATCH_SIZES \
            --warmup $WARMUP \
            --outdir "$OUTDIR" || {
            echo "ERROR: Benchmark failed for precision=$PRECISION, quantization=$QUANT_MODE"
            continue
        }
        
        echo "Completed: Precision=$PRECISION, Quantization=$QUANT_MODE at $(date)"
        
        # Clear GPU memory between runs
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 5
    done
done

echo ""
echo "=== Precision + Quantization Benchmark Complete ==="
echo "Finished at: $(date)"
echo "Results saved in: $OUTDIR"
echo "=================================================="