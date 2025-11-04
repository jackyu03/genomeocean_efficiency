#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 8:00:00
#SBATCH --ntasks=8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=quant_benchmark
# Create logs directory first (before SLURM tries to write to it)
#SBATCH --output=quant_benchmark_%j.out
#SBATCH --error=quant_benchmark_%j.err

# Comprehensive quantization benchmark script
# Tests all quantization modes: standard, 8bit, 4bit variants

# Print initial info for debugging
echo "=== JOB INITIALIZATION DEBUG ==="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Shell: $SHELL"
echo "PATH: $PATH"
echo "Initial Python: $(which python 2>/dev/null || echo 'python not found')"
echo "Initial Python version: $(python --version 2>/dev/null || echo 'python not available')"
echo "================================="

# Create results and logs directories first
mkdir -p ./results/logs

# Move log files to proper location
mv "quant_benchmark_${SLURM_JOB_ID}.out" "./results/logs/" 2>/dev/null || true
mv "quant_benchmark_${SLURM_JOB_ID}.err" "./results/logs/" 2>/dev/null || true

# Load environment and move to benchmark directory
echo "Changing to benchmark directory..."
cd /global/scratch/users/mutianyu2026/genomeocean_efficiency/benchmark/ || {
    echo "ERROR: Cannot change to benchmark directory"
    exit 1
}

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate GO

echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Comprehensive system diagnostics
echo "=== COMPREHENSIVE SYSTEM DIAGNOSTICS ==="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

echo "--- GPU Information ---"
nvidia-smi
echo ""
echo "GPU Name: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "GPU Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep release || echo 'nvcc not found')"
echo ""

echo "--- Software Versions ---"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Transformers Version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "BitsAndBytes Version: $(python -c 'import bitsandbytes; print(bitsandbytes.__version__)' 2>/dev/null || echo 'Not installed')"
echo "NumPy Version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "Pandas Version: $(python -c 'import pandas; print(pandas.__version__)')"
echo ""

echo "--- System Resources ---"
echo "CPU Info: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPU Cores: $(nproc)"
echo "Total RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available RAM: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "Disk Space: $(df -h . | tail -1 | awk '{print $4}' | xargs) available"
echo ""

echo "--- Environment Variables ---"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-'Not set'}"
echo "SLURM_PROCID: ${SLURM_PROCID:-'Not set'}"
echo "=============================================="

# Check if required files exist
echo "Checking required files..."
CSV_FILE="dataset/arc53_2000_seq_50k.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: Dataset file not found: $CSV_FILE"
    echo "Available files in dataset/:"
    ls -la dataset/ 2>/dev/null || echo "Dataset directory not found"
    exit 1
fi

# Check if Python scripts exist
if [ ! -f "go_benchmark.py" ]; then
    echo "ERROR: go_benchmark.py not found"
    exit 1
fi

if [ ! -f "analyze_quantization_results.py" ]; then
    echo "ERROR: analyze_quantization_results.py not found"
    exit 1
fi

# Common benchmark parameters
MODEL="DOEJGI/GenomeOcean-100M"
DEVICE="cuda"
PRECISION="float16"
SAMPLES_PER_COND=1000
BATCH_SIZES="1 4 8"
WARMUP=3
OUTDIR="./results"

# Quantization modes to test
QUANT_MODES=("standard" "8bit" "4bit_nf4" "4bit_fp4" "4bit_nf4_double")

# Test configurations: truncation by base pairs (focused on 10k bp sequences)
TRUNCATE_BP="10000"

echo "=== Starting Quantization Benchmark ==="
echo "Testing quantization modes: ${QUANT_MODES[@]}"
echo "Truncation targets (bp): $TRUNCATE_BP"
echo "Batch sizes: $BATCH_SIZES"
echo "========================================"

# Run benchmark for each quantization mode
for QUANT_MODE in "${QUANT_MODES[@]}"; do
    echo ""
    echo "=== Testing Quantization Mode: $QUANT_MODE ==="
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
        echo "ERROR: Benchmark failed for quantization mode: $QUANT_MODE"
        continue
    }
    
    echo "Completed quantization mode: $QUANT_MODE at $(date)"
    
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Brief pause between runs
    sleep 10
done

echo ""
echo "=== Quantization Benchmark Complete ==="
echo "Finished at: $(date)"
echo "Results saved in: $OUTDIR"
echo "======================================="

# Combine all results into a single summary file
echo ""
echo "=== COMBINING RESULTS ==="
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
COMBINED_CSV="$OUTDIR/quantization_benchmark_combined_$TIMESTAMP.csv"

# Find all benchmark CSV files from this run and combine them
find "$OUTDIR" -name "benchmark_*.csv" -type f -newer "$OUTDIR" 2>/dev/null | head -10 | while read file; do
    if [ ! -f "$COMBINED_CSV" ]; then
        # First file: copy with header
        cp "$file" "$COMBINED_CSV"
        echo "Created combined file with: $(basename "$file")"
    else
        # Subsequent files: append without header
        tail -n +2 "$file" >> "$COMBINED_CSV"
        echo "Added data from: $(basename "$file")"
    fi
done

if [ -f "$COMBINED_CSV" ]; then
    echo "[OK] Combined results saved to: $COMBINED_CSV"
    echo "[INFO] Total benchmark rows: $(wc -l < "$COMBINED_CSV")"
else
    echo "[WARNING] No results files found to combine"
    # Try to find any CSV files
    echo "Available CSV files:"
    find "$OUTDIR" -name "*.csv" -type f | head -5
fi

# Run comprehensive analysis and create visualizations
echo ""
echo "=== RUNNING ANALYSIS & CREATING VISUALIZATIONS ==="
ANALYSIS_DIR="$OUTDIR/analysis_$TIMESTAMP"
mkdir -p "$ANALYSIS_DIR"

if [ -f "$COMBINED_CSV" ]; then
    echo "[INFO] Analyzing results and generating visualizations..."
    
    # Run the comprehensive analysis script
    python analyze_quantization_results.py \
        --results-dir "$OUTDIR" \
        --output-dir "$ANALYSIS_DIR" || {
        echo "[ERROR] Analysis script failed, but continuing..."
    }

    echo ""
    echo "=== ANALYSIS RESULTS ==="
    if [ -f "$ANALYSIS_DIR/quantization_benchmark_report.txt" ]; then
        echo "[REPORT] Summary Report:"
        cat "$ANALYSIS_DIR/quantization_benchmark_report.txt"
    fi
    
    echo ""
    echo "[FILES] Generated Files:"
    find "$ANALYSIS_DIR" -type f -name "*.png" -o -name "*.txt" | while read file; do
        echo "  * $(basename "$file")"
    done
    
    echo ""
    echo "[PLOTS] Key Visualizations Created:"
    echo "  * throughput_comparison.png - Sequences/sec and Tokens/sec by quantization mode"
    echo "  * memory_comparison.png - Peak VRAM usage by quantization mode"  
    echo "  * energy_efficiency.png - Tokens per watt by quantization mode"
    echo "  * performance_memory_tradeoff.png - Performance vs memory usage scatter plot"
    
else
    echo "[WARNING] No combined CSV file found, skipping analysis"
fi

echo ""
echo "=== FINAL SUMMARY ==="
echo "[FOCUS] Benchmark Focus: 10k bp sequences across all quantization modes"
echo "[TIME] Total Runtime: Started at benchmark start, finished at $(date)"
echo "[RESULTS] Results Location: $OUTDIR"
echo "[ANALYSIS] Analysis Location: $ANALYSIS_DIR"
echo "[COMPLETE] QUANTIZATION BENCHMARK COMPLETE!"
echo "=================================="