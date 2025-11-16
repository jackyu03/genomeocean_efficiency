# Quantization Benchmark

Comprehensive benchmarking and quality evaluation for quantized genomic models.

## Quick Start

### Run Complete Benchmark

```bash
sbatch run_quantization_benchmark.sh
```

This will:
- Test all quantization modes (standard, 8bit, 4bit variants)
- Measure performance (throughput, memory, latency, energy)
- Evaluate quality (KL divergence, cosine similarity)
- Create visualizations
- Save results to `./results/analysis/`

### View Results

```bash
# Main performance metrics
cat ./results/analysis/quant_summary.csv

# Quality metrics
cat ./results/analysis/quality_metrics.csv

# Visualizations
ls ./results/analysis/*.png
```

### Run Analysis Only (If You Have Results)

```bash
python scripts/run_complete_analysis.py \
    --results-dir ./results \
    --output-dir ./results/analysis \
    --skip-quality
```

## Output Files

All saved to `./results/analysis/`:

**Performance Analysis:**
- `quant_summary.csv` - Main performance metrics by quantization mode
- `memory_analysis.csv` - Memory usage statistics
- `throughput_comparison.csv` - Throughput comparison
- `best_performers.csv` - Best configurations

**Quality Evaluation:**
- `quality_metrics.csv` - KL divergence, cosine similarity, SNR
- `distribution_statistics.csv` - Distribution stats per layer

**Visualizations:**
- `throughput_comparison.png`
- `memory_comparison.png`
- `performance_memory_tradeoff.png`
- `divergence_comparison.png`
- `cosine_similarity.png`
- And more...

## Quantization Modes

- `standard` - No quantization (baseline)
- `8bit` - 8-bit quantization
- `4bit_nf4` - 4-bit NF4 quantization
- `4bit_fp4` - 4-bit FP4 quantization
- `4bit_nf4_double` - 4-bit NF4 with double quantization

## Structure

```
benchmark/
├── quantization_benchmark/     # Core library
│   ├── analysis.py            # Performance analysis
│   ├── quality_eval.py        # Quality evaluation
│   └── visualization.py       # Plotting functions
├── scripts/
│   └── run_complete_analysis.py
├── run_quantization_benchmark.sh
├── model_to_benchmark.py
└── go_benchmark.py
```
