# GenomeOcean Efficiency Benchmark

A comprehensive benchmarking suite for genomic language models, integrating **Performance** (Speed/Energy), **Quality** (Distributional), and **Downstream Utility** (Binning) into a single pipeline.

## Features

1.  **Performance**: Measures throughput (tokens/s), perplexity-agnostic FLOPs, and Energy Consumption (NVML).
2.  **Quality**: Evaluates quantization degradation using KL Divergence, Cosine Similarity, and Signal-to-Noise Ratio.
3.  **Binning**: validatess biological signal preservation by clustering metagenomic contigs (UMAP + DBSCAN).

## Structure

```text
benchmark/
├── run_full_benchmark.py       # Main Entry Point
├── run_benchmark_hpc.sh        # SLURM Submission Script
├── common/                     # Shared Utilities
│   └── metrics.py              # EnergyMeter & FLOPs logic
├── quantization_benchmark/     # Quality Metrics
├── binning_benchmark/          # Binning Logic (UMAP/DBSCAN)
└── model_to_benchmark.py       # Model Loading
```

## Usage

### 1. Install Dependencies
```bash
pip install torch transformers bitsandbytes scikit-learn pandas matplotlib seaborn umap-learn scipy
```

### 2. Run Benchmark
Use `run_full_benchmark.py`. This script runs all phases (Performance -> Quality -> Binning) for each specified quantization mode.

```bash
python benchmark/run_full_benchmark.py \
    --csv path/to/dataset.csv \
    --model DOEJGI/GenomeOcean-4B \
    --device cuda \
    --quant-modes standard 8bit 4bit_nf4
```

### 3. HPC Submission
Adjust headers in `benchmark/run_benchmark.sh` and submit:
```bash
sbatch benchmark/run_benchmark.sh
```

## Output

Results are saved to `results_full/run_YYYYMMDD_HHMMSS/`:

-   **`performance_metrics.csv`**: Speed (tok/s), Power (W), Efficiency (tok/W).
-   **`binning_metrics.csv`**: ARI, Purity, Silhouette Score, N_Clusters.
-   **`quality/`**: `quality_metrics.csv` (KL, Cosine), `distribution_statistics.csv`.
-   **`plots/{mode}/`**:
    -   `cluster_viz_predicted.png`: UMAP scatterplot colored by predicted cluster.
    -   `cluster_viz_truth.png`: UMAP scatterplot colored by true species.
