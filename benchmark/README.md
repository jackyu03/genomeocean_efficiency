# GenomeOcean Efficiency Benchmark

A comprehensive benchmarking suite for genomic language models, evaluating both **Quantization Quality** (distributional preservation) and **Downstream Utility** (Metagenomics Binning).

## Overview

This benchmark evaluates how different quantization methods (8-bit, 4-bit NF4/FP4) affect model performance on biological tasks.

It consists of two phases:
1.  **Quality Evaluation**: Measures KL divergence, Cosine Similarity, and Perplexity of quantized vs. standard embeddings.
2.  **Binning Evaluation**: Clusters sequence embeddings using DBSCAN to verify if species-specific signal is preserved.

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Requires: torch, transformers, bitsandbytes, scikit-learn, pandas
```

### 2. Run Benchmark
Use the unified runner script:

```bash
python benchmark/run_full_benchmark.py \
    --csv path/to/dataset.csv \
    --model DOEJGI/GenomeOcean-4B \
    --device cuda
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | **Required** | Input CSV file containing `genome_id` and `seq` columns. |
| `--model` | **Required** | HuggingFace model ID or path. |
| `--quant-modes` | `standard 8bit 4bit_nf4` | List of quantization modes to evaluate. |
| `--max-len` | `5000` | Maximum sequence length (bp) to process. |
| `--n-binning-species` | `50` | Number of species to sample for the binning task. |
| `--outdir` | `./results_full` | Base directory for results (timestamped subfolders created automatically). |

## Output Structure

Results are saved to `results_full/run_YYYYMMDD_HHMMSS/`:

- **`binning_metrics.csv`**: Contains ARI (Adjusted Rand Index), Purity, and fraction of recovered genomes for each quantization mode.
- **`quality/`**: Directory containing:
    - `quality_metrics.csv`: KL Divergence, Cosine Similarity, SNR.
    - `distribution_statistics.csv`: Detailed embedding stats.
