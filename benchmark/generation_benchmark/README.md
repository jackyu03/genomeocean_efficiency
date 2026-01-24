# Generation Quality Benchmark

This module evaluates the **next-token prediction capabilities** of the genomics model under different quantization modes. Unlike embedding benchmarks which test representation quality, this tests the model's core ability to "speak DNA."

## Metrics Evaluated

### 1. Perplexity (PPL) & Negative Log Likelihood (NLL)
Measures the model's "surprise" at the real biological sequence.
*   **Formula:** $PPL = \exp(-\frac{1}{N} \sum \ln P(x_i | x_{<i}))$
*   Lower is better.
*   Uses a **Sliding Window** to ensure deep context for every token.

### 2. Next-Token Accuracy
Measures the percentage of times the model's top prediction (rank 1) matches the actual next base/token.
*   **Formula:** $\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$
*   Higher is better.
*   Uses the **same Sliding Window** logic as Perplexity to ensure fair comparison.

## Methodology: Sliding Window Evaluation

To extend the model's limited context window (e.g., 2048 tokens) to infinite genomic sequences (e.g., 50k bp contigs), we use a sliding window approach with a **stride**.

**Example:** Context = 1024, Stride = 512.

1.  **Window 1:** Feed `[0 ... 1024]`.
    *   **Context:** `[0 ... 512]` (Used for history only).
    *   **Target:** `[512 ... 1024]` (Only these 512 tokens are graded).
2.  **Window 2:** Slide forward by 512. Feed `[512 ... 1536]`.
    *   **Context:** `[512 ... 1024]` (Recycled from prev window).
    *   **Target:** `[1024 ... 1536]` (New tokens graded).

This ensures every single token (except the very first block) is predicted with **at least 512 tokens of prior context**.

## Usage

Run the benchmark from the root directory:

```bash
python benchmark/run_generation_benchmark.py \
    --csv path/to/gtdb.csv \
    --model jackyu03/genome-ocean-4b \
    --quant-modes standard 8bit 4bit_nf4 \
    --context-len 2048 \
    --stride 1024 \
    --n-genomes 50 \
    --n-fragments 5
```

### Arguments
*   `--quant-modes`: List of modes to test (default: `standard 8bit`).
*   `--context-len`: Max window size. If not set, reads `model.config.max_position_embeddings`.
*   `--stride`: Step size. If not set, defaults to `context // 2`.
*   `--n-genomes` / `--n-fragments`: Subsampling parameters to speed up testing.
