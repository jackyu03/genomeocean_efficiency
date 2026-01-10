# GenomeOcean Efficiency Benchmark

A comprehensive benchmarking suite for genomic language models, integrating **Performance** (Speed/Energy), **Quality** (Distributional), and **Downstream Utility** (Binning) into a single, unified pipeline.

## 📂 Project Structure

```text
benchmark/
├── run_full_benchmark.py       # 🚀 Main entry point
├── run_benchmark_hpc.sh        # SLURM submission script
├── core/                       # Core utilities
│   ├── model_loader.py         # Model loading & quantization configuration
│   └── metrics.py              # Performance metrics (Energy/FLOPs)
├── quantization_benchmark/     # Quality evaluation
│   └── quality_eval.py         # KL-Divergence, Cosine Sim, etc.
├── binning_benchmark/          # Downstream evaluation
│   └── eval.py                 # UMAP + DBSCAN logic
```

## ⚙️ The Pipeline: How It Works

When you run `run_full_benchmark.py`, the following process occurs for **each** quantization mode (e.g., `standard`, `8bit`, `4bit_nf4`) specified:

### 1. Model Loading
-   The script imports `load_model` from `core/model_loader.py`.
-   Depending on `os.environ["QUANT_MODE"]`, it loads the HF model with the correct quantization config (BitsAndBytes).

### 2. Performance Benchmarking (Combined with Inference)
-   The model iterates through your dataset in batches.
-   **EnergyMeter** (`core/metrics.py`) runs in the background, logging GPU power usage via NVML.
-   **Speed**: We calculate `Sequences/sec` and `Tokens/sec`.
-   **Embeddings**: While measuring speed, we extract and mean-pool the last hidden states for use in steps 3 & 4.
-   **Result**: Saves `performance_metrics.csv` (Speed, Power, Tokens/Watt).

### 3. Quality Evaluation
-   The script compares the embeddings of the current quantized model against the `standard` (FP16/BF16) baseline.
-   **Metrics**: KL Divergence, Jensen-Shannon Divergence, Cosine Similarity, Signal-to-Noise Ratio (SNR).
-   **Result**: Saves `quality/quality_metrics.csv`.

### 4. Downstream Binning Evaluation
-   Using the **same** embeddings extracted in Step 2:
    1.  **Dimensionality Reduction**: UMAP reduces the high-dim embeddings (e.g., 3072d) to `10d` (for clustering) and `2d` (for visualization).
    2.  **Clustering**: DBSCAN attempts to group sequences into "species bins" based on density.
    3.  **Validation**: Clusters are compared to the ground truth `genome_id` labels.
-   **Result**:
    -   `binning_metrics.csv`: ARI (Adjusted Rand Index), Purity, Silhouette Score.
    -   `plots/{mode}/`: Visual 2D Scatterplots showing predicted vs actual clusters.

---

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install torch transformers bitsandbytes scikit-learn pandas matplotlib seaborn umap-learn scipy
```

### 2. Run Benchmark
```bash
python benchmark/run_full_benchmark.py \
    --csv path/to/dataset.csv \
    --model DOEJGI/GenomeOcean-4B \
    --device cuda \
    --quant-modes standard 8bit 4bit_nf4 \
    --batch-size 8 \
    --umap-dim 10 \
    --n-genomes 50
```

**Input CSV Format:**
The dataset must contain the following columns:
-   `genome_id`: The label/species ID (e.g., `GCA_0000123.1`). Used as ground truth.
-   `fragment_id`: Unique identifier for the sequence fragment (e.g., `GCA_0000123.1_0`).
-   `seq`: The DNA sequence string.

### 3. HPC Submission
Use the provided SLURM script:
```bash
sbatch benchmark/run_benchmark.sh
```

## 📊 Output

All results are saved to `results_full/run_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `performance_metrics.csv` | Throughput (tok/s), Power (W), Efficiency (tok/W), VRAM Usage. |
| `binning_metrics.csv` | Clustering accuracy (ARI), Purity, and Noise %. |
| `quality/quality_metrics.csv` | Distributional fidelity vs Standard model. |
| `plots/{mode}/*.png` | 2D UMAP visualizations of the binning space. |
