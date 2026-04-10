# GenomeOcean Efficiency Benchmark

A comprehensive benchmarking suite for genomic language models, integrating **Performance** (Speed/Energy), **Quality** (Distributional), and **Downstream Utility** (Binning) into a single, Unified Pipeline.

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

This benchmark runs a strictly controlled experiment to evaluate how model quantization affects genomic tasks. When you run `run_full_benchmark.py`, it executes the following steps:

### 1. Data Loading & Filtering
The script enforces reproducibility by controlling exactly which data is seen by the model.
-   **Input**: Loads the provided CSV dataset.
-   **Genome Selection** (`--n-genomes`, `--seed`): Randomly selects $N$ unique species (e.g., 20) from the dataset using a fixed random seed.
-   **Fragment Subsampling** (`--n-fragments`): For each selected species, it randomly selects $M$ non-overlapping fragments (e.g., 50).
-   **Output**: A fixed subset of sequences (e.g., $20 \times 50 = 1000$ sequences) that will be used for all subsequent steps.

### 2. Initialization
-   It iterates through each requested **Quantization Mode** (e.g., `standard` (FP16), `8bit`, `4bit_nf4`).
-   For each mode, it loads the GenomeOcean model using `BitsAndBytesConfig`.

### 3. Performance Profiling
-   **Inference**: The model processes the filtered dataset in batches.
-   **Metrics**:
    -   **Throughput**: Sequences/sec and Tokens/sec.
    -   **Power**: Logs real-time GPU power usage (Watts) via NVML to calculate **Tokens/Watt**.
    -   **VRAM**: Tracks peak GPU memory usage.
-   **Embeddings**: Simultaneously extracts mean-pooled hidden states (embeddings) for downstream tasks.

### 4. Quality Evaluation
-   **Comparison**: Compares the embeddings of the current quantized model against the `standard` (FP16) baseline.
-   **Metrics**:
    -   **KL Divergence / Jensen-Shannon**: Measures how much the output distribution shifts.
    -   **Cosine Similarity**: Measures directional fidelity of the embeddings.
    -   **Signal-to-Noise Ratio (SNR)**: Quantifies the "noise" introduced by quantization.

### 5. Downstream Binning (Clustering)
-   **Goal**: Can the quantized embeddings still distinguish between different species?
-   **Dimensionality Reduction**: UMAP projects high-dimensional embeddings (e.g., 3072d) to:
    -   **10d** for DBSCAN clustering.
    -   **2d** for visualization.
-   **Clustering**: DBSCAN clusters the sequences based on density.
-   **Validation**:
    -   **ARI (Adjusted Rand Index)**: 1.0 = Perfect match with Ground Truth `genome_id`.
    -   **Purity**: How "pure" each cluster is (percentage of dominant species).
-   **Visualization**: Generates side-by-side scatter plots (Predicted Clusters vs. True Species).

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
    --max-tokens 5000 \
    --batch-size 8 \
    --umap-dim 10 \
    --n-genomes 50 \
    --n-fragments 100 \
    --seed 42
```

**Input CSV Requirements:**
The dataset **must** contain the following headers:
-   `genome_id`: The ground truth label (e.g., `GCA_0000123.1`). Used for calculating ARI/Purity.
-   `fragment_id`: Unique identifier for each sequence (e.g., `GCA_0000123.1_0`).
-   `seq`: The actual DNA sequence string.

### 3. HPC Submission
Adjust the variables in `benchmark/run_benchmark.sh` and submit:
```bash
# Parameters in shell script:
# N_GENOMES=20      <-- Number of species to sample
# N_FRAGMENTS=50    <-- Fragments per species
# SEED=42           <-- Random seed for reproducibility

sbatch benchmark/run_benchmark.sh
```

## 📊 Output Structure

Results are saved to timestamped directories, e.g., `results_full/run_20240101_120000/`:

| File/Folder | Content |
|-------------|---------|
| `performance_metrics.csv` | **Speed & Power**: `tokens_per_s`, `avg_power_W`, `tokens_per_watt`, `peak_vram_GB`. |
| `binning_metrics.csv` | **Downstream Task**: `ARI`, `completeness`, `silhouette_score`, `n_clusters`. |
| `quality/` | **Fidelity**: `quality_metrics.csv` (KL, Cosine vs Standard). |
| `plots/{mode}/` | **Visuals**: `cluster_viz_predicted.png` (DBSCAN results) vs `cluster_viz_truth.png` (Species labels). |
