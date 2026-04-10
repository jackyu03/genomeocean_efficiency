# GenomeOcean Efficiency Benchmark

This repository contains scripts for evaluating the efficiency and biological fidelity of Genomic Foundation Models (such as GenomeOcean) under post-training FP8 quantization. The codebase includes tools for data generation, embedding analysis, and high-throughput causal generation benchmarking.

## 1. Dataset Generation
Edit the parameters at the bottom of `assistive_scripts/generate_dataset.py` to produce datasets for different benchmarking protocols.

Protocol A Dataset (Embedding)
Requires at least 20 species, 100 fragments per species, and 50,000 bp length each:
```python
df_result = sample_sequences_parallel(
    directory='<GTDB dataset root>/GTDB/Bacteria/',
    id_list=genome_ids, 
    seqs_per_genome=100, 
    seq_length=50000, 
    n_species=20, 
    seed=42
)
df_result.to_csv('embedding_dataset.csv', index=False)
```

Protocol B Dataset (Generation)
Requires 100 species, 1 fragment per species, and 550,000 bp length each:
```python
df_result = sample_sequences_parallel(
    directory='<GTDB dataset root>/GTDB/Bacteria/',
    id_list=genome_ids, 
    seqs_per_genome=1, 
    seq_length=550000, 
    n_species=100, 
    seed=42
)
df_result.to_csv('generation_dataset.csv', index=False)
```

## 2. Protocol A: Embedding Benchmark
Evaluates embedding extraction through PyTorch FBGEMM and tests the downstream representational fidelity via a DBSCAN species binning task.

```bash
python run_embedding_benchmark.py \
    --csv "$BASE_DATA_DIR/embedding_dataset.csv" \
    --model "DOEJGI/GenomeOcean-100M" \
    --outdir "./results_final/protocol_A_100M_test" \
    --quant-modes standard \
    --loader native \
    --max-tokens 1024 \
    --subdivide-fragments 5000 \
    --umap-dim 10 \
    --dbscan-eps 0.5 \
    --dbscan-min-samples 5 \
    --batch-size 256 \
    --n-genomes 4 \
    --n-fragments 10 \
    --seed 42 \
    --device cuda
```

## 3. Protocol B: Generation Benchmark
Evaluates causal generation (next-token prediction) using vLLM for high-throughput decoding efficiency. Evaluates sequences over sliding-window chunks.

Baseline & Protocol B.1 KV Cache Quantization
Evaluates quality (Perplexity/NLL) across standard BFloat16 and B.1 FP8 KV Cache:
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=3

python run_generation_benchmark.py \
    --csv "$BASE_DATA_DIR/generation_dataset.csv" \
    --model "DOEJGI/GenomeOcean-4B" \
    --quant-modes bf16 fp8 \
    --batch-sizes 48 96 \
    --outdir ./results_final/generation_bench \
    --n-genomes 100 \
    --tokens-per-genome 102400 \
    --context-len 10240 \
    --stride 2560 \
    --gen-len 2560 \
    --precision bfloat16
```

Extended Throughput Generation Measurement
Repeats sequence operations to measure saturated generation throughput limits over fixed time windows:
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=3

python run_generation_benchmark.py \
    --csv "$BASE_DATA_DIR/generation_dataset.csv" \
    --model "DOEJGI/GenomeOcean-4B" \
    --quant-modes fp8 \
    --batch-sizes 96 \
    --outdir ./results_final/generation_bench \
    --n-genomes 96 \
    --tokens-per-genome 102400 \
    --context-len 10240 \
    --stride 2560 \
    --gen-len 2560 \
    --precision bfloat16 \
    --n-repeats 5
```

Protocol B.2 Dynamic W8A8 Quantization
Utilizes the llm-compressor tool to produce a dynamically quantized W8A8 weight checkpoint with an excluded standard language modeling head:
```bash
python quantize_genomeocean_fp8.py --model "DOEJGI/GenomeOcean-4B"
```

Generates sequences and tracks evaluation perplexity utilizing the B.2 W8A8 quantization model:
```bash
python run_generation_benchmark.py \
    --csv "$BASE_DATA_DIR/generation_dataset.csv" \
    --model "DOEJGI/GenomeOcean-4B" \
    --model-w8a8 "<path_to_quantized_model>/GenomeOcean-4B-B2-FP8-W8A8" \
    --quant-modes fp8_v2 \
    --batch-sizes 96 \
    --outdir ./results_final/generation_bench_real_fp8 \
    --n-genomes 96 \
    --tokens-per-genome 102400 \
    --context-len 10240 \
    --stride 2560 \
    --gen-len 2560 \
    --precision bfloat16 \
    --n-repeats 5
```
