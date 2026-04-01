# Thesis Scratchpad: Experiment Design for FP8 Inference Optimization of GenomeOcean-4B

> Research Notes — *last updated: March 2026*

---

## Table of Contents

1. [Overview: The Inference Optimization Hierarchy](#overview)
2. [Protocol B.1: KV-Cache Quantization (Storage-First)](#b1)
3. [Protocol B.2: W8A8 Dynamic Quantization (Compute-First)](#b2)
4. [Protocol B.3: Full FP8 (Zero 16-bit Components)](#b3)
5. [Appendix: Benchmark Infrastructure](#appendix)

---

## 1. Overview: The Inference Optimization Hierarchy {#overview}

This document is the technical narrative and experiment log for the FP8 inference optimization experiments conducted on the GenomeOcean-4B Genomic Foundation Model (GFM) using NVIDIA H100 80GB hardware. It is organized as an introduction to the experiment design for the thesis methodology chapter.

**Central research question:** *How far can we push inference throughput of a large genomic foundation model through progressive precision reduction, without compromising biological representational fidelity?*

To answer this, we define a progressive hierarchy of three quantization protocols — from most conservative (maximum fidelity) to most aggressive (maximum throughput):

| Protocol | Name | Weights | KV Cache / Activations |
|----------|------|---------|------------------------|
| **B.1** | KV-Cache Quantization *(Storage-First)* | BFloat16 (W16) | FP8 (A8) |
| **B.2** | W8A8 Dynamic Quantization *(Compute-First)* | FP8 (W8), *lm_head*: **BFloat16** | FP8-Dynamic (A8) |
| **B.3** | W8A8 Full FP8 *(B.2 ablation — lm_head included)* | FP8 (W8), *lm_head*: **FP8** | FP8-Dynamic (A8) |

> **Note:** Protocol A refers to the PyTorch-native preprocessing experiments (embedding extraction and quality evaluation at full BFloat16 precision) which establish the baseline biological fidelity metrics. These are discussed in the preceding experimental chapter.

---

## 2. Protocol B.1: KV-Cache Quantization (Storage-First) {#b1}

### Intuition and Motivation

During autoregressive genomic sequence generation, the bottleneck is not raw compute but **memory bandwidth**. Each newly generated token requires the GPU to reload the entire accumulated Key-Value (KV) cache — which encodes all prior biological context — from High-Bandwidth Memory (HBM). For a 4B parameter model generating long microbial genome sequences (up to 10,240 tokens ≈ 51 kbp), this KV cache can occupy tens of gigabytes.

The core insight of Protocol B.1 is that the *storage format* of the KV cache can be **decoupled from the compute precision** of the attention mechanism. By instructing vLLM to store attention keys and values in FP8-E4M3 format, we compress the KV cache from 16-bit to 8-bit — halving data volume without modifying model weights at all. The attention dot-product math still runs in BFloat16 (8-bit tensors are dequantized on-the-fly), making this a **"Storage-First"** optimization: bandwidth savings with no 8-bit tensor core engagement.

This is intentionally the **most conservative protocol**: weights stay BFloat16, compute stays full precision, and biological fidelity risk is minimal.

### Why This Protocol?

Protocol B.1 establishes the **baseline for zero-model-modification throughput gains**. It answers: *"What is achievable just by compressing memory traffic, without changing the model itself?"*

This is the pragmatically safest starting point where biological fidelity must be verifiable and where taxonomically incorrect sequences would undermine the entire evaluation pipeline.

### Benchmark Design

The benchmark runs in two phases:

1. **Phase A — Biological Fidelity (Quality):** Sliding-window perplexity over 96 randomly sampled microbial genomes (*N* = 96). Each genome is chunked with a 10,240-token context window (≈ 51 kbp) and 2,560-token stride. NLL is averaged across all windows per genome. This measures whether FP8 KV storage alters the model's prediction distribution.

2. **Phase B — Efficiency (Throughput):** Saturated single-request throughput. All 96 genomes receive a 10,240-token prefix and the model generates 2,560 tokens each. The full workload (*N × K* = 96 × 5 = 480 total sequences) is submitted in one call. Total generation time ÷ total tokens = honest steady-state throughput.

**Measurement integrity choices:**
- **Prefix caching disabled:** `enable_prefix_caching=False` — prevents KV cache hits from inflating throughput artificially.
- **Warmup separation:** A single-sequence generation run precedes the timed call, burning in CUDA graphs and JIT kernels.
- **Saturated workload:** Maximum batch size used per mode (B.1: 96, BF16 baseline: 48).

---

## 3. Protocol B.2: W8A8 Dynamic Quantization (Compute-First) {#b2}

### Intuition and Motivation

Protocol B.1 is a "Storage Victory": it saves bandwidth but does **not engage the H100's SM90 FP8 tensor cores**, which can perform fused 8-bit matrix multiplications at up to ~989 TFLOPS — roughly 2× the throughput of BFloat16. To access this hardware capability, the model's **weights themselves** must be quantized to FP8.

Protocol B.2 applies **W8A8 Post-Training Quantization** via `llm-compressor` using the `FP8_DYNAMIC` scheme:
- **Weights:** Quantized *statically* per-channel in FP8-E4M3. Scaling factors computed once from the weight tensors and saved into the checkpoint (one-time offline operation).
- **Activations:** Quantized *dynamically* during inference per-token. Scaling factor recomputed from observed min/max at runtime — ensuring accuracy for unseen biological sequences.

### Why Dynamic Activation Quantization? (The Calibration Problem)

Static activation quantization requires a **calibration dataset**: a small representative set passed through the model to observe activation distributions, from which scaling factors are fixed.

For a GFM trained on a vast and taxonomically diverse metagenomic corpus, composing such a dataset is non-trivial:

- **Taxonomic coverage:** A calibration set overrepresenting common phyla (e.g., *Proteobacteria*) would produce poorly calibrated scaling factors for rare taxa — causing outsized quantization error on precisely the biological sequences the model must generalize to.
- **Sequence diversity:** Genomic sequences exhibit extreme compositional heterogeneity (GC content, codon bias, repeat structures) across organisms. A small calibration set cannot capture this diversity.

By using **dynamic scaling**, we eliminate the calibration problem entirely. Each forward pass re-evaluates its own activation range, making the system robust to any unseen species or sequence composition. This makes Protocol B.2 a **"data-free"** quantization approach.

### Why Exclude the `lm_head`?

The `lm_head` is the final linear projection mapping the model's 3072-dimensional hidden state to the vocabulary logit space (the probability distribution over the next biological token: A, C, G, T, etc.). It directly determines the biological "grammar" of generated sequences.

Because the GFM vocabulary encodes subtle evolutionary probabilities (e.g., codon usage bias in highly conserved gene regions), quantizing `lm_head` to 8-bit introduces rounding noise into the logit space, potentially biasing generation toward "common" sequences and away from rare but biologically meaningful ones.

Since `lm_head` accounts for **< 1% of total parameter count**, keeping it in BFloat16 incurs essentially zero performance penalty while preserving the high dynamic range required for accurate biological synthesis.

### Benchmark Design

Phase A and B methodology is identical to Protocol B.1. The key difference is the model loaded:

- **B.1 engine:** Loads the raw `DOEJGI/GenomeOcean-4B` BFloat16 weights with FP8 KV cache storage.
- **B.2 engine:** Loads a locally quantized checkpoint (`GenomeOcean-4B-B2-FP8-W8A8`) with FP8 KV cache storage. vLLM detects the `"quantization": "fp8"` field in `config.json` and automatically routes matrix multiplications through FP8 GEMM kernels.

Quantization is performed offline via `quantize_genomeocean_fp8.py`, and the resulting checkpoint is passed to the benchmark runner via the `--model-w8a8` flag with `--quant-modes fp8_v2`.

---

## 4. Protocol B.3: Full FP8 (Zero 16-bit Components) {#b3}

### Intuition and Motivation

Protocols B.1 and B.2 both retain at least one BFloat16 component. Protocol B.3 asks the **maximal ablation question:** *What happens to efficiency and biological fidelity if we eliminate all 16-bit components entirely?*

This protocol is not intended to achieve "best" biological fidelity — by definition it will be worse than B.2. It is an **ablation study**: by measuring the marginal fidelity cost of quantizing the `lm_head`, we quantify how much biological precision is traded away for maximum throughput, providing a principled bound on the "compute-first" optimization frontier.

### Design

Identical to B.2, with one change: `lm_head` is **included** in the quantization recipe (`ignore=[]`). The quantization script automatically produces this checkpoint alongside the B.2 checkpoint in the same run, named `GenomeOcean-4B-B3-FP8-W8A8-FullFP8`. The benchmark is run using the `--model-w8a8-full` flag with `--quant-modes fp8_v3`.

### ⚠️ Discovered Limitation: vLLM Does Not Support a Quantized `lm_head`

Protocol B.3 was attempted on H100 hardware but **fails to load** with the current version of vLLM. The root cause is a software-layer constraint in vLLM's `LlamaForCausalLM` model implementation:

When `llm-compressor` quantizes the `lm_head`, it saves a `lm_head.weight_scale` tensor into the checkpoint. However, vLLM's internal Llama model class does not define a `weight_scale` parameter for the `lm_head` layer — it is treated as a standard BFloat16 linear layer regardless of the checkpoint's quantization config. On load, vLLM encounters the unexpected key and raises:

> `ValueError: There is no module or parameter named 'lm_head.weight_scale' in LlamaForCausalLM`

**Why B.2 succeeds:** Because B.2 excludes `lm_head` from quantization, no `weight_scale` is saved for it, and vLLM loads the checkpoint cleanly.

**Implication for the thesis:** Protocol B.3 is quantization-theoretically valid and its checkpoint can be produced correctly by `llm-compressor`. The blocker is a **software limitation in the vLLM inference backend**, not a hardware or mathematical constraint. This represents another instance of the broader theme in this thesis: the gap between theoretical FP8 capability and practical deployability in current HPC software stacks. Protocol B.3 remains as a defined protocol for future evaluation pending vLLM support for a quantized `lm_head`.

---

## 5. Appendix: Benchmark Infrastructure {#appendix}

### Hardware and Software Stack

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA H100 80GB SXM (SM90, HBM3, 3.35 TB/s) |
| Inference Engine | vLLM (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`) |
| Attention Backend | FlashAttention-3 (FA3, for FP8 KV layout support) |
| Quantization Tool | `llm-compressor` |
| Model | `DOEJGI/GenomeOcean-4B` — Transformer Decoder, 4B params, GQA (8 KV heads, 40 attn heads), 40 layers, hidden dim 3072 |

### Why H100 is Architecturally Required

The H100 is necessary — not just preferred:

1. **HBM3 Bandwidth (3.35 TB/s):** Must stream the large KV caches required by Protocol B's batch sizes. Lower-bandwidth GPUs (e.g., A100 at 2 TB/s) would further exacerbate the bandwidth bottleneck being measured.
2. **SM90 FP8 Tensor Core Support:** Only SM90 (Hopper) and Ada Lovelace GPUs have native 8-bit GEMM hardware. Protocols B.2 and B.3 are designed to engage these cores. On A100, FP8 GEMM kernels would fall back to emulation.
3. **PagedAttention + FP8 KV Descriptor Support:** The vLLM/FlashAttention-3 FP8 memory layout requires hardware-level support for 8-bit cache descriptors in the memory hierarchy, which is native only to H100.

### Known Software Limitation: Protocol B.3 Blocked by vLLM `lm_head` Constraint

vLLM's `LlamaForCausalLM` implementation does not define a `weight_scale` parameter for the `lm_head` layer. When a checkpoint quantized with `ignore=[]` (i.e., `lm_head` included) is loaded, vLLM raises a `ValueError` on the unexpected `lm_head.weight_scale` key. This is not a quantization error or a hardware limitation — the B.3 checkpoint is mathematically valid. It is a software-layer gap between the `llm-compressor` checkpoint format and vLLM's current model loader. Protocol B.3 is therefore catalogued as a **theoretical protocol pending vLLM backend support**.
