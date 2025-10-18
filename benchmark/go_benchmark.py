
#!/usr/bin/env python3
"""
go_benchmark.py — GPU-agnostic GenomeOcean baseline benchmark

Usage (example):
  python go_benchmark.py \
    --csv /path/to/gtdb.csv \
    --model DOEJGI/GenomeOcean-100M \
    --device cuda \
    --precision float16 \
    --samples-per-bin 1000 \
    --bins 1000 5000 10000 50000 \
    --batch-sizes 1 4 8 16 \
    --warmup 3 \
    --outdir ./results

Expected CSV columns: genome_id, seq  (seq capped at ~50k bp as you described)
"""

import os
import sys
import csv
import time
import math
import json
import argparse
import threading
import queue
from datetime import datetime
from typing import List, Tuple, Dict

# Third-party (install as needed): torch, transformers, pandas, numpy, pynvml, tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn

try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from transformers import AutoTokenizer, AutoModel, AutoConfig

# -----------------------------
# Logging utils
# -----------------------------

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("go_bench")


# -----------------------------
# NVML energy meter
# -----------------------------

class EnergyMeter:
    """
    Context manager that samples GPU power (mW) at a fixed interval
    and integrates to energy (kWh). Works best on NVIDIA GPUs with NVML.
    """
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.kwh = None
        self.ok = NVML_AVAILABLE

    def _runner(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop.is_set():
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            except Exception:
                p_mw = 0
            ts = time.perf_counter()
            self.samples.append((ts, p_mw))
            time.sleep(self.interval)

    def __enter__(self):
        if self.ok:
            try:
                pynvml.nvmlInit()
                self._thread = threading.Thread(target=self._runner, daemon=True)
                self._thread.start()
            except Exception as e:
                log.warning(f"NVML init failed: {e}")
                self.ok = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.ok:
            self._stop.set()
            self._thread.join(timeout=1.0)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            # integrate energy via trapezoidal rule (power in W, time in s) → Joules → kWh
            if len(self.samples) >= 2:
                joules = 0.0
                for (t0, p0_mw), (t1, p1_mw) in zip(self.samples, self.samples[1:]):
                    dt = (t1 - t0)
                    p0 = p0_mw / 1000.0  # W
                    p1 = p1_mw / 1000.0  # W
                    joules += 0.5 * (p0 + p1) * dt
                self.kwh = joules / 3_600_000.0  # 1 kWh = 3.6e6 Joules
            else:
                self.kwh = None


# -----------------------------
# FLOPs per sequence (analytical)
# -----------------------------

def flops_per_seq_decoder(L:int, d:int, T:int, d_ff:int=None) -> float:
    """
    Analytical forward FLOPs for a decoder-only transformer pass on length T.
    Assumes full attention (no cache) and standard MLP with expansion m = d_ff/d.

    FLOPs_seq ≈ L * [ (4 + 4m) * T * d^2 + 2 * T^2 * d ]

    Returns: FLOPs (not TFLOPs) as a float
    """
    if d_ff is None:
        # fallback m ≈ 4
        m = 4.0
    else:
        m = float(d_ff) / float(d)

    term_proj = (4.0 + 4.0*m) * T * (d**2)
    term_attn = 2.0 * (T**2) * d
    flops = L * (term_proj + term_attn)
    return float(flops)


# -----------------------------
# Tokenization helper (DNA → tokens len)
# -----------------------------

def tokenize_lengths(tokenizer, seqs: List[str], batch_size: int = 128) -> List[int]:
    lens = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_length=True
        )
        # HF returns 'length' as number of tokens *including* special tokens
        # We use that directly as T
        lens.extend([int(x) for x in enc["length"]])
    return lens


# -----------------------------
# Binning by token length
# -----------------------------

def make_bins_by_bp(df: pd.DataFrame, bins_bp: List[int], max_per_bin: int,
                    tokenizer, sample_seed: int = 42) -> Dict[int, pd.DataFrame]:
    """
    bins_bp: list of approximate bp targets (e.g., [1000, 5000, 10000, 50000])
    We compute token lengths, then create bins by nearest bp target.
    """
    # Estimate bp per token by sampling a small subset
    samp = df.sample(min(len(df), 1000), random_state=sample_seed)
    samp_tokens = tokenize_lengths(tokenizer, samp["seq"].tolist(), batch_size=64)
    mean_bp_per_token = np.mean([len(s)/max(t,1) for s, t in zip(samp["seq"], samp_tokens)])

    log.info(f"Estimated ~{mean_bp_per_token:.2f} bp / token (from 1k-sample).")
    # Now compute tokens for all (may be large; chunked)
    lens_tokens = tokenize_lengths(tokenizer, df["seq"].tolist(), batch_size=128)
    df = df.copy()
    df["len_tokens"] = lens_tokens
    df["len_bp"] = df["seq"].str.len()

    # Assign each row to closest target bin by bp length
    bins_map = {b: [] for b in bins_bp}
    for idx, row in df.iterrows():
        bp = row["len_bp"]
        target = min(bins_bp, key=lambda b: abs(bp - b))
        bins_map[target].append(idx)

    out = {}
    rng = np.random.default_rng(seed=sample_seed)
    for b in bins_bp:
        idxs = bins_map[b]
        if len(idxs) == 0:
            log.warning(f"No sequences mapped to bin ~{b} bp")
            continue
        if len(idxs) > max_per_bin:
            idxs = rng.choice(idxs, size=max_per_bin, replace=False)
        sub = df.loc[idxs].copy()
        out[b] = sub
        log.info(f"Bin ~{b} bp: {len(sub)} sequences (mean bp={sub['len_bp'].mean():.0f}, mean tokens={sub['len_tokens'].mean():.0f})")
    return out


# -----------------------------
# Inference pass to get embeddings
# -----------------------------

@torch.no_grad()
def embed_batch(model: nn.Module, tokenizer, seqs: List[str], device: str) -> torch.Tensor:
    """
    Forward pass to obtain a pooled embedding per sequence.
    We take last hidden states and mean-pool across tokens.
    """
    inputs = tokenizer(
        seqs, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    if hasattr(outputs, "last_hidden_state"):
        hs = outputs.last_hidden_state  # [B, T, d]
    elif hasattr(outputs, "hidden_states"):
        hs = outputs.hidden_states[-1]
    else:
        raise RuntimeError("Model output missing hidden states")
    emb = hs.mean(dim=1)  # [B, d]
    return emb


# -----------------------------
# Benchmark loop
# -----------------------------

def run_benchmark(args):
    # Env and seed hygiene
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = args.device
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.precision]

    # Load data
    log.info(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if not {"genome_id", "seq"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: genome_id, seq")
    log.info(f"Loaded {len(df):,} sequences.")

    # Load tokenizer & model
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    log.info(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    # Report model config excerpt
    n_params = sum(p.numel() for p in model.parameters())
    d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    n_layer = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    d_ff = getattr(config, "intermediate_size", None)
    log.info(f"Model loaded: params={n_params/1e6:.1f}M, hidden={d_model}, layers={n_layer}, d_ff={d_ff}, dtype={dtype}, device={device}")

    # Make bins
    bins_bp = [int(x) for x in args.bins]
    bins = make_bins_by_bp(df, bins_bp, args.samples_per_bin, tokenizer, sample_seed=42)

    # Prepare output
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(args.outdir, f"benchmark_{timestamp}.csv")
    fieldnames = [
        "model","commit","gpu_name","driver","cuda","torch","precision",
        "bin_bp","mean_bp","mean_tokens","batch_size",
        "runs","warmup","E2EL_ms","seqs_per_s","tokens_per_s",
        "peak_vram_GB","energy_kWh","flops_per_seq","TFLOPs_per_s"
    ]

    # System info
    commit = os.environ.get("GIT_COMMIT", "")
    gpu_name = driver = cuda = ""
    if torch.cuda.is_available() and device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        driver = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip() or ""
        cuda = torch.version.cuda or ""

    # CSV writer
    f_out = open(out_csv, "w", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    try:
        for bin_bp, sub in bins.items():
            seqs = sub["seq"].tolist()
            lens_tokens = sub["len_tokens"].tolist()
            mean_tokens = int(np.mean(lens_tokens))
            mean_bp = int(sub["len_bp"].mean())

            # FLOPs per seq (analytical) for mean token length
            if (n_layer is not None) and (d_model is not None):
                flops_seq = flops_per_seq_decoder(n_layer, d_model, mean_tokens, d_ff)
            else:
                flops_seq = float("nan")

            # Build batches
            # Shuffle for fairness
            idx = np.arange(len(seqs))
            np.random.shuffle(idx)
            seqs = [seqs[i] for i in idx]

            for bs in args.batch_sizes:
                # warmup
                warm = max(0, args.warmup)
                log.info(f"[Bin ~{bin_bp} bp | mean tokens ~{mean_tokens} | BS={bs}] Warmup x{warm} ...")
                for _ in range(warm):
                    sample = seqs[:bs]
                    _ = embed_batch(model, tokenizer, sample, device=device)
                    torch.cuda.synchronize() if device == "cuda" else None

                # timed runs
                n_runs = max(1, math.floor(len(seqs) / bs))
                if args.max_batches is not None:
                    n_runs = min(n_runs, args.max_batches)
                log.info(f"[Bin ~{bin_bp} bp | BS={bs}] Timed runs: {n_runs}")

                # Reset peak memory
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                with EnergyMeter(gpu_index=0, interval_s=0.05) as em:
                    n_tokens_total = 0
                    n_seqs_total = 0
                    for i in range(n_runs):
                        batch = seqs[i*bs:(i+1)*bs]
                        t0 = time.perf_counter()
                        _ = embed_batch(model, tokenizer, batch, device=device)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        # accumulate counts
                        toks = tokenize_lengths(tokenizer, batch, batch_size=bs)
                        n_tokens_total += sum(toks)
                        n_seqs_total += len(batch)
                end = time.perf_counter()

                elapsed = (end - start)
                e2el_ms = (elapsed * 1000.0) / max(1, n_runs)

                seqs_per_s = n_seqs_total / max(elapsed, 1e-9)
                tokens_per_s = n_tokens_total / max(elapsed, 1e-9)

                peak_vram_gb = None
                if device == "cuda":
                    peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9

                # TFLOPs/s based on analytical FLOPs per seq × seqs/s
                if not math.isnan(flops_seq):
                    tflops_per_s = (flops_seq * seqs_per_s) / 1e12
                else:
                    tflops_per_s = float("nan")

                row = {
                    "model": args.model,
                    "commit": commit,
                    "gpu_name": gpu_name,
                    "driver": driver,
                    "cuda": cuda,
                    "torch": torch.__version__,
                    "precision": args.precision,
                    "bin_bp": bin_bp,
                    "mean_bp": mean_bp,
                    "mean_tokens": mean_tokens,
                    "batch_size": bs,
                    "runs": n_runs,
                    "warmup": warm,
                    "E2EL_ms": round(e2el_ms, 3),
                    "seqs_per_s": round(seqs_per_s, 4),
                    "tokens_per_s": round(tokens_per_s, 2),
                    "peak_vram_GB": None if peak_vram_gb is None else round(peak_vram_gb, 3),
                    "energy_kWh": None if (not NVML_AVAILABLE or (em.kwh is None)) else round(em.kwh, 6),
                    "flops_per_seq": None if math.isnan(flops_seq) else int(flops_seq),
                    "TFLOPs_per_s": None if math.isnan(tflops_per_s) else round(tflops_per_s, 3),
                }
                writer.writerow(row)
                f_out.flush()
                log.info(
                    f"[RESULT] bin~{bin_bp}bp, BS={bs} | "
                    f"E2EL={row['E2EL_ms']} ms | seq/s={row['seqs_per_s']} | tok/s={row['tokens_per_s']} | "
                    f"VRAM={row['peak_vram_GB']} GB | kWh={row['energy_kWh']} | TFLOPs/s={row['TFLOPs_per_s']}"
                )

    finally:
        f_out.close()
        log.info(f"Saved results CSV → {out_csv}")
        # Save a small run metadata JSON
        meta = {
            "model": args.model,
            "precision": args.precision,
            "bins_bp": bins_bp,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "device": args.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "timestamp": timestamp
        }
        meta_path = os.path.join(args.outdir, f"benchmark_{timestamp}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"Saved meta JSON → {meta_path}")


def parse_args():
    p = argparse.ArgumentParser(description="GenomeOcean baseline benchmark (GPU-agnostic)")
    p.add_argument("--csv", type=str, required=True, help="GTDB CSV with columns: genome_id, seq")
    p.add_argument("--model", type=str, required=True, help="HF model id (e.g., DOEJGI/GenomeOcean-4B)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Device")
    p.add_argument("--precision", type=str, default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--samples-per-bin", type=int, default=1000, help="Max sequences per bin")
    p.add_argument("--bins", type=int, nargs="+", default=[1000, 5000, 10000, 50000], help="Approx. bp bin centers")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1,4,8,16], help="Batch sizes to test")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations per (bin, batch)")
    p.add_argument("--max-batches", type=int, default=None, help="Optional cap on timed batches per (bin, batch)")
    p.add_argument("--outdir", type=str, default="./results", help="Output directory for CSV/JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Benchmark failed: {e}")
        sys.exit(2)
