#!/usr/bin/env python3
"""
go_benchmark_v2.py — GenomeOcean baseline benchmark with:
- Hardware-agnostic FLOPs accounting using post-truncation tokens
- Energy logging (NVML) and TOKENS-PER-WATT metric
- Optional explicit truncation (--truncate-bp / --truncate-tokens)
- Robust per-combination error handling
- Clean CLI logs and CSV+JSON outputs
"""
import os, sys, csv, time, math, json, argparse, threading
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
from torch import nn

try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("go_bench_v2")

from transformers import AutoTokenizer, AutoModel, AutoConfig

# Import the model loader
from model_to_benchmark import load_model, get_model_info, get_max_length


class EnergyMeter:
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
                p_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
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
            if len(self.samples) >= 2:
                joules = 0.0
                for (t0, p0_mw), (t1, p1_mw) in zip(self.samples, self.samples[1:]):
                    dt = (t1 - t0)
                    p0 = p0_mw / 1000.0
                    p1 = p1_mw / 1000.0
                    joules += 0.5 * (p0 + p1) * dt
                self.kwh = joules / 3_600_000.0
            else:
                self.kwh = None


def flops_per_seq_decoder(L:int, d:int, T:int, d_ff:int=None) -> float:
    if T <= 0 or L <= 0 or d <= 0:
        return 0.0
    m = 4.0 if (d_ff is None) else float(d_ff)/float(d)
    term_proj = (4.0 + 4.0*m) * T * (d**2)
    term_attn = 2.0 * (T**2) * d
    return float(L) * (term_proj + term_attn)


def tokenize_lengths(tokenizer, seqs: List[str], batch_size: int = 128, max_len: int | None = None) -> List[int]:
    lens = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=False,
            truncation=(max_len is not None),
            max_length=max_len if (max_len is not None) else None,
            add_special_tokens=True,
            return_attention_mask=False,
            return_length=True
        )
        lens.extend([int(x) for x in enc["length"]])
    return lens


def truncate_by_bp(seq: str, target_bp: int) -> str:
    return seq[:max(0, min(len(seq), target_bp))] if target_bp is not None else seq


@torch.no_grad()
def embed_batch(model: nn.Module, tokenizer, seqs: List[str], device: str, max_len: int | None = None) -> torch.Tensor:
    inputs = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_len if (max_len is not None) else None,
        return_tensors="pt",
        add_special_tokens=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    inputs.pop('token_type_ids', None)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = getattr(outputs, "last_hidden_state", None)
    if hs is None:
        hs = outputs.hidden_states[-1]
    emb = hs.mean(dim=1)
    return emb


def run_benchmark(args):
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = args.device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.precision]

    log.info(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if not {"genome_id", "seq"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: genome_id, seq")
    log.info(f"Loaded {len(df):,} sequences.")

    log.info(f"Loading model and tokenizer: {args.model}")
    model, tokenizer, config = load_model(args.model, device, dtype)
    model.eval()

    # Determine max token length
    max_len = get_max_length(config, tokenizer)
    if max_len is not None:
        log.info(f"Using max_length={max_len} tokens for tokenizer truncation.")
    else:
        log.warning("No max_length found; proceeding without enforced tokenizer truncation.")

    # Get model architecture info
    model_info = get_model_info(config, model)
    n_params = model_info["n_params"]
    d_model = model_info["d_model"]
    n_layer = model_info["n_layer"]
    d_ff = model_info["d_ff"]
    log.info(f"Model loaded: params={n_params/1e6:.1f}M, hidden={d_model}, layers={n_layer}, d_ff={d_ff}, dtype={dtype}, device={device}")

    conditions: List[Dict] = []
    rng = np.random.default_rng(42)

    if args.truncate_bp or args.truncate_tokens:
        if args.truncate_bp and args.truncate_tokens:
            raise ValueError("Use only one of --truncate-bp or --truncate-tokens.")
        if args.truncate_bp:
            for tgt_bp in args.truncate_bp:
                seqs_trunc = [truncate_by_bp(s, tgt_bp) for s in df["seq"].tolist()]
                tok_lens = tokenize_lengths(tokenizer, seqs_trunc, batch_size=128, max_len=max_len)
                keep = [i for i, t in enumerate(tok_lens) if t > 0]
                if not keep:
                    log.warning(f"No sequences remain after bp truncation={tgt_bp}. Skipping.")
                    continue
                if len(keep) > args.samples_per_cond:
                    keep = list(rng.choice(keep, size=args.samples_per_cond, replace=False))
                seqs_cond = [seqs_trunc[i] for i in keep]
                tok_cond = [tok_lens[i] for i in keep]
                mean_bp = int(np.mean([len(x) for x in seqs_cond]))
                mean_tokens = int(np.mean(tok_cond))
                conditions.append(dict(label=f"bp={tgt_bp}", seqs=seqs_cond, tok_lens=tok_cond,
                                       mean_bp=mean_bp, mean_tokens=mean_tokens))
        else:
            for tgt_tok in args.truncate_tokens:
                eff_max_len = tgt_tok if (max_len is None) else min(tgt_tok, max_len)
                seqs_full = df["seq"].tolist()
                tok_lens = tokenize_lengths(tokenizer, seqs_full, batch_size=128, max_len=eff_max_len)
                keep = [i for i, t in enumerate(tok_lens) if t > 0]
                if not keep:
                    log.warning(f"No sequences remain after token truncation={tgt_tok}. Skipping.")
                    continue
                if len(keep) > args.samples_per_cond:
                    keep = list(rng.choice(keep, size=args.samples_per_cond, replace=False))
                seqs_cond = [seqs_full[i] for i in keep]
                tok_cond = [tok_lens[i] for i in keep]
                mean_bp = int(np.mean([len(x) for x in seqs_cond]))
                mean_tokens = int(np.mean(tok_cond))
                conditions.append(dict(label=f"tok={tgt_tok}", seqs=seqs_cond, tok_lens=tok_cond,
                                       mean_bp=mean_bp, mean_tokens=mean_tokens))
    else:
        bins_bp = [int(x) for x in args.bins]
        lens_tokens_all = tokenize_lengths(tokenizer, df["seq"].tolist(), batch_size=128, max_len=max_len)
        df = df.copy()
        df["len_bp"] = df["seq"].str.len()
        df["len_tokens"] = lens_tokens_all
        bins_map = {b: [] for b in bins_bp}
        for idx, row in df.iterrows():
            target = min(bins_bp, key=lambda b: abs(row["len_bp"] - b))
            bins_map[target].append(idx)
        for b in bins_bp:
            idxs = bins_map[b]
            if len(idxs) == 0:
                log.warning(f"No sequences mapped to bin ~{b} bp")
                continue
            if len(idxs) > args.samples_per_cond:
                idxs = list(rng.choice(idxs, size=args.samples_per_cond, replace=False))
            sub = df.loc[idxs]
            mean_bp = int(sub["len_bp"].mean())
            mean_tokens = int(sub["len_tokens"].mean())
            conditions.append(dict(label=f"bin~{b}", seqs=sub["seq"].tolist(),
                                   tok_lens=sub["len_tokens"].tolist(),
                                   mean_bp=mean_bp, mean_tokens=mean_tokens))

    if not conditions:
        raise RuntimeError("No valid conditions to run.")

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(args.outdir, f"benchmark_{timestamp}.csv")
    fieldnames = [
        "model","commit","gpu_name","driver","cuda","torch","precision",
        "condition","mean_bp","mean_tokens","batch_size",
        "runs","warmup","E2EL_ms","seqs_per_s","tokens_per_s",
        "peak_vram_GB","energy_kWh","avg_power_W",
        "flops_per_seq","TFLOPs_per_s",
        "eff_seq_per_TFLOP","eff_tokens_per_TFLOP","tokens_per_watt"
    ]

    commit = os.environ.get("GIT_COMMIT", "")
    gpu_name = driver = cuda = ""
    if torch.cuda.is_available() and device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        driver = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip() or ""
        cuda = torch.version.cuda or ""

    f_out = open(out_csv, "w", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    try:
        for cond in conditions:
            label = cond["label"]
            seqs_all = cond["seqs"]
            tok_lens_all = cond["tok_lens"]
            mean_bp = cond["mean_bp"]
            mean_tokens = cond["mean_tokens"]

            if (n_layer is not None) and (d_model is not None):
                flops_each = [flops_per_seq_decoder(int(n_layer), int(d_model), int(T), int(d_ff) if d_ff else None)
                              for T in tok_lens_all]
                flops_seq_mean = float(np.mean(flops_each)) if flops_each else float("nan")
            else:
                flops_seq_mean = float("nan")

            idx = np.arange(len(seqs_all))
            np.random.shuffle(idx)
            seqs_all = [seqs_all[i] for i in idx]

            for bs in args.batch_sizes:
                try:
                    warm = max(0, args.warmup)
                    log.info(f"[{label} | mean tokens ~{mean_tokens} | BS={bs}] Warmup x{warm} ...")
                    for _ in range(warm):
                        sample = seqs_all[:bs]
                        _ = embed_batch(model, tokenizer, sample, device=device, max_len=max_len)
                        if device == "cuda":
                            torch.cuda.synchronize()

                    n_runs = max(1, math.floor(len(seqs_all) / bs))
                    if args.max_batches is not None:
                        n_runs = min(n_runs, args.max_batches)
                    log.info(f"[{label} | BS={bs}] Timed runs: {n_runs}")

                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()

                    start = time.perf_counter()
                    with EnergyMeter(gpu_index=0, interval_s=0.05) as em:
                        n_tokens_total = 0
                        n_seqs_total = 0
                        for i in range(n_runs):
                            batch = seqs_all[i*bs:(i+1)*bs]
                            _ = embed_batch(model, tokenizer, batch, device=device, max_len=max_len)
                            if device == "cuda":
                                torch.cuda.synchronize()
                            toks = tokenize_lengths(tokenizer, batch, batch_size=bs, max_len=max_len)
                            n_tokens_total += sum(toks)
                            n_seqs_total += len(batch)
                    end = time.perf_counter()

                    elapsed = max(end - start, 1e-9)
                    e2el_ms = (elapsed * 1000.0) / max(1, n_runs)

                    seqs_per_s = n_seqs_total / elapsed
                    tokens_per_s = n_tokens_total / elapsed

                    peak_vram_gb = (torch.cuda.max_memory_reserved() / 1e9) if (device == "cuda") else None

                    tflops_per_s = (flops_seq_mean * seqs_per_s) / 1e12 if not math.isnan(flops_seq_mean) else float("nan")
                    eff_seq_per_TFLOP = (seqs_per_s / (flops_seq_mean / 1e12)) if (flops_seq_mean and not math.isnan(flops_seq_mean)) else None
                    eff_tokens_per_TFLOP = (tokens_per_s / (flops_seq_mean / 1e12)) if (flops_seq_mean and not math.isnan(flops_seq_mean)) else None

                    if NVML_AVAILABLE and (em.kwh is not None):
                        avg_power_W = (em.kwh * 3_600_000.0) / elapsed
                        tokens_per_watt = tokens_per_s / avg_power_W if avg_power_W > 0 else None
                        energy_kWh = em.kwh
                    else:
                        avg_power_W = None
                        tokens_per_watt = None
                        energy_kWh = None

                    row = {
                        "model": args.model,
                        "commit": commit,
                        "gpu_name": gpu_name,
                        "driver": driver,
                        "cuda": cuda,
                        "torch": torch.__version__,
                        "precision": args.precision,
                        "condition": label,
                        "mean_bp": mean_bp,
                        "mean_tokens": mean_tokens,
                        "batch_size": bs,
                        "runs": n_runs,
                        "warmup": warm,
                        "E2EL_ms": round(e2el_ms, 3),
                        "seqs_per_s": round(seqs_per_s, 4),
                        "tokens_per_s": round(tokens_per_s, 2),
                        "peak_vram_GB": None if peak_vram_gb is None else round(peak_vram_gb, 3),
                        "energy_kWh": None if energy_kWh is None else round(energy_kWh, 6),
                        "avg_power_W": None if avg_power_W is None else round(avg_power_W, 2),
                        "flops_per_seq": None if (flops_seq_mean is None or math.isnan(flops_seq_mean)) else int(flops_seq_mean),
                        "TFLOPs_per_s": None if (tflops_per_s is None or math.isnan(tflops_per_s)) else round(tflops_per_s, 3),
                        "eff_seq_per_TFLOP": None if (eff_seq_per_TFLOP is None) else round(eff_seq_per_TFLOP, 6),
                        "eff_tokens_per_TFLOP": None if (eff_tokens_per_TFLOP is None) else round(eff_tokens_per_TFLOP, 6),
                        "tokens_per_watt": None if (tokens_per_watt is None) else round(tokens_per_watt, 6),
                    }
                    writer.writerow(row); f_out.flush()
                    log.info(
                        f"[RESULT] {label}, BS={bs} | "
                        f"E2EL={row['E2EL_ms']} ms | seq/s={row['seqs_per_s']} | tok/s={row['tokens_per_s']} | "
                        f"VRAM={row['peak_vram_GB']} GB | kWh={row['energy_kWh']} | W={row['avg_power_W']} | "
                        f"TFLOPs/s={row['TFLOPs_per_s']} | tok/W={row['tokens_per_watt']}"
                    )
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg:
                        log.error(f"OOM at {label}, BS={bs}; skipping this combo.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        log.exception(f"Failure at {label}, BS={bs}; skipping this combo.")
                        continue

    finally:
        f_out.close()
        log.info(f"Saved results CSV → {out_csv}")
        meta = {
            "model": args.model,
            "precision": args.precision,
            "conditions": [c["label"] for c in conditions],
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
    p = argparse.ArgumentParser(description="GenomeOcean baseline benchmark (v2) with tokens-per-watt and truncation controls")
    p.add_argument("--csv", type=str, required=True, help="GTDB CSV with columns: genome_id, seq")
    p.add_argument("--model", type=str, required=True, help="HF model id (e.g., DOEJGI/GenomeOcean-4B)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Device")
    p.add_argument("--precision", type=str, default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--truncate-bp", type=int, nargs="+", default=None,
                   help="Use explicit bp truncation targets (e.g., 1000 2500 5000) instead of binning.")
    p.add_argument("--truncate-tokens", type=int, nargs="+", default=None,
                   help="Use explicit token truncation targets (mutually exclusive with --truncate-bp).")
    p.add_argument("--samples-per-cond", type=int, default=1000, help="Max sequences per condition (bin or truncation target).")
    p.add_argument("--bins", type=int, nargs="+", default=[1000, 5000, 10000], help="Approx. bp bin centers (legacy mode).")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1,4,8], help="Batch sizes to test")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations per condition × batch")
    p.add_argument("--max-batches", type=int, default=None, help="Optional cap on timed batches per condition × batch")
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
