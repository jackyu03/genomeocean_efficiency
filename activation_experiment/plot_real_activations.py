import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from tqdm import tqdm

def quantize_int8_per_layer(activations_f32: torch.Tensor):
    """
    Simulate per-layer dynamic INT8 (Absmax).
    Each layer gets its OWN scale based on its own max value.
    This is what real W8A8 inference engines (bitsandbytes, vLLM) actually do.
    """
    scale = activations_f32.abs().max() / 127.0
    q = torch.clamp(torch.round(activations_f32 / scale), -127, 127)
    dq = (q * scale).to(torch.bfloat16)
    return dq, scale

def quantize_fp8_per_layer(activations_f32: torch.Tensor):
    """
    Simulate per-layer dynamic FP8 (E4M3) with proper dynamic scaling.
    Scale stretches activations to fill E4M3's max representable value (448.0).
    """
    fp8_max_val = 448.0
    scale = fp8_max_val / activations_f32.abs().max()
    scaled = activations_f32 * scale
    try:
        fp8_q = scaled.to(torch.float8_e4m3fn)
        dq = (fp8_q.to(torch.float32) / scale).to(torch.bfloat16)
        if dq.isnan().any() or dq.isinf().any():
            dq = activations_f32.to(torch.bfloat16)
    except AttributeError:
        print("Warning: torch.float8_e4m3fn not available. Using BF16 fallback for FP8.")
        dq = activations_f32.to(torch.bfloat16)
    return dq, scale

def plot_layer(activations: torch.Tensor, layer_idx: int, outdir: Path):
    """Generate and save the 3-panel comparison plot for a single layer."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)

    # Subsample if huge (>5M values) for speed
    if len(activations) > 5_000_000:
        indices = torch.randperm(len(activations))[:5_000_000]
        activations = activations[indices]

    activations_f32 = activations.float()

    # Per-layer quantization (the scientifically correct simulation)
    int8_dq, int8_scale = quantize_int8_per_layer(activations_f32)
    fp8_dq, fp8_scale   = quantize_fp8_per_layer(activations_f32)

    # Per-layer errors
    int8_err = (activations_f32 - int8_dq.float()).abs()
    fp8_err  = torch.nan_to_num(
        (activations_f32 - fp8_dq.float()).abs(), nan=0.0, posinf=0.0, neginf=0.0
    )

    # Bin range: 99.99th percentile to keep axis tight
    plot_max = torch.quantile(activations_f32.abs(), 0.9999).item() * 1.5
    bins = np.linspace(-plot_max, plot_max, 150)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    label = "Embedding Layer" if layer_idx == 0 else f"Transformer Layer {layer_idx}"

    # Panel 1: BF16
    axes[0].hist(activations_f32.numpy(), bins=bins, color='gray', density=True, log=True)
    axes[0].set_title(f"{label} — BFloat16 Activations (Log Scale)", fontweight='bold')

    # Panel 2: INT8
    axes[1].hist(int8_dq.float().numpy(), bins=bins, color='orange', density=True, log=True)
    axes[1].set_title(f"INT8 (Per-Layer Absmax Scale: {int8_scale:.4f})", fontweight='bold')
    axes[1].text(0.05, 0.85, f"MAE: {int8_err.mean():.5f}", transform=axes[1].transAxes, fontsize=10)

    # Panel 3: FP8
    axes[2].hist(fp8_dq.float().numpy(), bins=bins, color='green', density=True, log=True)
    axes[2].set_title(f"FP8 E4M3 (Per-Layer Scale: {1/fp8_scale:.4f})", fontweight='bold')
    axes[2].text(0.05, 0.85, f"MAE: {fp8_err.mean():.5f}", transform=axes[2].transAxes, fontsize=10)

    plt.xlabel("Activation Value")
    plt.tight_layout()

    fname = outdir / f"layer_{layer_idx:03d}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return int8_err.mean().item(), fp8_err.mean().item()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot per-layer quantization distributions")
    parser.add_argument("--indir",  type=str, default="layer_activations", help="Dir with layer_*.pt files from extract_real_activations.py")
    parser.add_argument("--outdir", type=str, default="layer_plots",       help="Output dir for per-layer PNG plots")
    args = parser.parse_args()

    indir  = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    layer_files = sorted(indir.glob("layer_*.pt"))
    if not layer_files:
        print(f"No layer_*.pt files found in '{indir}/'. Run extract_real_activations.py first!")
        return

    print(f"Found {len(layer_files)} layer files. Plotting per-layer quantization...")

    all_int8_mae = []
    all_fp8_mae  = []

    for fpath in tqdm(layer_files, desc="Plotting layers"):
        layer_idx = int(fpath.stem.split("_")[1])
        activations = torch.load(fpath, weights_only=False)
        int8_mae, fp8_mae = plot_layer(activations, layer_idx, outdir)
        all_int8_mae.append(int8_mae)
        all_fp8_mae.append(fp8_mae)

    # Summary plot: MAE per layer
    print("Generating MAE summary plot...")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(all_int8_mae))
    ax.plot(x, all_int8_mae, color='orange', marker='o', markersize=4, label='INT8 MAE')
    ax.plot(x, all_fp8_mae,  color='green',  marker='s', markersize=4, label='FP8 E4M3 MAE')
    ax.set_xlabel("Layer Index (0 = Embedding)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Per-Layer Quantization Error: INT8 vs FP8 E4M3 (GenomeOcean-100M)", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    summary_path = outdir / "summary_mae_per_layer.png"
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll done! Saved {len(layer_files)} layer plots + summary to '{outdir}/'")
    print(f"Average INT8 MAE across all layers: {np.mean(all_int8_mae):.5f}")
    print(f"Average FP8  MAE across all layers: {np.mean(all_fp8_mae):.5f}")

if __name__ == "__main__":
    main()
