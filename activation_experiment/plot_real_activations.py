import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_real_quantization(tensor_path="real_activations.pt"):
    if not os.path.exists(tensor_path):
        print(f"Error: Could not find {tensor_path}. Please run extract_real_activations.py first!")
        return

    print(f"Loading activations from {tensor_path}...")
    activations = torch.load(tensor_path)
    
    # Optional: if the tensor is extremely huge (e.g. > 10M values), we can subsample it for faster plotting
    # but 15M floats usually plots fine in a histogram
    if len(activations) > 5_000_000:
        print(f"Tensor is very large ({len(activations)} values). Subsampling to 5M for plot rendering speed...")
        indices = torch.randperm(len(activations))[:5_000_000]
        activations = activations[indices]

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    # We do quantization math in float32 to simulate standard quantizers, then cast back to bfloat16
    activations_f32 = activations.float()
    
    # 2. INT8 Quantization (Absmax - No Clipping)
    int8_max_scale = activations_f32.abs().max() / 127.0
    int8_absmax_q = torch.clamp(torch.round(activations_f32 / int8_max_scale), -127, 127)
    int8_absmax_dq = (int8_absmax_q * int8_max_scale).to(torch.bfloat16)
    
    # 3. INT8 Quantization (99.9th Percentile - Clipping)
    clip_val = torch.quantile(activations_f32.abs(), 0.999)
    int8_clip_scale = clip_val / 127.0
    int8_clip_q = torch.clamp(torch.round(activations_f32 / int8_clip_scale), -127, 127)
    int8_clip_dq = (int8_clip_q * int8_clip_scale).to(torch.bfloat16)
    
    # 4. FP8 Quantization (E4M3)
    try:
        # For PyTorch >= 2.1
        fp8_q = activations.to(torch.float8_e4m3fn)
        fp8_dq = fp8_q.to(torch.bfloat16)
        
        # Check if Mac CPU emulation generated NaNs
        if fp8_dq.isnan().any() or fp8_dq.isinf().any():
            print("Warning: Mac CPU produced NaNs during float8 conversion. Falling back to simulated.")
            fp8_dq = activations
    except AttributeError:
        print("Warning: torch.float8_e4m3fn not found. Please run on an environment with PyTorch >= 2.1")
        print("Falling back to a simulated E4M3 for plotting purposes.")
        fp8_dq = activations
        
    # Calculate errors against original BFloat16 values
    int8_absmax_err = (activations.float() - int8_absmax_dq.float()).abs()
    int8_clip_err = (activations.float() - int8_clip_dq.float()).abs()
    fp8_err = (activations.float() - fp8_dq.float()).abs()
    
    # Clean up NaNs from errs if any
    fp8_err = torch.nan_to_num(fp8_err, nan=0.0, posinf=0.0, neginf=0.0)
    
    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Calculate a reasonable bin range based on the 99.99th percentile to avoid giant empty tails stretching the plot
    plot_max = torch.quantile(activations_f32.abs(), 0.9999).item() * 1.5
    # If the true max is much larger, ensure we still see the outlier clip line
    plot_max = max(plot_max, clip_val.item() * 1.5)
    bins = np.linspace(-plot_max, plot_max, 150)
    
    # Plot 1: Original
    axes[0].hist(activations.float().numpy(), bins=bins, color='gray', density=True, log=True)
    axes[0].set_title(f"Real GenomeOcean BFloat16 Activations (Log Scale)", fontweight='bold')
    
    # Plot 2: INT8 Absmax
    axes[1].hist(int8_absmax_dq.float().numpy(), bins=bins, color='orange', density=True, log=True)
    axes[1].set_title(f"INT8 (Absmax Scale: {int8_max_scale:.3f}) - Precision loss near zero", fontweight='bold')
    axes[1].text(0.05, 0.8, f"Mean Abs Error: {int8_absmax_err.mean():.4f}", transform=axes[1].transAxes)
    
    # Plot 3: INT8 Clipped
    axes[2].hist(int8_clip_dq.float().numpy(), bins=bins, color='red', density=True, log=True)
    axes[2].set_title(f"INT8 (Clipped at {clip_val:.2f}, Scale: {int8_clip_scale:.3f}) - Outliers clipped", fontweight='bold')
    axes[2].axvline(clip_val, color='black', linestyle='--')
    axes[2].axvline(-clip_val, color='black', linestyle='--')
    axes[2].text(0.05, 0.8, f"Mean Abs Error: {int8_clip_err.mean():.4f}", transform=axes[2].transAxes)
    
    # Plot 4: FP8
    axes[3].hist(fp8_dq.float().numpy(), bins=bins, color='green', density=True, log=True)
    axes[3].set_title("FP8 (E4M3) - Natural dynamic range preserves tails", fontweight='bold')
    axes[3].text(0.05, 0.8, f"Mean Abs Error: {fp8_err.mean():.4f}", transform=axes[3].transAxes)
    
    plt.xlabel("Activation Value")
    plt.tight_layout()
    plt.savefig("real_activation_outliers.png", dpi=300, bbox_inches='tight')
    print("Saved plot to real_activation_outliers.png")

if __name__ == "__main__":
    plot_real_quantization()
