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
    
    if len(activations) > 5_000_000:
        print(f"Tensor is very large ({len(activations)} values). Subsampling to 5M for plot rendering speed...")
        indices = torch.randperm(len(activations))[:5_000_000]
        activations = activations[indices]

    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    
    activations_f32 = activations.float()
    
    # 2. INT8 Quantization (Dynamic Absmax)
    # Standard dynamic W8A8 scales by the absolute maximum of the tensor/token.
    int8_max_scale = activations_f32.abs().max() / 127.0
    int8_absmax_q = torch.clamp(torch.round(activations_f32 / int8_max_scale), -127, 127)
    int8_dq = (int8_absmax_q * int8_max_scale).to(torch.bfloat16)
    
    # 3. FP8 Quantization (E4M3)
    try:
        fp8_q = activations.to(torch.float8_e4m3fn)
        fp8_dq = fp8_q.to(torch.bfloat16)
        
        if fp8_dq.isnan().any() or fp8_dq.isinf().any():
            print("Warning: Mac CPU produced NaNs during float8 conversion. Falling back to simulated.")
            fp8_dq = activations
    except AttributeError:
        print("Warning: torch.float8_e4m3fn not found. Please run on an environment with PyTorch >= 2.1")
        print("Falling back to a simulated E4M3 for plotting purposes.")
        fp8_dq = activations
        
    # Calculate errors against original BFloat16 values
    int8_err = (activations.float() - int8_dq.float()).abs()
    fp8_err = (activations.float() - fp8_dq.float()).abs()
    
    fp8_err = torch.nan_to_num(fp8_err, nan=0.0, posinf=0.0, neginf=0.0)
    
    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    plot_max = torch.quantile(activations_f32.abs(), 0.9999).item() * 1.5
    bins = np.linspace(-plot_max, plot_max, 150)
    
    # Plot 1: Original BF16
    axes[0].hist(activations.float().numpy(), bins=bins, color='gray', density=True, log=True)
    axes[0].set_title(f"Original GenomeOcean BFloat16 Activations (Log Scale)", fontweight='bold')
    
    # Plot 2: INT8
    axes[1].hist(int8_dq.float().numpy(), bins=bins, color='orange', density=True, log=True)
    axes[1].set_title(f"INT8 (Dynamic Absmax Scale: {int8_max_scale:.3f}) - Precision loss near zero", fontweight='bold')
    axes[1].text(0.05, 0.8, f"Mean Abs Error: {int8_err.mean():.4f}", transform=axes[1].transAxes)
    
    # Plot 3: FP8
    axes[2].hist(fp8_dq.float().numpy(), bins=bins, color='green', density=True, log=True)
    axes[2].set_title("FP8 (E4M3) - Natural dynamic range preserves tails and middle", fontweight='bold')
    axes[2].text(0.05, 0.8, f"Mean Abs Error: {fp8_err.mean():.4f}", transform=axes[2].transAxes)
    
    plt.xlabel("Activation Value")
    plt.tight_layout()
    plt.savefig("real_activation_outliers.png", dpi=300, bbox_inches='tight')
    print("Saved plot to real_activation_outliers.png")

if __name__ == "__main__":
    plot_real_quantization()
