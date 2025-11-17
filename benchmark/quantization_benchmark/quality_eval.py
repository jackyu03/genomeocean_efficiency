"""
Quantization Quality Evaluation Module

Evaluates the quality of quantized models by comparing output distributions,
computing KL divergence, cosine similarity, and other quality metrics.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

log = logging.getLogger(__name__)


def extract_layer_outputs(model, tokenizer, sequences: List[str], 
                          device: str, max_len: int = None,
                          layers: List[str] = ['last', 'second_last']) -> Dict[str, torch.Tensor]:
    """
    Extract outputs from specified layers of the model.
    Uses mean pooling over sequence dimension for each sequence.
    
    Args:
        model: The model to extract from
        tokenizer: Tokenizer for the model
        sequences: List of input sequences
        device: Device to run on
        max_len: Maximum sequence length
        layers: Which layers to extract ('last', 'second_last')
        
    Returns:
        Dictionary mapping layer names to output tensors of shape (num_sequences, hidden_dim)
    """
    
    model.eval()
    layer_outputs = {layer: [] for layer in layers}
    
    with torch.no_grad():
        for seq in sequences:
            # Only set truncation and max_length if max_len is provided
            tokenizer_kwargs = {
                "padding": True,
                "return_tensors": "pt"
            }
            if max_len is not None:
                tokenizer_kwargs["truncation"] = True
                tokenizer_kwargs["max_length"] = max_len
            
            inputs = tokenizer(seq, **tokenizer_kwargs)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            inputs.pop('token_type_ids', None)
            
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            
            # Extract requested layers and mean pool over sequence length
            # This gives us one vector per sequence (hidden_dim,)
            if 'last' in layers:
                # Mean pool over sequence length: (seq_len, hidden_dim) -> (hidden_dim,)
                pooled = hidden_states[-1].mean(dim=1).cpu()  # (1, hidden_dim)
                layer_outputs['last'].append(pooled)
            if 'second_last' in layers:
                pooled = hidden_states[-2].mean(dim=1).cpu()  # (1, hidden_dim)
                layer_outputs['second_last'].append(pooled)
    
    # Concatenate all outputs: list of (1, hidden_dim) -> (num_sequences, hidden_dim)
    for layer in layers:
        if layer_outputs[layer]:
            layer_outputs[layer] = torch.cat(layer_outputs[layer], dim=0)
    
    return layer_outputs


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two distributions.
    
    Args:
        p: Reference distribution
        q: Comparison distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        p: Reference distribution
        q: Comparison distribution
        
    Returns:
        JS divergence value
    """
    return jensenshannon(p, q) ** 2


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors/matrices.
    Uses normalized vectors to avoid overflow.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Cosine similarity value
    """
    a_flat = a.flatten().astype(np.float64)  # Use float64 for better precision
    b_flat = b.flatten().astype(np.float64)
    
    # Compute norms with overflow protection
    with np.errstate(over='ignore', invalid='ignore'):
        norm_a = np.sqrt(np.sum(a_flat ** 2))
        norm_b = np.sqrt(np.sum(b_flat ** 2))
    
    # Handle zero or invalid norms
    if not np.isfinite(norm_a) or not np.isfinite(norm_b) or norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Normalize vectors first to avoid overflow in dot product
    a_normalized = a_flat / norm_a
    b_normalized = b_flat / norm_b
    
    # Compute similarity on normalized vectors
    with np.errstate(over='ignore', invalid='ignore'):
        similarity = np.sum(a_normalized * b_normalized)
    
    # Handle invalid results
    if not np.isfinite(similarity):
        return 0.0
    
    # Clip to valid range [-1, 1]
    return float(np.clip(similarity, -1.0, 1.0))


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two tensors."""
    return np.mean((a - b) ** 2)


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Absolute Error between two tensors."""
    return np.mean(np.abs(a - b))


def compute_snr(reference: np.ndarray, quantized: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        reference: Reference (standard) outputs
        quantized: Quantized model outputs
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - quantized) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def analyze_distribution_statistics(outputs: np.ndarray) -> Dict:
    """
    Compute statistical properties of output distribution.
    
    Args:
        outputs: Model outputs as numpy array
        
    Returns:
        Dictionary of statistics
    """
    flat_outputs = outputs.flatten()
    
    return {
        'mean': float(np.mean(flat_outputs)),
        'std': float(np.std(flat_outputs)),
        'min': float(np.min(flat_outputs)),
        'max': float(np.max(flat_outputs)),
        'median': float(np.median(flat_outputs)),
        'q25': float(np.percentile(flat_outputs, 25)),
        'q75': float(np.percentile(flat_outputs, 75)),
        'skewness': float(stats.skew(flat_outputs)),
        'kurtosis': float(stats.kurtosis(flat_outputs))
    }


def compare_quantization_outputs(standard_outputs: Dict[str, torch.Tensor],
                                quantized_outputs: Dict[str, torch.Tensor],
                                quant_mode: str) -> Dict:
    """
    Compare outputs between standard and quantized models using pairwise comparisons.
    Computes metrics for each sequence individually, then averages.
    
    Args:
        standard_outputs: Outputs from standard (non-quantized) model, shape (num_sequences, hidden_dim)
        quantized_outputs: Outputs from quantized model, shape (num_sequences, hidden_dim)
        quant_mode: Name of quantization mode
        
    Returns:
        Dictionary of comparison metrics (averaged across sequences)
    """
    
    comparison = {
        'quantization_mode': quant_mode,
        'layers': {}
    }
    
    for layer_name in standard_outputs.keys():
        if layer_name not in quantized_outputs:
            continue
        
        std_out = standard_outputs[layer_name].numpy()  # (num_sequences, hidden_dim)
        quant_out = quantized_outputs[layer_name].numpy()  # (num_sequences, hidden_dim)
        
        # Ensure same shape
        if std_out.shape != quant_out.shape:
            log.warning(f"Shape mismatch for {layer_name}: {std_out.shape} vs {quant_out.shape}")
            continue
        
        # Pairwise comparisons: compute metrics for each sequence, then average
        num_sequences = std_out.shape[0]
        
        kl_divs = []
        js_divs = []
        cosine_sims = []
        mses = []
        maes = []
        snrs = []
        correlations = []
        
        for i in range(num_sequences):
            std_seq = std_out[i]  # (hidden_dim,)
            quant_seq = quant_out[i]  # (hidden_dim,)
            
            # Compute histogram-based divergence metrics for this sequence pair
            std_hist, bins = np.histogram(std_seq, bins=50, density=True)
            quant_hist, _ = np.histogram(quant_seq, bins=bins, density=True)
            
            # Normalize histograms
            std_hist = std_hist / (std_hist.sum() + 1e-10)
            quant_hist = quant_hist / (quant_hist.sum() + 1e-10)
            
            kl_divs.append(compute_kl_divergence(std_hist, quant_hist))
            js_divs.append(compute_js_divergence(std_hist, quant_hist))
            
            # Compute other similarity metrics
            cosine_sims.append(compute_cosine_similarity(std_seq, quant_seq))
            mses.append(compute_mse(std_seq, quant_seq))
            maes.append(compute_mae(std_seq, quant_seq))
            snrs.append(compute_snr(std_seq, quant_seq))
            
            # Compute correlation with error handling
            try:
                corr_matrix = np.corrcoef(std_seq, quant_seq)
                corr_value = corr_matrix[0, 1]
                # Handle NaN (can occur if std is zero)
                if np.isnan(corr_value):
                    corr_value = 0.0
                correlations.append(float(corr_value))
            except:
                correlations.append(0.0)
        
        # Average metrics across all sequences
        # Filter out inf/nan values before averaging
        finite_snrs = [s for s in snrs if np.isfinite(s)]
        finite_kl_divs = [k for k in kl_divs if np.isfinite(k)]
        finite_js_divs = [j for j in js_divs if np.isfinite(j)]
        
        avg_kl = float(np.mean(finite_kl_divs)) if finite_kl_divs else float('inf')
        avg_js = float(np.mean(finite_js_divs)) if finite_js_divs else float('inf')
        avg_cosine = float(np.mean(cosine_sims))
        avg_mse = float(np.mean(mses))
        avg_mae = float(np.mean(maes))
        avg_snr = float(np.mean(finite_snrs)) if finite_snrs else float('inf')
        avg_corr = float(np.mean(correlations))
        
        layer_metrics = {
            # All metrics are pairwise (averaged across sequences)
            'kl_divergence': avg_kl,
            'js_divergence': avg_js,
            'cosine_similarity': avg_cosine,
            'mse': avg_mse,
            'mae': avg_mae,
            'snr_db': avg_snr,
            'pearson_correlation': avg_corr,
            
            # Distribution statistics (aggregated for reference)
            'standard_stats': analyze_distribution_statistics(std_out),
            'quantized_stats': analyze_distribution_statistics(quant_out)
        }
        
        comparison['layers'][layer_name] = layer_metrics
    
    return comparison


def evaluate_quantization_quality(model_loader_func,
                                  quantization_modes: List[str],
                                  test_sequences: List[str],
                                  model_name: str,
                                  device: str,
                                  dtype: torch.dtype,
                                  output_dir: str,
                                  max_len: int = None) -> Dict:
    """
    Main function to evaluate quantization quality across different modes.
    
    Args:
        model_loader_func: Function to load model with quantization mode
        quantization_modes: List of quantization modes to evaluate
        test_sequences: List of test sequences
        model_name: Name of the model
        device: Device to run on
        dtype: Data type for model
        output_dir: Directory to save results
        max_len: Maximum sequence length
        
    Returns:
        Dictionary containing all quality evaluation results
    """
    
    print("\n" + "=" * 60)
    print("QUANTIZATION QUALITY EVALUATION")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, get standard (non-quantized) outputs as reference
    print("\nLoading standard (non-quantized) model...")
    import os
    os.environ['QUANT_MODE'] = 'standard'
    
    from model_to_benchmark import load_model
    std_model, std_tokenizer, _ = load_model(model_name, device, dtype)
    
    print("Extracting standard model outputs...")
    standard_outputs = extract_layer_outputs(
        std_model, std_tokenizer, test_sequences, device, max_len
    )
    
    # Clean up
    del std_model
    torch.cuda.empty_cache()
    
    # Evaluate each quantization mode
    all_comparisons = []
    
    for quant_mode in quantization_modes:
        if quant_mode == 'standard':
            continue
        
        print(f"\nEvaluating quantization mode: {quant_mode}")
        os.environ['QUANT_MODE'] = quant_mode
        
        try:
            # Load quantized model
            quant_model, quant_tokenizer, _ = load_model(model_name, device, dtype)
            
            # Extract outputs
            print(f"  Extracting {quant_mode} outputs...")
            quantized_outputs = extract_layer_outputs(
                quant_model, quant_tokenizer, test_sequences, device, max_len
            )
            
            # Debug: Check output shapes and values
            for layer_name in quantized_outputs.keys():
                std_shape = standard_outputs[layer_name].shape
                quant_shape = quantized_outputs[layer_name].shape
                std_vals = standard_outputs[layer_name].numpy()
                quant_vals = quantized_outputs[layer_name].numpy()
                
                print(f"    {layer_name}: std shape={std_shape}, quant shape={quant_shape}")
                print(f"      std range=[{std_vals.min():.3f}, {std_vals.max():.3f}], "
                      f"mean={std_vals.mean():.3f}, std={std_vals.std():.3f}")
                print(f"      quant range=[{quant_vals.min():.3f}, {quant_vals.max():.3f}], "
                      f"mean={quant_vals.mean():.3f}, std={quant_vals.std():.3f}")
                
                # Check if outputs are identical (shouldn't happen)
                if torch.allclose(standard_outputs[layer_name], quantized_outputs[layer_name], atol=1e-6):
                    print(f"    WARNING: {quant_mode} outputs are nearly identical to standard!")
            
            # Compare with standard
            print(f"  Computing quality metrics...")
            comparison = compare_quantization_outputs(
                standard_outputs, quantized_outputs, quant_mode
            )
            
            # Debug: Print computed metrics
            for layer_name, metrics in comparison['layers'].items():
                print(f"    {layer_name}: KL={metrics['kl_divergence']:.4f}, "
                      f"JS={metrics['js_divergence']:.4f}, "
                      f"cos_sim={metrics['cosine_similarity']:.4f}")
            
            all_comparisons.append(comparison)
            
            # Clean up
            del quant_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            log.error(f"Failed to evaluate {quant_mode}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    save_quality_results(all_comparisons, output_dir)
    
    return {
        'comparisons': all_comparisons,
        'standard_outputs': standard_outputs
    }


def save_quality_results(comparisons: List[Dict], output_dir: Path):
    """Save quality evaluation results to files."""
    
    # Create summary dataframe
    summary_data = []
    
    for comp in comparisons:
        quant_mode = comp['quantization_mode']
        
        for layer_name, metrics in comp['layers'].items():
            row = {
                'quantization_mode': quant_mode,
                'layer': layer_name,
                'kl_divergence': metrics['kl_divergence'],
                'js_divergence': metrics['js_divergence'],
                'cosine_similarity': metrics['cosine_similarity'],
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'snr_db': metrics['snr_db'],
                'pearson_correlation': metrics['pearson_correlation']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'quality_metrics.csv', index=False)
    
    # Save detailed statistics
    detailed_stats = []
    for comp in comparisons:
        quant_mode = comp['quantization_mode']
        for layer_name, metrics in comp['layers'].items():
            for stat_type in ['standard_stats', 'quantized_stats']:
                stats_dict = metrics[stat_type]
                row = {'quantization_mode': quant_mode, 'layer': layer_name, 'type': stat_type}
                row.update(stats_dict)
                detailed_stats.append(row)
    
    stats_df = pd.DataFrame(detailed_stats)
    stats_df.to_csv(output_dir / 'distribution_statistics.csv', index=False)
    
    print(f"\nQuality evaluation results saved:")
    print(f"  - quality_metrics.csv (KL divergence, cosine similarity, etc.)")
    print(f"  - distribution_statistics.csv (detailed distribution stats)")
