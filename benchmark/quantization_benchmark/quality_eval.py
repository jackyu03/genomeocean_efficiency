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
    
    Args:
        model: The model to extract from
        tokenizer: Tokenizer for the model
        sequences: List of input sequences
        device: Device to run on
        max_len: Maximum sequence length
        layers: Which layers to extract ('last', 'second_last')
        
    Returns:
        Dictionary mapping layer names to output tensors
    """
    
    model.eval()
    layer_outputs = {layer: [] for layer in layers}
    
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(
                seq,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            inputs.pop('token_type_ids', None)
            
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            
            # Extract requested layers
            if 'last' in layers:
                layer_outputs['last'].append(hidden_states[-1].cpu())
            if 'second_last' in layers:
                layer_outputs['second_last'].append(hidden_states[-2].cpu())
    
    # Concatenate all outputs
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
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Cosine similarity value
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


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
    Compare outputs between standard and quantized models.
    
    Args:
        standard_outputs: Outputs from standard (non-quantized) model
        quantized_outputs: Outputs from quantized model
        quant_mode: Name of quantization mode
        
    Returns:
        Dictionary of comparison metrics
    """
    
    comparison = {
        'quantization_mode': quant_mode,
        'layers': {}
    }
    
    for layer_name in standard_outputs.keys():
        if layer_name not in quantized_outputs:
            continue
        
        std_out = standard_outputs[layer_name].numpy()
        quant_out = quantized_outputs[layer_name].numpy()
        
        # Ensure same shape
        if std_out.shape != quant_out.shape:
            log.warning(f"Shape mismatch for {layer_name}: {std_out.shape} vs {quant_out.shape}")
            continue
        
        # Compute distribution histograms for KL/JS divergence
        std_hist, bins = np.histogram(std_out.flatten(), bins=100, density=True)
        quant_hist, _ = np.histogram(quant_out.flatten(), bins=bins, density=True)
        
        # Normalize histograms
        std_hist = std_hist / (std_hist.sum() + 1e-10)
        quant_hist = quant_hist / (quant_hist.sum() + 1e-10)
        
        layer_metrics = {
            # Divergence metrics
            'kl_divergence': compute_kl_divergence(std_hist, quant_hist),
            'js_divergence': compute_js_divergence(std_hist, quant_hist),
            
            # Similarity metrics
            'cosine_similarity': compute_cosine_similarity(std_out, quant_out),
            'mse': compute_mse(std_out, quant_out),
            'mae': compute_mae(std_out, quant_out),
            'snr_db': compute_snr(std_out, quant_out),
            
            # Correlation
            'pearson_correlation': float(np.corrcoef(std_out.flatten(), quant_out.flatten())[0, 1]),
            
            # Distribution statistics
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
            
            # Compare with standard
            print(f"  Computing quality metrics...")
            comparison = compare_quantization_outputs(
                standard_outputs, quantized_outputs, quant_mode
            )
            all_comparisons.append(comparison)
            
            # Clean up
            del quant_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            log.error(f"Failed to evaluate {quant_mode}: {e}")
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
