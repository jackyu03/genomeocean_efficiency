#!/usr/bin/env python3
"""
Model loading configurations for benchmarking.
Modify the load_model() function to test different loading strategies.
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
import logging

log = logging.getLogger("model_loader")


def load_model_standard(model_name: str, device: str, dtype: torch.dtype):
    """Standard model loading without quantization."""
    log.info(f"Loading model (standard): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


def load_model_quantized(model_name: str, device: str, dtype: torch.dtype, quant_config: str):
    """quantized model loading."""
    log.info(f"Loading model ({quant_config}): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = None # add more information later
    
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config



# =============================================================================
# MAIN LOADING FUNCTION - MODIFY THIS TO CHANGE LOADING STRATEGY
# =============================================================================

def load_model(model_name: str, device: str, dtype: torch.dtype):
    """
    Main model loading function. 
    
    MODIFY THIS FUNCTION to change the loading strategy.
    Simply uncomment the desired loading method and comment out others.
    
    Args:
        model_name: HuggingFace model identifier
        device: "cuda" or "cpu"
        dtype: torch.dtype (e.g., torch.float16, torch.bfloat16)
    
    Returns:
        tuple: (model, tokenizer, config)
    """
    
    # =========================================================================
    # CHOOSE YOUR LOADING STRATEGY - UNCOMMENT ONE OF THE FOLLOWING:
    # =========================================================================
    
    # Standard loading (default)
    return load_model_standard(model_name, device, dtype)
    
    # quantization
    # return load_model_quantized(model_name, device, dtype, '8_bit')


def get_model_info(config, model):
    """Extract model architecture information for benchmarking."""
    n_params = sum(p.numel() for p in model.parameters())
    d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    n_layer = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    d_ff = getattr(config, "intermediate_size", None)
    
    return {
        "n_params": n_params,
        "d_model": d_model,
        "n_layer": n_layer,
        "d_ff": d_ff
    }


def get_max_length(config, tokenizer):
    """Determine maximum sequence length for the model."""
    max_len = getattr(config, "max_position_embeddings", None)
    if (max_len is None) or (isinstance(max_len, int) and max_len <= 0):
        tml = getattr(tokenizer, "model_max_length", None)
        if isinstance(tml, int) and tml < 10_000_000:
            max_len = tml
        else:
            max_len = None
    return max_len