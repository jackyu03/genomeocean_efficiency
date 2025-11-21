#!/usr/bin/env python3
"""
Model loading configurations for benchmarking.
Modify the load_model() function to test different loading strategies.
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
import logging
import os

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


def load_model_4bit(model_name: str, device: str, dtype: torch.dtype):
    """4-bit quantized model loading with BitsAndBytesConfig."""
    log.info(f"Loading model (4-bit quantized): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


def load_model_8bit(model_name: str, device: str, dtype: torch.dtype):
    """8-bit quantized model loading with BitsAndBytesConfig."""
    log.info(f"Loading model (8-bit quantized): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


def load_model_4bit_fp4(model_name: str, device: str, dtype: torch.dtype):
    """4-bit FP4 quantized model loading."""
    log.info(f"Loading model (4-bit FP4 quantized): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4"
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


def load_model_4bit_nf4_double(model_name: str, device: str, dtype: torch.dtype):
    """4-bit NF4 with double quantization."""
    log.info(f"Loading model (4-bit NF4 double quantized): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config



# =============================================================================
# MAIN LOADING FUNCTION - CONTROLLED BY ENVIRONMENT VARIABLE
# =============================================================================

def load_model(model_name: str, device: str, dtype: torch.dtype):
    """
    Main model loading function controlled by QUANT_MODE environment variable.
    
    Set QUANT_MODE to one of:
    - "standard" (default): No quantization
    - "8bit": 8-bit quantization
    - "4bit_nf4": 4-bit NF4 quantization
    - "4bit_fp4": 4-bit FP4 quantization  
    - "4bit_nf4_double": 4-bit NF4 with double quantization
    
    Args:
        model_name: HuggingFace model identifier
        device: "cuda" or "cpu"
        dtype: torch.dtype (e.g., torch.float16, torch.bfloat16)
    
    Returns:
        tuple: (model, tokenizer, config)
    """
    
    quant_mode = os.environ.get("QUANT_MODE", "standard").lower()
    
    if quant_mode == "8bit":
        return load_model_8bit(model_name, device, dtype)
    elif quant_mode == "4bit_nf4":
        return load_model_4bit(model_name, device, dtype)
    elif quant_mode == "4bit_fp4":
        return load_model_4bit_fp4(model_name, device, dtype)
    elif quant_mode == "4bit_nf4_double":
        return load_model_4bit_nf4_double(model_name, device, dtype)
    elif quant_mode == "standard":
        return load_model_standard(model_name, device, dtype)
    else:
        log.warning(f"Unknown QUANT_MODE '{quant_mode}', falling back to standard loading")
        return load_model_standard(model_name, device, dtype)


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