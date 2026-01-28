#!/usr/bin/env python3
"""
Model loading configurations for benchmarking.
Supports explicit selection between Native (dtype) and BitsAndBytes (quantization) loading.
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
import logging
import os

log = logging.getLogger("model_loader")

# =============================================================================
# Helper: Native Loader (PyTorch Dtype)
# =============================================================================
def load_model_native(model_name: str, device: str, precision: str, model_type: str = "base"):
    """
    Load model natively with a specific torch.dtype.
    Supports: bf16, fp16, fp32, and fp8 (if supported by torch).
    """
    log.info(f"Loading model (NATIVE, precision={precision}): {model_name}")
    
    # Map precision string to torch dtype
    target_dtype = None
    load_dtype = torch.bfloat16 # Default loading dtype
    
    if precision == "bf16":
        load_dtype = torch.bfloat16
        target_dtype = torch.bfloat16
    elif precision == "fp16":
        load_dtype = torch.float16
        target_dtype = torch.float16
    elif precision == "fp32":
        load_dtype = torch.float32
        target_dtype = torch.float32
    elif precision == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            # CURRENT WORKAROUND: Transformers cannot set default dtype to fp8.
            # We load in BF16, then cast to FP8.
            # This is not "true" native loading (which would save memory during load),
            # but it allows running the FP8 kernels.
            load_dtype = torch.bfloat16
            target_dtype = torch.float8_e4m3fn
            log.info("FP8 Native: Loading in BF16 first, then casting to torch.float8_e4m3fn")
        else:
            raise ValueError("Requested 'fp8' mode but torch.float8_e4m3fn is NOT available in this PyTorch version.")
    else:
        # Fallback
        load_dtype = torch.bfloat16
        target_dtype = torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    model_class = AutoModelForCausalLM if model_type == "causal" else AutoModel
    
    # Load model (using safe load_dtype)
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=load_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Post-load cast if needed (e.g. for FP8)
    if precision == "fp8":
        try:
            import torchao
            log.info(f"Found torchao version: {torchao.__version__}")
            
            # Try Modern API (torchao >= 0.5)
            from torchao.quantization import quantize_, float8_weight_only
            log.info("Applying torchao.quantization.float8_weight_only()...")
            quantize_(model, float8_weight_only())
            log.info("Successfully quantized model to Native FP8 (via torchao).")
            
        except (ImportError, AttributeError) as e:
            # Fallback failed or API missing
            error_msg = (
                f"\n\n[ERROR] TorchAO FP8 Quantization failed (Error: {e}).\n"
                f"Installed torchao version: {getattr(torchao, '__version__', 'Unknown')}\n"
                "Your torchao version does not support the required 'float8_weight_only' API.\n"
                "Possible Fix:\n"
                "  pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121\n"
                "OR, use standard BF16 by setting --precision bf16 (instead of fp8).\n"
            )
            log.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            log.error(f"Failed to apply torchao quantization: {e}")
            raise e
    
    return model, tokenizer, config

# =============================================================================
# Helper: BitsAndBytes Loader
# =============================================================================
def load_model_bitsandbytes(model_name: str, device: str, quant_mode: str, compute_dtype: torch.dtype, model_type: str = "base"):
    """
    Load model using BitsAndBytes quantization.
    Supports: 8bit, 4bit_nf4, 4bit_fp4.
    """
    log.info(f"Loading model (BitsAndBytes, mode={quant_mode}): {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Build Config
    if quant_mode == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode == "4bit_nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quant_mode == "4bit_fp4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="fp4"
        )
    else:
        log.warning(f"Unknown BnB mode {quant_mode}, defaulting to 8bit")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
    model_class = AutoModelForCausalLM if model_type == "causal" else AutoModel
    
    model = model_class.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    return model, tokenizer, config


# =============================================================================
# MAIN INTERFACE
# =============================================================================
def load_model(model_name: str, device: str, dtype: torch.dtype, model_type: str = "base", loader_type: str = "native"):
    """
    Main model loading function.
    
    Args:
        model_name: HF Model ID
        device: "cuda" or "cpu"
        dtype: Base compute dtype (used for bnb dequantization or native fallback)
        model_type: "base" or "causal"
        loader_type: "native" or "bitsandbytes" (Default: native)
        
    Env Vars:
        QUANT_MODE: 
          - If loader_type="native", expects "bf16", "fp8", etc.
          - If loader_type="bitsandbytes", expects "8bit", "4bit_nf4", etc.
    """
    
    # Get precision/mode from Env used by benchmark script
    mode_arg = os.environ.get("QUANT_MODE", "bf16").lower()
    
    if loader_type == "native":
        # In native mode, QUANT_MODE is treated as the target precision
        return load_model_native(model_name, device, precision=mode_arg, model_type=model_type)
        
    elif loader_type == "bitsandbytes":
        return load_model_bitsandbytes(model_name, device, quant_mode=mode_arg, compute_dtype=dtype, model_type=model_type)
        
    else:
        log.warning(f"Unknown loader type {loader_type}, using Native")
        return load_model_native(model_name, device, precision=mode_arg, model_type=model_type)


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