#!/usr/bin/env python3
"""
Quick test script to validate all quantization modes work correctly.
Run this before submitting the full benchmark job.
"""

import os
import sys
import torch
from model_to_benchmark import load_model

def test_quantization_mode(mode, model_name="DOEJGI/GenomeOcean-100M"):
    """Test a specific quantization mode."""
    print(f"\n=== Testing Quantization Mode: {mode} ===")
    
    # Set environment variable
    os.environ["QUANT_MODE"] = mode
    
    try:
        # Test loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        
        print(f"Loading model with {mode} quantization...")
        model, tokenizer, config = load_model(model_name, device, dtype)
        
        # Test basic inference
        test_seq = "ATCGATCGATCGATCG" * 10  # Simple test sequence
        inputs = tokenizer(test_seq, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        print("Running test inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check memory usage
        if device == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak GPU memory: {memory_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        print(f"[OK] {mode} quantization mode works correctly!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {mode} quantization mode failed: {str(e)}")
        return False

def main():
    """Test all quantization modes."""
    print("=== Quantization Mode Validation ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test all modes
    modes = ["standard", "8bit", "4bit_nf4", "4bit_fp4", "4bit_nf4_double"]
    results = {}
    
    for mode in modes:
        results[mode] = test_quantization_mode(mode)
    
    # Summary
    print("\n=== Test Results Summary ===")
    for mode, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{mode:20s}: {status}")
    
    failed_modes = [mode for mode, success in results.items() if not success]
    if failed_modes:
        print(f"\n[WARNING] Failed modes: {', '.join(failed_modes)}")
        print("Please check your environment and dependencies.")
        return 1
    else:
        print("\n[SUCCESS] All quantization modes working correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())