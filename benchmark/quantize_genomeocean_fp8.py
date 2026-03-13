import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main():
    parser = argparse.ArgumentParser(description="Quantize GenomeOcean to FP8 W8A8 (Protocol B.2) using llm-compressor")
    parser.add_argument("--model", type=str, required=True, help="Path or ID of the HF model to quantize")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the quantized model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for quantization")
    args = parser.parse_args()

    model_id = args.model
    save_dir = args.output_dir
    if save_dir is None:
        save_dir = model_id.rstrip("/").split("/")[-1] + "-FP8-W8A8"

    print(f"Loading model: {model_id} on {args.device}...")
    # Use bfloat16 to match the original model's precision before quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=args.device, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Configuring FP8_DYNAMIC quantization (W8A8)...")
    # Protocol B.2 Logic: 
    # Static per-channel weights (W8)
    # Dynamic per-token activations (A8)
    # Ignore lm_head to avoid rounding noise in biological output space
    recipe = QuantizationModifier(
        targets="Linear", 
        scheme="FP8_DYNAMIC", 
        ignore=["lm_head"]
    )

    print("Applying one-shot quantization (No calibration data needed for dynamic scheme)...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving quantized model to: {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done! You can now run this model in vLLM with 'kv_cache_dtype=fp8' for the full B.2 effect.")

if __name__ == "__main__":
    main()
