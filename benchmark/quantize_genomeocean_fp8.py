import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main():
    parser = argparse.ArgumentParser(
        description="Quantize GenomeOcean to FP8 W8A8 using llm-compressor.\n"
                    "  Protocol B.2: --model ...              (lm_head excluded, BF16 preserved)\n"
                    "  Protocol B.3: --model ... --include-lm-head  (full FP8, zero BF16 components)"
    )
    parser.add_argument("--model", type=str, required=True, help="Path or HF ID of the model to quantize")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the quantized model (auto-named by default)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for quantization (default: cuda)")
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Protocol B.3: Include lm_head in FP8 quantization (full zero-16-bit model). "
             "By default (Protocol B.2), lm_head is kept in BFloat16 to preserve biological logit fidelity."
    )
    args = parser.parse_args()

    model_id = args.model
    save_dir = args.output_dir

    # Auto-name the output directory based on protocol
    if save_dir is None:
        base_name = model_id.rstrip("/").split("/")[-1]
        if args.include_lm_head:
            save_dir = base_name + "-FP8-W8A8-FullFP8"   # Protocol B.3
        else:
            save_dir = base_name + "-FP8-W8A8"            # Protocol B.2

    protocol = "B.3 (Full FP8 — Zero 16-bit Components)" if args.include_lm_head else "B.2 (W8A8, lm_head excluded)"
    print(f"Quantization Protocol: {protocol}")
    print(f"Loading model: {model_id} on {args.device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Build the quantization recipe
    # FP8_DYNAMIC: static per-channel weights (W8), dynamic per-token activations (A8)
    if args.include_lm_head:
        # Protocol B.3: include ALL Linear layers, including lm_head
        print("Configuring FP8_DYNAMIC — ALL Linear layers (including lm_head)...")
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=[]
        )
    else:
        # Protocol B.2: exclude lm_head to preserve biological logit fidelity
        print("Configuring FP8_DYNAMIC — all Linear layers except lm_head (BF16 preserved)...")
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"]
        )

    print("Applying one-shot quantization (no calibration data required for FP8_DYNAMIC)...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving quantized model to: {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if args.include_lm_head:
        print(f"\nDone! Protocol B.3 model saved to: {save_dir}")
        print("Run with --quant-modes fp8_v3 and --model-w8a8-full <path> (if supported) or point --model at this path.")
    else:
        print(f"\nDone! Protocol B.2 model saved to: {save_dir}")
        print("Run with: --model-w8a8 <path> --quant-modes fp8_v2")

if __name__ == "__main__":
    main()
