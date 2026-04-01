import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def quantize_and_save(model_id, device, ignore_layers, save_dir, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Output: {save_dir}")
    print(f"  Ignored layers: {ignore_layers if ignore_layers else 'None (full FP8)'}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=ignore_layers
    )

    print("Applying one-shot FP8_DYNAMIC quantization (no calibration data required)...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving to: {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Done -> {save_dir}")

    # Free VRAM before next quantization
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantize GenomeOcean to FP8 W8A8 using llm-compressor.\n"
            "Produces TWO checkpoints by default:\n"
            "  *-B2-FP8-W8A8          : Protocol B.2 (lm_head kept in BF16)\n"
            "  *-B3-FP8-W8A8-FullFP8  : Protocol B.3 (all layers in FP8, zero BF16)"
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Path or HF ID of the model to quantize")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory prefix (default: derived from model name)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    args = parser.parse_args()

    model_id = args.model
    base_name = args.output_dir or model_id.rstrip("/").split("/")[-1]

    # Protocol B.2: lm_head excluded → kept in BFloat16 for biological logit fidelity
    quantize_and_save(
        model_id=model_id,
        device=args.device,
        ignore_layers=["lm_head"],
        save_dir=f"{base_name}-B2-FP8-W8A8",
        label="Protocol B.2 — W8A8 (lm_head in BF16)"
    )

    # Protocol B.3: all layers including lm_head → zero 16-bit components
    quantize_and_save(
        model_id=model_id,
        device=args.device,
        ignore_layers=[],
        save_dir=f"{base_name}-B3-FP8-W8A8-FullFP8",
        label="Protocol B.3 — Full FP8 (zero BF16 components)"
    )

    print("\nAll done! Both Protocol B.2 and B.3 checkpoints are ready.")
    print(f"  B.2 (--model-w8a8): {base_name}-B2-FP8-W8A8")
    print(f"  B.3 (--model-w8a8): {base_name}-B3-FP8-W8A8-FullFP8")


if __name__ == "__main__":
    main()
