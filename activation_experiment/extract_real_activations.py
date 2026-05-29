import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
import random
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract real per-layer activations from a GenomeOcean model")
    parser.add_argument("--model", type=str, default="DOEJGI/GenomeOcean-100M", help="HuggingFace Model ID or local path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--num-seqs", type=int, default=5, help="Number of random DNA sequences to run")
    parser.add_argument("--seq-len", type=int, default=5000, help="Length of each DNA sequence (bp)")
    parser.add_argument("--outdir", type=str, default="layer_activations", help="Output directory for per-layer tensors")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device_map = "auto" if args.device == "cuda" else None
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map
    )
    model.eval()

    print(f"Generating and processing {args.num_seqs} representative {args.seq_len}-bp DNA sequences...")

    # per_layer_activations[i] = list of 1D tensors from layer i across all sequences
    per_layer_activations = None

    for _ in tqdm(range(args.num_seqs), desc="Running forward passes"):
        seq = "".join(random.choices(['A', 'C', 'G', 'T'], k=args.seq_len))
        inputs = tokenizer(seq, return_tensors="pt")

        # GenomeOcean is Mistral-based, doesn't accept token_type_ids
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        if args.device == "cuda":
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # Initialize storage on first sequence (we now know the number of layers)
        if per_layer_activations is None:
            n_layers = len(outputs.hidden_states)
            per_layer_activations = [[] for _ in range(n_layers)]
            print(f"Model has {n_layers} hidden state tensors (embedding + {n_layers - 1} transformer layers)")

        for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
            per_layer_activations[layer_idx].append(layer_tensor.view(-1).cpu())

    print(f"\nSaving per-layer tensors to {outdir}/...")
    for layer_idx, layer_chunks in enumerate(tqdm(per_layer_activations, desc="Saving layers")):
        merged = torch.cat(layer_chunks)
        out_path = outdir / f"layer_{layer_idx:03d}.pt"
        torch.save(merged, out_path)

    print(f"\nDone! Saved {len(per_layer_activations)} layer tensors to '{outdir}/'")
    print(f"Each file contains ~{len(per_layer_activations[0][0]) * args.num_seqs:,} activation values.")

if __name__ == "__main__":
    main()
