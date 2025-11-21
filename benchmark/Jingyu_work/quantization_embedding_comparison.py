"""Example of embedding sequences using a pre-trained model.

This script demonstrates how to embed sequences using a pre-trained model.

python quantization_embedding_comparison.py --model_dir DOEJGI/GenomeOcean-100M --model_max_length 1024 --sequence_file dataset/arc53_2000_seq_50k.csv --batch_size 4 --precision float16 --output_file outputs/quant_embeddings_comparison.csv

"""

from embedding_utils import LLMUtils
import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from types import MethodType
from load_model_to_compare import load_model
from types import MethodType


quant_modes = ["standard", "8bit", "4bit_nf4", "4bit_fp4", "4bit_nf4_double"]
dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

def compute_quant_embeddings(sequences, model_dir, batch_size, quant_modes = quant_modes, dtype = torch.float16, max_len = 10240):
    quant_dict = {}
    for mode in quant_modes:
        print(f"\n=== Generating embeddings in {mode} mode ===")
        llm = LLMUtils(
            model_dir=model_dir,
            quant_mode = mode,
            dtype = dtype,
            max_len = max_len
        )
        embeddings = llm.embed(sequences, batch_size=batch_size)
        if embeddings is None:
            print(f"!!! CRITICAL ERROR: {mode} mode returned None embeddings !!!")
        elif np.isnan(embeddings).any():
            print(f"!!! CRITICAL ERROR: {mode} embeddings contain NAN !!!")
        quant_dict[mode] = embeddings
    return quant_dict

def compute_similarity_metrics(ref, emb_t, mode):
    sims = F.cosine_similarity(ref, emb_t, dim=1)
    mean_sim = sims.mean().item()
    std_sim = sims.std().item()
    print(f"{mode:15s} → cosine similarity mean={mean_sim:.6f} ± {std_sim:.6f}")

    loss_mse = F.mse_loss(ref, emb_t, reduction='mean')
    mean_mse = loss_mse.item()
    print(f"{mode:15s} → mean square error={mean_mse:.6f}")

    diff = ref - emb_t
    l2_distances = torch.linalg.norm(diff, dim=1)
    mean_l2 = l2_distances.mean().item()
    std_l2 = l2_distances.std().item()
    print(f"{mode:15s} → mean L2 distance={mean_l2:.6f} ± {std_l2:.6f}")

    return (mean_sim, std_sim, mean_mse, mean_l2, std_l2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='comparison of embedding sequences with quantization')
    parser.add_argument('--model_dir', type=str, help='Path to the model')
    parser.add_argument('--model_max_length', type=int, default=10240, help='model max length')
    parser.add_argument('--sequence_file', type=str, help='Path to the sequence file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--precision', type=str, default="float16", choices=["float16","bfloat16","float32"])
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    args = parser.parse_args()

    sequence_df = pd.read_csv(args.sequence_file)
    sequences = sequence_df["seq"].to_list()
    # with open(args.sequence_file, "r") as f: 
    #     sequences = f.read().splitlines()
    print(f"Get {len(sequences)} sequences from {args.sequence_file} with max length {np.max([len(seq) for seq in sequences])}")
    
    precision = dtype_map[args.precision]
    quant_dict = compute_quant_embeddings(sequences, args.model_dir, args.batch_size, dtype = precision, max_len = args.model_max_length)
    print(f"Quant dict keys: {list(quant_dict.keys())}")
    ref = torch.tensor(quant_dict["standard"])
    print("\n=== Cosine Similarity vs Standard ===")
    results = {}
    for mode, emb in quant_dict.items():
        if mode == "standard":
            continue
        emb_t = torch.tensor(emb)
        mean_sim, std_sim, mean_mse, mean_l2, std_l2 = compute_similarity_metrics(ref, emb_t, mode)
        results[mode] = (mean_sim, std_sim, mean_mse, mean_l2, std_l2)

    # Save
    df = pd.DataFrame([
    {"model": args.model_dir,
     "batch_size": args.batch_size,
     "precision": args.precision,
     "quant_mode": k, 
     "mean_cosine_similarity": v[0], 
     "std_cosine_similarity": v[1], 
     "mean_square_error": v[2], 
     "mean_L2_distance": v[3], 
     "std_l2_distance": v[4]} for k, v in results.items()
    ])
    folder = os.path.dirname(args.output_file)
    if folder:  
        os.makedirs(folder, exist_ok=True)
    file_name = os.path.basename(args.output_file)
    model_name = os.path.basename(args.model_dir)
    file = f"{model_name}_{args.batch_size}_{args.precision}_{file_name}"
    output_path = os.path.join(folder if folder else ".", file)
    df.to_csv(output_path, index=False)
    print(f"\n✅ comparison results saved to {output_path}")