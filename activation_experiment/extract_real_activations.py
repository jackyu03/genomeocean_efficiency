import torch
from transformers import AutoModel, AutoTokenizer
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Extract real activations from a GenomeOcean model")
    parser.add_argument("--model", type=str, default="DOEJGI/GenomeOcean-100M", help="HuggingFace Model ID or local path (e.g. 'DOEJGI/GenomeOcean-100M')")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    args = parser.parse_args()
    
    print(f"Loading {args.model} on {args.device} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Auto-map to device if cuda, else just load
    device_map = "auto" if args.device == "cuda" else None
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device_map)
    model.eval()
    
    from tqdm import tqdm
    
    num_seqs = 5
    seq_len = 5000
    print(f"Generating and processing {num_seqs} representative {seq_len}-bp DNA sequences...")
    
    all_activations = []
    
    for _ in tqdm(range(num_seqs), desc="Running forward passes"):
        seq = "".join(random.choices(['A', 'C', 'G', 'T'], k=seq_len))
        inputs = tokenizer(seq, return_tensors="pt")
        
        # Some tokenizers return token_type_ids, which Mistral (GenomeOcean) doesn't accept
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
            
        if args.device == "cuda":
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # output_hidden_states=True intercepts the intermediate layers natively
            # use_cache=False prevents HuggingFace from using DynamicCache, bypassing transformers version mismatches
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            
        # Grab ALL hidden states across the entire network (embeddings + all intermediate layers)
        for layer_tensor in outputs.hidden_states:
            all_activations.append(layer_tensor.view(-1).cpu())
        
    print("Concatenating tensors...")
    # Flatten into a 1D array of values for the histogram
    flattened = torch.cat(all_activations)
    print(f"Captured activation tensor of length {len(flattened)} and dtype {flattened.dtype}")
    
    out_file = "real_activations.pt"
    torch.save(flattened, out_file)
    print(f"Saved {len(flattened)} activation values to {out_file}!")

if __name__ == "__main__":
    main()
