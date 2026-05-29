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
    
    print("Generating a representative 5,000-bp DNA sequence...")
    seq = "".join(random.choices(['A', 'C', 'G', 'T'], k=5000))
    
    inputs = tokenizer(seq, return_tensors="pt")
    if args.device == "cuda":
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
    
    print("Running forward pass and extracting hidden states...")
    with torch.no_grad():
        # output_hidden_states=True intercepts the intermediate layers natively
        outputs = model(**inputs, output_hidden_states=True)
        
    # Grab the final hidden state before the embedding pooling or lm_head
    last_hidden_state = outputs.hidden_states[-1]
    print(f"Captured activation tensor of shape: {last_hidden_state.shape} and dtype {last_hidden_state.dtype}")
    
    # Flatten it into a 1D array of values for the histogram
    flattened = last_hidden_state.view(-1).cpu()
    
    out_file = "real_activations.pt"
    torch.save(flattened, out_file)
    print(f"Saved {len(flattened)} activation values to {out_file}!")

if __name__ == "__main__":
    main()
