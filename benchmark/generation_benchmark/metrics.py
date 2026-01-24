
import torch
import torch.nn as nn
from tqdm import tqdm
import math

def compute_perplexity(model, input_ids, stride=512, device="cuda"):
    """
    Computes perplexity using a sliding window approach, similar to the standard HF script.
    
    Args:
        model: AutoModelForCausalLM
        input_ids: Tensor of shape (1, seq_len) - flatten all data into one long sequence usually
        stride: Window stride
        device: "cuda" or "cpu"
        
    Returns:
        ppl: float
    """
    model.eval()
    
    # Ensure input_ids is on device
    input_ids = input_ids.to(device)
    
    max_length = model.config.max_position_embeddings
    # If model doesn't specify (some custom ones), safe default or from args
    if not isinstance(max_length, int):    
        max_length = 2048 # Fallback
        
    seq_len = input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    # We iterate over the sequence with a sliding window
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating Perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # How many new tokens we are predicting
        
        # Prepare input block
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        
        # The target is the same as input_ids, but we mask the context (previous tokens)
        target_ids = input_ids_chunk.clone()
        # Mask out the context (tokens before the current prediction window)
        # However, for PPL, we usually feed the whole context and predict the tokens in [trg_start:trg_end]
        # But HF standard approach: 
        #   labels set to -100 are ignored
        #   we want to calculate loss for the last `trg_len` tokens
        
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # Loss is averaged over tokens in the chunk
            # neg_log_likelihood = outputs.loss * trg_len
            
            # Better precision: outputs.loss is mean NLL per token.
            # We accumulate the sum of NLLs
            nlls.append(outputs.loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    total_tokens = end_loc
    
    avg_nll = total_nll / total_tokens
    ppl = torch.exp(avg_nll)
    
    return ppl.item(), avg_nll.item()


def compute_accuracy(model, input_ids_list, device="cuda", batch_size=4):
    """
    Computes next-token prediction accuracy on a list of sequences.
    
    Args:
        model: AutoModelForCausalLM
        input_ids_list: List of Tensors (1, L)
        device: "cuda"
        
    Returns:
        accuracy: float (0.0 to 1.0)
    """
    model.eval()
    
    total_correct = 0
    total_tokens = 0
    
    # Process in batches?
    # Since sequences might be different lengths, we might process one by one or batch if padded.
    # For simplicity/correctness without padding complexity, let's just do one by one or create a collator.
    # Given the requirements, let's do batching with padding to be efficient.
    
    # Actually, the runner will probably pass a big tensor or list.
    # If input_ids_list is a Tensor (N, L), good.
    
    for i in tqdm(range(0, len(input_ids_list), batch_size), desc="Calculating Accuracy"):
        batch = input_ids_list[i:i+batch_size].to(device) # (B, L)
        
        # Target is next token.
        # labels = batch[:, 1:]
        # preds = batch[:, :-1]
        # But we feed the whole sequence and predict next step.
        # logits: (B, L, V)
        # Shift logits and labels
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits # (B, L, V)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        
        # Argmax
        preds = torch.argmax(shift_logits, dim=-1) # (B, L-1)
        
        # Compare
        correct_mask = (preds == shift_labels)
        
        # If padding exists, we should mask it out?
        # Assuming inputs are already padded with PAD token, we should ignore PAD in accuracy?
        # Typically yes. But if the model predicts PAD correctly, is that good?
        # Usually we only care about real tokens.
        # Let's assume PAD token is handled by caller or ignore specific ID if known.
        # For now, simplistic exact match on all positions.
        
        total_correct += correct_mask.sum().item()
        total_tokens += correct_mask.numel()
        
    return total_correct / total_tokens if total_tokens > 0 else 0.0
