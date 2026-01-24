
import torch
import torch.nn as nn
from tqdm import tqdm
import math

def compute_perplexity(model, input_ids, stride=512, context_len=None, device="cuda"):
    """
    Computes perplexity using a sliding window approach.
    
    Args:
        model: AutoModelForCausalLM
        input_ids: Tensor of shape (1, seq_len)
        stride: Window stride
        context_len: Max context length (window size). If None, auto-detects.
        device: "cuda" or "cpu"
        
    Returns:
        ppl: float
        nll: float (average negative log likelihood)
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    # Auto-detect context length if not provided
    if context_len is None:
        context_len = getattr(model.config, "max_position_embeddings", 2048)
        # Handle edge cases (some models set it to enormous values)
        if not isinstance(context_len, int) or context_len > 100000:
             context_len = 2048

    seq_len = input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    # Sliding Window Loop
    # We stride across the sequence, predicting [begin_loc : end_loc]
    # using context [begin_loc : end_loc] (clipped to max_len).
    
    pbar = tqdm(range(0, seq_len, stride), desc="PPL")
    for begin_loc in pbar:
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc  
        
        # Update description with position info
        pbar.set_description(f"PPL | Window: [{begin_loc}:{end_loc}] | Target: {trg_len} toks")
        
        input_ids_chunk = input_ids[:, begin_loc : end_loc]

        # Calculate target length (number of new tokens at the END of this chunk)
        trg_len = end_loc - prev_end_loc 
        if trg_len <= 0: break
        
        # Mask out context (previous tokens) so we don't double-count loss
        target_ids = input_ids_chunk.clone()
        if trg_len < target_ids.size(1):
            target_ids[:, :-trg_len] = -100 

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # outputs.loss is scalar mean NLL over valid target tokens
            nlls.append(outputs.loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    total_tokens = end_loc
    
    avg_nll = total_nll / total_tokens
    ppl = torch.exp(avg_nll)
    
    return ppl.item(), avg_nll.item()


def compute_accuracy_sliding_window(model, input_ids, stride=512, context_len=None, device="cuda"):
    """
    Computes next-token prediction accuracy using the same sliding window logic as Perplexity.
    
    Args:
        model: AutoModelForCausalLM
        input_ids: Tensor of shape (1, seq_len)
        stride: Window stride (evaluation step size)
        context_len: Max context length.
        device: "cuda"
        
    Returns:
        accuracy: float (0.0 to 1.0)
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    if context_len is None:
        context_len = getattr(model.config, "max_position_embeddings", 2048)
        if not isinstance(context_len, int) or context_len > 100000:
             context_len = 2048
             
    seq_len = input_ids.size(1)
    total_correct = 0
    total_evaluated = 0
    prev_end_loc = 0
    
    # Reuse exact same loop structure as PPL
    pbar = tqdm(range(0, seq_len, stride), desc="Accuracy")
    for begin_loc in pbar:
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc
        
        pbar.set_description(f"ACC | Window: [{begin_loc}:{end_loc}] | Target: {trg_len} toks")
        
        if trg_len <= 0: break
        
        input_ids_chunk = input_ids[:, begin_loc : end_loc]
        
        with torch.no_grad():
            outputs = model(input_ids_chunk)
            logits = outputs.logits # (1, L, V)
            
        # Shift logits and labels for next-token prediction
        # Logic: logits[i] predicts input[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids_chunk[..., 1:].contiguous()
        
        # Get predictions
        preds = torch.argmax(shift_logits, dim=-1) # (1, L-1)
        
        # Calculate boolean correctness tensor
        correct_mask = (preds == shift_labels) # (1, L-1)
        
        # Slice relevant part of the mask: last `trg_len` tokens
        if trg_len >= correct_mask.size(1):
            # Evaluate all available predictions (first window case)
            eval_mask = correct_mask
        else:
            # Evaluate only the last trg_len predictions (sliding window case)
            eval_mask = correct_mask[:, -trg_len:]
            
        total_correct += eval_mask.sum().item()
        total_evaluated += eval_mask.numel()
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    return total_correct / total_evaluated if total_evaluated > 0 else 0.0
