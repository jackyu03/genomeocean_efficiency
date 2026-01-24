
import torch
import torch.nn as nn
from tqdm import tqdm
import math

def compute_metrics_sliding_window(model, input_ids, stride=512, context_len=None, verbose=False, device="cuda"):
    """
    Computes both Perplexity (NLL) and Next-Token Accuracy using a sliding window.
    
    Args:
        model: AutoModelForCausalLM
        input_ids: Tensor of shape (1, seq_len)
        stride: Window stride
        context_len: Max context length.
        verbose: If True, show detailed sliding window progress bar.
        device: "cuda"
        
    Returns:
        dict: {
            'perplexity': float,
            'neg_log_likelihood': float,
            'accuracy': float,
            'total_tokens': int
        }
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    if context_len is None:
        context_len = getattr(model.config, "max_position_embeddings", 2048)
        if not isinstance(context_len, int) or context_len > 100000:
             context_len = 2048
             
    seq_len = input_ids.size(1)
    
    nll_sum = 0.0
    acc_correct = 0
    total_tokens_loss = 0
    total_tokens_acc = 0
    prev_end_loc = 0
    
    # Progress Bar (Only if verbose)
    iterator = range(0, seq_len, stride)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating", leave=False)
    
    for begin_loc in iterator:
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0: break
        
        if verbose:
            iterator.set_description(f"Eval | [{begin_loc}:{end_loc}] | Tgt: {trg_len}")
        
        input_ids_chunk = input_ids[:, begin_loc : end_loc]
        target_ids = input_ids_chunk.clone()
        if trg_len < target_ids.size(1):
            target_ids[:, :-trg_len] = -100 
            
        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # Loss Calculation
            # outputs.loss is mean NLL over valid tokens
            # Valid tokens = (target_ids[:, 1:] != -100).sum()
            valid_loss_tokens = (target_ids[:, 1:] != -100).sum().item()
            
            if valid_loss_tokens > 0:
                nll_sum += outputs.loss.item() * valid_loss_tokens
                total_tokens_loss += valid_loss_tokens
            
            # Accuracy Calculation
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_chunk[..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            
            # Shift target mask for accuracy
            shift_target_mask = target_ids[:, 1:] != -100
            
            correct_preds = (preds == shift_labels) & shift_target_mask
            acc_correct += correct_preds.sum().item()
            total_tokens_acc += shift_target_mask.sum().item()

        prev_end_loc = end_loc
        if end_loc == seq_len: break
            
    avg_nll = nll_sum / total_tokens_loss if total_tokens_loss > 0 else 0.0
    accuracy = acc_correct / total_tokens_acc if total_tokens_acc > 0 else 0.0
    
    return {
        'perplexity': math.exp(avg_nll),
        'neg_log_likelihood': avg_nll,
        'accuracy': accuracy,
        'total_tokens': total_tokens_loss
    }
