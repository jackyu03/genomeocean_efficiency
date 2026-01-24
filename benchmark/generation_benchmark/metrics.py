
import torch
import torch.nn as nn
from tqdm import tqdm
import math

def compute_metrics_sliding_window(model, input_ids, stride=512, context_len=None, device="cuda"):
    """
    Computes both Perplexity (NLL) and Next-Token Accuracy using a sliding window.
    Optimized to run in a single forward pass per window.
    
    Args:
        model: AutoModelForCausalLM
        input_ids: Tensor of shape (1, seq_len)
        stride: Window stride
        context_len: Max context length.
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
    
    nlls = []
    total_correct = 0
    total_evaluated = 0
    prev_end_loc = 0
    
    # Progress Bar
    pbar = tqdm(range(0, seq_len, stride), desc="Evaluating")
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc
        
        # Determine effective target area (new tokens)
        if trg_len <= 0: break
        
        pbar.set_description(f"Eval | Window: [{begin_loc}:{end_loc}] | Target: {trg_len} toks")
        
        input_ids_chunk = input_ids[:, begin_loc : end_loc]
        
        # Prepare targets for Loss calculation
        # Mask out context (tokens we already evaluated or serve as history)
        target_ids = input_ids_chunk.clone()
        if trg_len < target_ids.size(1):
            target_ids[:, :-trg_len] = -100 
            
        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # 1. PERPLEXITY / LOSS
            # outputs.loss is mean NLL over the valid target tokens
            loss = outputs.loss
            nlls.append(loss * trg_len)
            
            # 2. ACCURACY
            logits = outputs.logits
            
            # Shift for prediction: logits[i] predicts input[i+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_chunk[..., 1:].contiguous()
            
            preds = torch.argmax(shift_logits, dim=-1)
            correct_mask = (preds == shift_labels)
            
            # Slice relevant part of the mask: last `trg_len` tokens
            # Note: shift_labels matches `target_ids` logic but offset by 1 position conceptually in stream
            # The shift operation reduces length by 1.
            # If trg_len == input_len (first window), we take all available predictions
            
            if trg_len >= correct_mask.size(1):
                eval_mask = correct_mask
            else:
                eval_mask = correct_mask[:, -trg_len:]
                
            total_correct += eval_mask.sum().item()
            
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    # Aggregate Metrics
    total_evaluated_tokens = sum([nll.numel() for nll in nlls]) if False else total_correct # No, wait, total_evaluated is sum of trg_len?
    # Correct accounting:
    # nlls contains (loss_mean * trg_len). Summing them gives Total NLL Sum.
    # total_correct is sum of correct predictions.
    
    # Re-calculate total tokens just to be safe from loop
    # Actually, we tracked trg_len in loop.
    # Let's trust total_evaluated if we tracked it
    # Fix: We didn't increment total_evaluated in loop above properly.
    
    # Re-calculating total tokens from NLL stack is safest?
    # No, let's use the explicit sum of trg_len tokens we processed.
    # Actually, nlls.append(loss * trg_len) -> loss is scalar.
    total_token_count = end_loc # This is just total seq len? No, we skip overlap.
    # The effective tokens we evaluated is "end_loc" minus "initial skipped"?
    # For standard PPL, total tokens = end_loc usually.
    # Let's use `pbar.n` derived logic or just sum chunks.
    
    # Robust way:
    total_nll_sum = torch.stack(nlls).sum().item()
    
    # We need exact count of tokens that contributed to 'loss'.
    # In the loop, we multiplied by `trg_len`. So we should sum those `trg_len` values.
    # But wait, does `trg_len` account for the "shift -1" in accuracy?
    # Accuracy loses 1 token at the very start of the sequence.
    # Loss calculation (HF default) also ignores the last token prediction if labels are shifted internally? 
    # Actually AutoModelForCausalLM handles shifting internally for Loss.
    # So Loss covers `trg_len` tokens exactly?
    # Usually `input_id` length L -> Loss calculated on L-1 tokens?
    # If using `labels`, HF model shifts labels. 
    # Labels: [A, B, C, D]. Model predicts [B, C, D, ?].
    # So we evaluate L-1 predictions.
    
    # Accuracy logic shifted manually: L -> L-1.
    # So `eval_mask` size is `trg_len` (if we sliced) OR `trg_len - 1` (at start)?
    # Let's assume for large sequences, the 1 token difference is negligible.
    # But to be precise:
    # Use `total_correct / total_accuracy_tokens`.
    
    # Since we didn't track total acc tokens in loop, let's just re-run logic or trust broad counts.
    # Let's assume total_token_count ~ end_loc.
    
    # Refined Loop to strictly track count
    return {
        'perplexity': math.exp(total_nll_sum / end_loc),
        'neg_log_likelihood': total_nll_sum / end_loc,
        'accuracy': total_correct / end_loc, # Approx
        'total_tokens': end_loc
    }

# Re-implementing strictly to return correct counts
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
    # If not verbose, we still iterate but without the bar to keep logs clean
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
            
            # Loss
            # outputs.loss is computed over tokens where labels != -100
            # We assume HF internals work correctly.
            # But we must know HOW MANY tokens were valid to un-average the mean.
            # Valid tokens = (target_ids != -100).sum()
            # Note: HF shifts labels by 1 internally. 
            # So actual valid tokens = (target_ids[..., 1:] != -100).sum()
            
            # Let's count explicitly
            # Be careful with the shift.
            # target_ids has same shape as input. 
            # Shifted labels -> size L-1.
            
            valid_loss_tokens = (target_ids[:, 1:] != -100).sum().item()
            if valid_loss_tokens > 0:
                nll_sum += outputs.loss.item() * valid_loss_tokens
                total_tokens_loss += valid_loss_tokens
            
            # Accuracy
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_chunk[..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            
            # We need to apply the SAME mask logic to accuracy
            # The target_ids mask was for INPUT (before shift).
            # We masked the first (L - trg_len) tokens.
            # So in the shifted labels, we should mask the first (L - trg_len - 1)?
            # Actually simpler: just use the -100 mask from target_ids on the shifted labels!
            
            # Shift the target_ids too to see which ones are valid
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

# Alias for compatibility
compute_metrics_sliding_window = compute_metrics_sliding_window_strict
