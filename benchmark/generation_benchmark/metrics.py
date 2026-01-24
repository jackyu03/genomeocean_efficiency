
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
    # We stride across the sequence.
    # For each step, we want to predict tokens [begin_loc : end_loc] 
    # using context [0 : end_loc] (clipped to max_len).
    
    # PPL Standard Logic (Hugging Face):
    # Iterate `begin_loc` from 0 to seq_len by stride.
    # end_loc = min(begin_loc + max_len, seq_len)
    # trg_len = end_loc - prev_end_loc  (This is wrong in standard implementation, standard uses stride as target len usually)
    
    # Let's stick to the robust HF implementation pattern:
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating Perplexity"):
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc  # How many NEW tokens are we covering? 
        
        # When begin_loc=0, end_loc=2048, prev=0 -> trg=2048 (Predict all)
        # When begin_loc=512, end_loc=2560 (if long enough), prev=2048 -> trg=512 (Predict last 512)
        
        # Actually, standard HF implementation logic:
        # trg_len is usually `end_loc - begin_loc` if we are jumping by stride?
        # NO. We want to evaluate the whole sequence exactly once.
        # Ideally: 
        # Window 1: [0...2048]. Targets: [0...2048].
        # Window 2: [512...2560]. Targets: [2048...2560] ?? No, that leaves gaps if stride < context.
        
        # Correct Sliding Window Logic (PPL & Accuracy):
        # We want to evaluate tokens [i ... i+stride].
        # We provide context [i-context+stride ... i+stride].
        
        # HF Style:
        # Input is `input_ids[:, begin_loc : end_loc]`
        # Target is same.
        # We set labels[:-trg_len] = -100 so we DONT calculate loss on context.
        # Ideally, `trg_len` should start at `stride` except for the first window.
        
        # Let's simplify:
        # If we use stride=stride, then for window starting at `begin_loc`, 
        # we predict the last `stride` tokens (mostly).
        
        # Refined Logic:
        # Window covers: [begin_loc : begin_loc + context_len]
        # We want to predict: The tokens that were NOT covered by previous effective windows.
        
        # Actually, let's just use the stride logic directly:
        # We evaluate the *last* `stride` tokens of the window, using the full window as context.
        # For the very first window, we evaluate everything.
        
        # Input Chunk: [begin_loc : begin_loc + context_len]
        # Target Len: 
        #   If begin_loc == 0: target_len = context_len (or end_loc)
        #   Else: target_len = stride (or whatever is left)
        
        # BUT `range(0, seq_len, stride)` steps by stride.
        # So at step 0: we cover [0:stride] (and more context).
        
        input_ids_chunk = input_ids[:, begin_loc : end_loc]
        
        # Calculate target length (the number of tokens at the END of this chunk we want to evaluate)
        # If this is the first window, we evaluate everything we see? 
        # Standard practice:
        # Window 1 [0:2048]: Eval [0:2048]
        # Window 2 [512:2560] (Stride=512): Eval [2048:2560]? No that skips [512:2048] redundancy.
        # The `prev_end_loc` variable tracks what we have already covered.
        
        trg_len = end_loc - prev_end_loc 
        if trg_len <= 0: break
        
        # We may need to mask the left side if trg_len < input_chunk_len
        # i.e. context is (input_len - trg_len)
        
        target_ids = input_ids_chunk.clone()
        if trg_len < target_ids.size(1):
            target_ids[:, :-trg_len] = -100 # Mask context

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            # outputs.loss is mean NLL over the valid target tokens
            
            # We must weight it by number of tokens to compute total NLL sum
            # Note: outputs.loss is scalar mean
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
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating Accuracy"):
        end_loc = min(begin_loc + context_len, seq_len)
        trg_len = end_loc - prev_end_loc
        
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
        
        # Now we only want to count the LAST `trg_len` tokens.
        # But wait, shift_len is L-1.
        # If input len is 2048, we have 2047 predictions.
        # If trg_len is 512 (meaning we want to evaluate the last 512 tokens of the INPUT),
        # that corresponds to the last 512 positions in the labels.
        
        # Correctly slicing the target area:
        # We want to evaluate input_ids_chunk[ -trg_len : ]
        # These correspond to labels[ -trg_len : ] inside shift_labels.
        # And correct_mask[ -trg_len : ]
        
        # Handle edge case where trg_len equals input len (first window)
        # If trg_len == input_len, we evaluate everything.
        # But we lose 1 token due to shift? 
        # Typically PPL calculation ignores the very first token of the entire sequence because it has no context.
        # The shift operation handles that (L -> L-1).
        
        # Slice the relevant part of the mask
        if trg_len >= correct_mask.size(1):
            # Evaluate all available predictions
            eval_mask = correct_mask
        else:
            # Evaluate only the last trg_len predictions
            eval_mask = correct_mask[:, -trg_len:]
            
        total_correct += eval_mask.sum().item()
        total_evaluated += eval_mask.numel()
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    return total_correct / total_evaluated if total_evaluated > 0 else 0.0
