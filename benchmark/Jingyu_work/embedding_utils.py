import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from types import MethodType


from load_model_to_compare import load_model, get_model_info, get_max_length

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def reorder_sequences(sequences, seed=0):
    """ reorder the sequences by length
    process sequences with similar lengths in the same batch can greatly speed up the computation
    need to adjust batch_size according to the GPU memory
    use all GPUs on a node"""

    lengths = [len(seq) for seq in sequences]
    idx = np.argsort(lengths)
    return [sequences[i] for i in idx], idx 

def max_divisor_of_12(number):
    """Return the maximum gpu number within [1, number] that divides 12 (attention head) evenly."""
    max_divisor = None
    for i in range(1, number + 1):
        if 12 % i == 0:
            max_divisor = i
    return max_divisor   

def print_model_dtype(model):
    """Prints the torch.dtype of the model's first parameter."""
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Model's primary dtype is: **{param.dtype}**")
            print(f"(Parameter: {name})")
            return
    for name, param in model.named_parameters():
        print(f"✅ Model's primary dtype is: **{param.dtype}**")
        print(f"(Parameter: {name})")
        return

    print("❌ Could not determine model dtype.")

# main class
class LLMUtils:
    def __init__(self, model_dir, max_len, quant_mode="standard", dtype=torch.float16):
        os.environ["QUANT_MODE"] = quant_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, config = load_model(model_dir, self.device, dtype)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.model = model
        self.model.config.use_cache = False
        # DynamicCache 兼容补丁
        def safe_forward(self_, *args, **kwargs):
            kwargs.pop("past_key_values", None)
            kwargs["use_cache"] = False
            return self_.__class__.forward(self_, *args, **kwargs)
        self.model.forward = MethodType(safe_forward, self.model)
        self.gpus = 1

    def embed(self, dna_sequences, batch_size=25):
 
        dna_sequences, idx = reorder_sequences(dna_sequences)
        tokenizer = self.tokenizer
        model = self.model
        print_model_dtype(model)
        if not getattr(model, "is_quantized", False):
            model.to(self.device)
        model.eval()
        train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*self.gpus, shuffle=False, num_workers=2*self.gpus, prefetch_factor=2)
  
        for j, batch in enumerate(tqdm.tqdm(train_loader)):
            with torch.no_grad():
                token_feat = tokenizer.batch_encode_plus(
                        batch,
                        max_length=self.max_len,
                        return_tensors='pt',
                        padding='longest',
                        truncation=True
                )
                if "token_type_ids" in token_feat:
                    token_feat.pop("token_type_ids")
                inputs = {k: v.to(self.device) for k, v in token_feat.items()}
                min_mask_sum = token_feat["attention_mask"].sum(dim=1).min().item()
                if min_mask_sum == 0:
                    print("!!! WARNING: A sequence produced an Attention Mask sum of 0 !!!") # <--- 新增
                out = model(**inputs, output_hidden_states=True, return_dict=True)
                # hidden = out.last_hidden_state.detach().cpu()
                hidden = out.hidden_states[-1].detach().cpu()
                mask = token_feat["attention_mask"].unsqueeze(-1).cpu()
                emb = (hidden * mask).sum(1) / mask.sum(1)
                if torch.isnan(emb).any():
                    print(f"!!! NAN DETECTED in batch {j} !!!") # <--- 新增
                if j == 0:
                    outputs = emb
                else:
                    outputs = torch.cat((outputs, emb), dim=0)

        outputs = np.array(outputs.detach().float().cpu())

        # reorder the embeddings according to the original order
        outputs = outputs[np.argsort(idx)]
        return outputs