# GenomeOcean Benchmark Suite

This benchmark suite allows you to test different model loading strategies and configurations.

## Quick Start

1. **Run with default settings:**
```bash
python go_benchmark.py --csv your_data.csv --model DOEJGI/GenomeOcean-4B
```

2. **Modify model loading strategy:**
Edit `model_to_benchmark.py` and uncomment your desired loading method in the `load_model()` function.

## Available Loading Strategies

### Standard Loading (Default)
```python
return load_model_standard(model_name, device, dtype)
```
- Regular HuggingFace model loading
- No optimizations

### 8-bit Quantization
```python
return load_model_8bit_quantized(model_name, device, dtype)
```
- Requires: `pip install bitsandbytes`
- Reduces memory usage ~50%
- Slight performance impact

### 4-bit Quantization  
```python
return load_model_4bit_quantized(model_name, device, dtype)
```
- Requires: `pip install bitsandbytes`
- Reduces memory usage ~75%
- More performance impact but significant memory savings

### Torch Compile Optimization
```python
return load_model_torch_compile(model_name, device, dtype)
```
- Requires: PyTorch 2.0+
- Can improve inference speed
- First run will be slower due to compilation

### Flash Attention 2
```python
return load_model_flash_attention(model_name, device, dtype)
```
- Requires: `pip install flash-attn`
- Faster attention computation
- Memory efficient for long sequences

### Custom Sharding
```python
return load_model_custom_sharding(model_name, device, dtype)
```
- For multi-GPU setups
- Customize device mapping in the function

## Usage Examples

### Test quantization impact:
1. Run benchmark with standard loading
2. Edit `model_to_benchmark.py` to use 8-bit quantization
3. Run benchmark again
4. Compare memory usage and performance

### Test different batch sizes:
```bash
python go_benchmark.py --csv data.csv --model your-model --batch-sizes 1 2 4 8 16 32
```

### Test different sequence length bins:
```bash
python go_benchmark.py --csv data.csv --model your-model --bins 1000 5000 10000 25000 50000
```

## Output

The benchmark generates:
- `benchmark_TIMESTAMP.csv` - Detailed results
- `benchmark_TIMESTAMP.meta.json` - Run metadata

Key metrics:
- `E2EL_ms` - End-to-end latency per batch
- `seqs_per_s` - Sequences processed per second  
- `tokens_per_s` - Tokens processed per second
- `peak_vram_GB` - Peak GPU memory usage
- `energy_kWh` - Energy consumption (if available)
- `TFLOPs_per_s` - Theoretical FLOPS throughput

## Adding New Loading Strategies

1. Add a new function in `model_to_benchmark.py`:
```python
def load_model_my_strategy(model_name: str, device: str, dtype: torch.dtype):
    # Your custom loading logic here
    return model, tokenizer, config
```

2. Add it as an option in the `load_model()` function:
```python
# My custom strategy
# return load_model_my_strategy(model_name, device, dtype)
```

3. Uncomment when you want to test it.

## Dependencies

Core requirements:
```bash
pip install torch transformers pandas numpy
```

Optional (for specific loading strategies):
```bash
pip install bitsandbytes  # For quantization
pip install flash-attn    # For Flash Attention 2
pip install pynvml        # For energy monitoring
```