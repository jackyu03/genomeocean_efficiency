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

### Quantization
```python
return load_model_quantized(model_name, device, dtype, quant_config)
```
