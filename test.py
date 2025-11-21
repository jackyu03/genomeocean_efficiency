from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("DOEJGI/GenomeOcean-100M")
tokenizer = AutoTokenizer.from_pretrained("DOEJGI/GenomeOcean-100M")

print("Config max_position_embeddings:", getattr(config, "max_position_embeddings", None))
print("Tokenizer model_max_length:", getattr(tokenizer, "model_max_length", None))