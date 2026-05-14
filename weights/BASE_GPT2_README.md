Run: huggingface-cli download openai-community/gpt2 --local-dir base_gpt2
Or use transformers: from transformers import AutoModel; AutoModel.from_pretrained('openai-community/gpt2')
