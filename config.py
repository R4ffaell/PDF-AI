EMBEDDING_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": "HF: all-mpnet-base-v2 (Best overall)",
    "sentence-transformers/all-MiniLM-L6-v2": "HF: all-MiniLM-L6-v2 (Fast and good)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "HF: paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)"
}

LLM_MODELS = {
    "google/flan-t5-base": "HF Pipeline: flan-t5-base (Text Generation)",
    "google/flan-t5-large": "HF Pipeline: flan-t5-large (Text Generation)",
    "microsoft/DialoGPT-medium": "HF Pipeline: DialoGPT-medium (Conversational)",
    "distilgpt2": "HF Pipeline: distilgpt2 (Text Generation)",
    "gpt2": "HF Pipeline: gpt2 (Text Generation)"
}

# Anda bisa menambahkan konstanta lain di sini jika ada
# Contoh: DEFAULT_CHUNK_SIZE = 1000