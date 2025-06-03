from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
import torch

# Gunakan @st.cache_resource untuk model karena ini adalah resource yang mahal untuk dimuat ulang
@st.cache_resource
def load_embedder_model(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """Memuat model SentenceTransformer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, device=device)
        st.caption(f"Model embedding '{model_name}' berhasil dimuat di {device}.")
    except Exception as e:
        st.error(f"Gagal memuat model embedding '{model_name}' di {device}. Error: {e}. Mencoba CPU.")
        # Fallback ke CPU jika ada error
        try:
            model = SentenceTransformer(model_name, device="cpu")
            st.caption(f"Model embedding '{model_name}' berhasil dimuat di CPU (fallback).")
        except Exception as e_cpu:
            st.error(f"Gagal memuat model embedding '{model_name}' bahkan di CPU. Error: {e_cpu}")
            return None # Kembalikan None jika gagal total
    return model

@st.cache_data # Cache hasil embedding karena ini adalah data hasil komputasi
def embed_chunks(_chunk_objects, _model_name_for_cache_key):
    """
    Meng-embed konten dari chunk objects menggunakan model yang ditentukan.
    '_chunk_objects' adalah list of dictionaries [{'content': str, 'source': str}, ...].
    '_model_name_for_cache_key' digunakan untuk memastikan cache di-invalidate jika model berubah,
                                dan untuk memuat model yang benar.
    """
    # Memuat model (akan diambil dari cache_resource jika sudah ada)
    model = load_embedder_model(_model_name_for_cache_key)
    if model is None:
        st.error("Model embedding tidak tersedia, tidak dapat melanjutkan proses embedding.")
        return None # Kembalikan None jika model gagal dimuat

    if not _chunk_objects:
        return np.array([]) # Kembalikan array numpy kosong jika tidak ada chunk

    chunk_contents = [chunk['content'] for chunk in _chunk_objects]
    
    # Tambahkan show_progress_bar untuk memberikan feedback visual
    embeddings = model.encode(chunk_contents, show_progress_bar=st.session_state.get("show_embedding_progress", True))
    return embeddings