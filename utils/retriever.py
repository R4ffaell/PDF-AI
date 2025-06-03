import faiss
import numpy as np
import streamlit as st
from .embedder import load_embedder_model # <-- Impor yang diperbaiki

@st.cache_data # Cache vector store
def build_vector_store(embeddings_array, chunk_objects_with_metadata, _embeddings_ref_for_cache_key):
    """
    Membangun FAISS index untuk embeddings yang diberikan.
    'chunk_objects_with_metadata' disimpan bersama untuk diambil dengan metadata.
    '_embeddings_ref_for_cache_key' adalah representasi hashable dari embeddings untuk invalidasi cache.
    """
    if embeddings_array is None or embeddings_array.size == 0: # Periksa jika array kosong
        return None, []

    # Pastikan embeddings_array adalah numpy array dengan tipe float32
    if not isinstance(embeddings_array, np.ndarray):
        embeddings_array = np.array(embeddings_array)
    
    if embeddings_array.ndim == 1: # Jika hanya satu embedding, reshape
        embeddings_array = embeddings_array.reshape(1, -1)

    embeddings_array = embeddings_array.astype('float32')
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index, chunk_objects_with_metadata

@st.cache_data # Cache hasil retrieval
def retrieve_relevant_chunks(query: str,
                             _model_name_for_cache_key, # Digunakan untuk load embedder yang benar
                             _vector_store_data,        # Diabaikan untuk hashing (FAISS index tidak hashable)
                             k: int = 5):
    """
    Mengambil top k chunk objects yang relevan berdasarkan query.
    '_model_name_for_cache_key' digunakan untuk load model embedder yang sesuai.
    '_vector_store_data' adalah tuple (faiss_index, list_of_chunk_objects).
    """
    # Memuat model (akan diambil dari cache_resource jika sudah ada)
    embedder = load_embedder_model(_model_name_for_cache_key)
    if embedder is None:
        st.error("Model embedding tidak tersedia, tidak dapat melanjutkan proses retrieval.")
        return []

    faiss_index, all_chunk_objects = _vector_store_data

    if faiss_index is None or not all_chunk_objects:
        return []

    query_vector = embedder.encode([query], show_progress_bar=False)
    query_vector = np.array(query_vector).astype('float32')
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    num_docs_in_index = faiss_index.ntotal
    actual_k = min(k, num_docs_in_index)
    
    if actual_k == 0:
        return []

    try:
        distances, indices = faiss_index.search(query_vector, actual_k)
    except Exception as e:
        st.error(f"Error saat melakukan search di FAISS: {e}")
        return []
    
    retrieved_chunks = [all_chunk_objects[i] for i in indices[0]]
    return retrieved_chunks