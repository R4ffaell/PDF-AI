import streamlit as st
from transformers import pipeline
import numpy as np
import torch

from pdf_loader import load_pdf_files
from chunking import chunk_documents
from embedder import embed_chunks
from retriever import retrieve_relevant_chunks

st.set_page_config(page_title="Chat with Multiple PDFs", layout="wide")
st.title("📚 Chat with Multiple PDFs")

# Session state init
if "show_embedding_progress" not in st.session_state:
    st.session_state.show_embedding_progress = True

# Sidebar
st.sidebar.header("⚙️ Pengaturan")

llm_model_choice = st.sidebar.selectbox(
    "Pilih Model LLM:",
    [
        "deepset/roberta-base-squad2",
        "google/flan-t5-base",
    ],
    index=0
)

embedding_model_choice = st.sidebar.selectbox(
    "Pilih Model Embedding:",
    [
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    index=0
)

k_retrieval = st.sidebar.slider(
    "Jumlah Konteks (Chunks) untuk Diambil:",
    min_value=1, max_value=20, value=5, step=1
)

# Main interface
uploaded_files = st.file_uploader(
    "Unggah file PDF Anda:", type=["pdf"], accept_multiple_files=True
)

query = st.text_input("Tanyakan sesuatu tentang PDF Anda:")

if uploaded_files and query:
    # 1. Load PDF
    docs_data = load_pdf_files(uploaded_files)
    if not docs_data:
        st.warning("Tidak ada teks yang berhasil diekstrak dari PDF.")
        st.stop()

    # 2. Chunking
    chunks = chunk_documents(docs_data)
    if not chunks:
        st.warning("Tidak ada chunk yang berhasil dibuat.")
        st.stop()
    st.caption(f"Total {len(chunks)} chunks dibuat dari {len(docs_data)} dokumen.")

    # 3. Embedding
    with st.spinner("🧠 Membuat embedding..."):
        st.session_state.show_embedding_progress = True
        embedded_chunks = embed_chunks(chunks, embedding_model_choice)
        st.session_state.show_embedding_progress = False

    # 4. Retrieval
    with st.spinner(f"🔍 Mengambil {k_retrieval} chunk relevan..."):
        relevant_chunks = retrieve_relevant_chunks(query, embedding_model_choice, embedded_chunks, k=k_retrieval)

    if not relevant_chunks:
        context_str = "Tidak ada konteks relevan ditemukan."
        st.warning("Tidak ada potongan teks yang relevan.")
    else:
        context_str = "\n\n---\n\n".join(chunk["content"] for chunk in relevant_chunks)

    # 5. Jawaban
    with st.spinner(f"💡 Menghasilkan jawaban dengan '{llm_model_choice}'..."):
        answer = "Tidak ada jawaban yang bisa dihasilkan."
        confidence = None
        device = 0 if torch.cuda.is_available() else -1

        try:
            if "squad" in llm_model_choice:
                qa = pipeline("question-answering", model=llm_model_choice, device=device)
                result = qa(question=query, context=context_str)
                answer = result["answer"]
                confidence = result["score"]
            else:
                t2t = pipeline("text2text-generation", model=llm_model_choice, device=device)
                prompt = (
                    "Berdasarkan konteks berikut, jawablah pertanyaan yang diberikan. "
                    "Jika informasi tidak ada dalam konteks, katakan bahwa tidak tersedia.\n\n"
                    f"Konteks:\n{context_str}\n\n"
                    f"Pertanyaan: {query}\n\nJawaban:"
                )
                result = t2t(prompt, max_new_tokens=250)
                answer = result[0]["generated_text"]

        except Exception as e:
            st.error(f"Error saat menjalankan model: {e}")
            st.stop()

        st.markdown("### 💡 Jawaban")
        st.write(answer)
        if confidence:
            st.caption(f"Confidence: {confidence:.2%}")

    # 6. Tampilkan konteks
    with st.expander("📄 Konteks yang Digunakan"):
        for i, chunk in enumerate(relevant_chunks):
            st.markdown(f"**Chunk #{i+1} (Dokumen: {chunk['doc_name']})**")
            preview = chunk["content"][:700] + ("..." if len(chunk["content"]) > 700 else "")
            st.markdown(f"```\n{preview}\n```")

elif uploaded_files and not query:
    st.info("Silakan masukkan pertanyaan Anda.")
elif not uploaded_files and query:
    st.info("Silakan unggah file PDF terlebih dahulu.")
else:
    st.info("Unggah file PDF dan ajukan pertanyaan untuk mulai.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dibuat dengan ❤️ pakai Streamlit & HuggingFace 🤗")
