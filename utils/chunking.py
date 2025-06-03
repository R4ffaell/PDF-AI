from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import streamlit as st

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Ganti whitespace berlebih dengan satu spasi
    text = re.sub(r'[\f\r\x0b\x0c]+', ' ', text) # Hapus form feed, carriage return, dll.
    return text.strip()

@st.cache_data # Cache hasil chunking
def chunk_documents(docs_data, chunk_size=800, chunk_overlap=100):
    """
    Memecah dokumen (teks dengan metadata sumber) menjadi potongan (chunks) yang lebih kecil.
    'docs_data' adalah list of dictionaries dari load_pdf_data.
    Mengembalikan list of dictionaries, di mana setiap dictionary berisi:
    - 'content': Konten teks dari chunk.
    - 'source': Sumber asli (nama file dan nomor halaman) dari chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # Prioritas pemisah
        length_function=len, # Fungsi untuk menghitung panjang teks
        add_start_index=False # Tidak perlu start index untuk kasus ini
    )

    all_chunks_with_metadata = []
    if not docs_data:
        return all_chunks_with_metadata

    for doc_info in docs_data:
        cleaned_page_text = clean_text(doc_info['text'])
        if not cleaned_page_text:
            continue

        split_chunks_text = text_splitter.split_text(cleaned_page_text)

        for chunk_text in split_chunks_text:
            stripped_chunk = chunk_text.strip()
            # Pastikan chunk memiliki konten yang cukup signifikan
            if len(stripped_chunk) > 50: # Minimal 50 karakter konten yang berarti
                all_chunks_with_metadata.append({
                    "content": stripped_chunk,
                    "source": doc_info['source'] # Sertakan metadata sumber
                })
    return all_chunks_with_metadata