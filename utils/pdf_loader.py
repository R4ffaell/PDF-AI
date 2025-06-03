import fitz  # PyMuPDF
import streamlit as st

@st.cache_data # Cache hasil pembacaan PDF
def load_pdf_data(uploaded_files):
    """
    Memuat konten teks dan metadata (nama file, nomor halaman) dari file PDF yang diunggah.
    Mengembalikan list of dictionaries, di mana setiap dictionary berisi:
    - 'text': Konten teks dari sebuah halaman.
    - 'source': String yang menandakan nama file dan nomor halaman.
    """
    docs_data = []
    if not uploaded_files:
        return docs_data

    for uploaded_file in uploaded_files: # Tidak perlu file_index jika tidak digunakan
        try:
            # Gunakan BytesIO untuk membaca dari UploadedFile object
            # file_stream = io.BytesIO(uploaded_file.getvalue())
            # pdf_document = fitz.open(stream=file_stream, filetype="pdf")
            # Atau cara yang lebih sederhana jika PyMuPDF versi baru mendukung langsung:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                if page_text.strip(): # Hanya tambahkan jika ada teks di halaman
                    docs_data.append({
                        "text": page_text,
                        "source": f"{uploaded_file.name} (Hal. {page_num + 1})"
                    })
            pdf_document.close()
        except Exception as e:
            st.error(f"Error memproses file {uploaded_file.name}: {e}")
            # Pertimbangkan untuk tidak melanjutkan jika ada error fatal, atau log errornya
            continue # Lanjutkan ke file berikutnya jika ada error
    return docs_data