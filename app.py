import streamlit as st
import torch # Hanya untuk torch.cuda.is_available() di info sistem
from dotenv import load_dotenv

# Impor dari file lokal Anda
from config import EMBEDDING_MODELS, LLM_MODELS
from utils import (
    load_langchain_embedding_model,
    load_langchain_llm,
    extract_pdf_content,
    split_into_langchain_documents,
    get_langchain_vectorstore,
    get_langchain_conversation_chain,
    handle_user_input_langchain
)

# Attempt to load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🤖 Advanced PDF Chat Assistant (Langchain Integrated)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat CSS eksternal
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Muat CSS dari file styles.css
load_css("styles.css") # Pastikan styles.css ada di direktori yang sama

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Advanced PDF Chat Assistant (Langchain Integrated)</h1>
    <p>Upload multiple PDFs and chat with your documents using advanced AI models and Langchain</p>
</div>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🤖 Model Selection")
    embedding_model_key = st.selectbox(
        "Embedding Model:",
        list(EMBEDDING_MODELS.keys()),
        format_func=lambda x: EMBEDDING_MODELS[x],
        help="Choose embedding model."
    )
    llm_model_key_selected = st.selectbox(
        "Language Model:",
        list(LLM_MODELS.keys()),
        format_func=lambda x: LLM_MODELS[x],
        help="Choose language model."
    )

    st.subheader("⚡ Advanced Settings")
    chunk_size_val = st.slider("Chunk Size (for splitting text)", 500, 2000, 1000, 100)
    chunk_overlap_val = st.slider("Chunk Overlap", 50, 500, 200, 50)
    retrieval_k_val = st.slider("Retrieved Chunks (for context)", 1, 10, 3, 1)

    st.subheader("💻 System Info")
    device_info = "🚀 GPU Available" if torch.cuda.is_available() else "💻 CPU Only"
    st.info(device_info)

    if st.button("🧹 Clear Chat History"):
        if 'display_chat_history' in st.session_state:
            st.session_state.display_chat_history = []
        if 'last_sources' in st.session_state:
            st.session_state.last_sources = []
        # Reset memori di chain jika sudah terinisialisasi
        if 'conversation_chain' in st.session_state and st.session_state.conversation_chain:
            st.session_state.conversation_chain.memory.clear()
        st.success("Chat history and memory cleared!")


    # Load models with proper error handling
    if 'loaded_embedding_model' not in st.session_state:
        st.session_state.loaded_embedding_model = None
    if 'loaded_llm' not in st.session_state:
        st.session_state.loaded_llm = None
    
    # Load models when selection changes
    # atau ketika tombol ditekan, untuk menghindari loading saat startup jika belum pernah dipilih
    if st.session_state.get('current_embedding_model') != embedding_model_key:
        with st.spinner(f"Loading embedding model {EMBEDDING_MODELS[embedding_model_key]}..."):
            st.session_state.loaded_embedding_model = load_langchain_embedding_model(embedding_model_key)
        st.session_state.current_embedding_model = embedding_model_key
        
    if st.session_state.get('current_llm_model') != llm_model_key_selected:
        with st.spinner(f"Loading LLM {LLM_MODELS[llm_model_key_selected]}..."):
            st.session_state.loaded_llm = load_langchain_llm(llm_model_key_selected)
        st.session_state.current_llm_model = llm_model_key_selected

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files_list = st.file_uploader(
        "📁 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to chat with."
    )
    
    if st.button("🚀 Process Documents & Initialize Chat"):
        if not uploaded_files_list:
            st.warning("Please upload PDF files first.")
        elif not st.session_state.loaded_embedding_model or not st.session_state.loaded_llm:
            st.error("Models not loaded. Check sidebar for errors or selection.")
        else:
            with st.spinner("🔄 Processing documents... This may take a moment."):
                raw_docs_data = extract_pdf_content(uploaded_files_list)
                if not raw_docs_data:
                    st.error("❌ No text could be extracted from the uploaded PDFs.")
                    st.stop()
                
                langchain_documents = split_into_langchain_documents(raw_docs_data, chunk_size_val, chunk_overlap_val)
                if not langchain_documents:
                    st.error("❌ No meaningful chunks (Langchain Documents) could be created.")
                    st.stop()
                
                st.metric("📄 Pages Processed (raw text entries)", len(raw_docs_data))
                st.metric("🧩 Langchain Docs Created (Chunks)", len(langchain_documents))
                
                vectorstore = get_langchain_vectorstore(langchain_documents, st.session_state.loaded_embedding_model)
                if not vectorstore:
                    st.error("❌ Failed to create vector store.")
                    st.stop()
                
                st.session_state.conversation_chain = get_langchain_conversation_chain(
                    vectorstore, 
                    st.session_state.loaded_llm,
                    retrieval_k_val
                )
                st.session_state.display_chat_history = [] # Reset display history
                st.session_state.last_sources = [] # Reset last sources
                st.success("✅ Documents processed and chat is ready!")

with col2:
    if uploaded_files_list:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📄 Files Uploaded", len(uploaded_files_list))
        st.markdown('</div>', unsafe_allow_html=True)
        try:
            total_size = sum([file.size for file in uploaded_files_list]) / (1024*1024)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📦 Total Size", f"{total_size:.1f} MB")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception: # Menangani jika file sudah di-close atau tidak bisa diakses sizenya lagi
             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
             st.metric("📦 Total Size", "N/A")
             st.markdown('</div>', unsafe_allow_html=True)


st.markdown("## 💬 Chat with your Documents")

if 'display_chat_history' not in st.session_state:
    st.session_state.display_chat_history = []

if st.session_state.display_chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.display_chat_history:
        if message["type"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Ask a question:", 
        disabled=not st.session_state.get("conversation_chain"),
        placeholder="Enter your question here...",
        key="question_input" # Menggunakan key yang berbeda dari submit button
    )
    
    col_send, col_status = st.columns([1, 3])
    with col_send:
        submitted = st.form_submit_button("📤 Send", use_container_width=True)
    
    with col_status:
        if not st.session_state.get("conversation_chain"):
            st.info("Please upload and process documents first")
        else:
            st.success("Ready to chat!")
    
    if submitted and user_question and st.session_state.get("conversation_chain"):
        handle_user_input_langchain(user_question)
        st.rerun() # Rerun untuk update chat display dan source context

if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
    with st.expander("📚 Source Context for Last Answer", expanded=True): # Default expander terbuka
        for i, doc in enumerate(st.session_state.last_sources):
            source_meta = doc.metadata
            st.markdown(f"**Source {i+1}:** {source_meta.get('doc_name', 'N/A')} (Page {source_meta.get('page_number', 'N/A')})")
            st.markdown(f"```\n{doc.page_content[:300]}...\n```")

if not uploaded_files_list:
    st.markdown("""
    ## 🚀 How to Use
    1. **Upload PDFs**: Use the uploader in the main panel.
    2. **Configure Models**: Choose embedding and language models from the sidebar. Adjust chunking if needed.
    3. **Process Documents**: Click "Process Documents & Initialize Chat".
    4. **Ask Questions**: Type your question in the form above and click "Send".
    
    ## 🔧 Model Recommendations
    - **For best results**: Use `all-mpnet-base-v2` for embeddings and `flan-t5-large` for generation
    - **For faster performance**: Use `all-MiniLM-L6-v2` for embeddings and `flan-t5-base` for generation
    
    ## 🛠️ Features
    - **Multiple PDF Support**: Upload and process multiple documents at once
    - **Advanced Chunking**: Configurable text splitting for optimal retrieval
    - **Chat History**: Persistent conversation history with clear option
    - **Source References**: See which document parts were used to answer your questions
    """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Langchain, and HuggingFace Transformers")