# utils.py

import streamlit as st # Diperlukan untuk @st.cache_resource, @st.cache_data, st.error, st.spinner, st.caption, st.session_state
from transformers import pipeline as hf_pipeline
import torch
import re
from typing import List, Dict
import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer # Tidak diimpor langsung jika hanya dipakai via HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document # Pastikan Document diimpor

# Tidak perlu load_dotenv() di sini jika sudah ada di main_app.py sebelum impor utils

@st.cache_resource
def load_langchain_embedding_model(model_name: str):
    """Load and cache embedding model as a Langchain compatible object"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if model_name.startswith("sentence-transformers/"):
            model_kwargs = {'device': device}
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            st.caption(f"Embedding model '{model_name}' loaded on {device} via Langchain wrapper.")
            return embeddings
        else:
            st.error(f"Unsupported embedding model type: {model_name}")
            return None
    except Exception as e:
        st.error(f"Failed to load Langchain embedding model '{model_name}': {e}")
        return None

@st.cache_resource
def load_langchain_llm(model_key: str):
    """Load and cache LLM as a Langchain compatible object"""
    device = 0 if torch.cuda.is_available() else -1
    try:
        model_name_hf = model_key
        model_kwargs_default = {
            "max_length": 512, "temperature": 0.7, "do_sample": True,
            "pad_token_id": 50256
        }
        task_specific_kwargs = {}

        if "flan-t5" in model_name_hf:
            task = "text2text-generation"
            task_specific_kwargs = {"max_length": 512, "temperature": 0.7, "do_sample": True}
        elif "DialoGPT" in model_name_hf or "gpt" in model_name_hf:
            task = "text-generation"
        else:
            task = "text-generation"

        final_model_kwargs = {**model_kwargs_default, **task_specific_kwargs}
        # Untuk DialoGPT dan GPT, pad_token_id mungkin sudah ada di tokenizer. Hapus jika menyebabkan error.
        # Jika model tidak memiliki pad_token_id dan task membutuhkannya, pipeline akan error.
        # Sebaiknya pastikan tokenizer memiliki pad_token_id atau set `tokenizer.pad_token_id = tokenizer.eos_token_id`
        # Namun, untuk Langchain HuggingFacePipeline, ini biasanya ditangani.

        try:
            hf_pipe = hf_pipeline(
                task, model=model_name_hf, device=device, **final_model_kwargs
            )
            llm = HuggingFacePipeline(pipeline=hf_pipe, model_kwargs={"temperature": 0.7}) # model_kwargs di sini untuk pemanggilan, bukan pembuatan pipeline
            st.caption(f"LLM '{model_name_hf}' ({task}) loaded on device {device} via Langchain wrapper.")
            return llm
        except Exception as pipeline_error:
            st.error(f"Error creating pipeline for {model_name_hf}: {pipeline_error}")
            st.warning("Falling back to distilgpt2...")
            hf_pipe = hf_pipeline("text-generation", model="distilgpt2", device=device, max_length=256)
            llm = HuggingFacePipeline(pipeline=hf_pipe, model_kwargs={"temperature": 0.7})
            return llm
            
    except Exception as e:
        st.error(f"Failed to load Langchain LLM for '{model_key}': {e}")
        return None

def clean_and_preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\f\r\x0b\x0c\x00-\x08\x0e-\x1f\x7f-\x9f]+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)
    return text.strip()

@st.cache_data
def extract_pdf_content(uploaded_files) -> List[Dict]:
    docs_data = []
    for uploaded_file in uploaded_files:
        try:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                if not page_text.strip():
                    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_WHITESPACE)
                    page_text_parts = []
                    for block in blocks.get("blocks", []):
                        if block.get("type") == 0:
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    page_text_parts.append(span.get("text", ""))
                    page_text = " ".join(page_text_parts)

                if page_text.strip():
                    cleaned_text = clean_and_preprocess_text(page_text)
                    if len(cleaned_text) > 50:
                        docs_data.append({
                            "page_content": cleaned_text,
                            "metadata": {
                                "source_file": uploaded_file.name,
                                "page_number": page_num + 1,
                                "doc_name": uploaded_file.name.replace('.pdf', '')
                            }
                        })
            pdf_document.close()
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue
    return docs_data

@st.cache_data
def split_into_langchain_documents(docs_data_with_page_content: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    if not docs_data_with_page_content:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    
    all_split_docs = []
    for doc_dict in docs_data_with_page_content:
        split_texts = text_splitter.split_text(doc_dict['page_content'])
        for i, text_chunk in enumerate(split_texts):
            if len(text_chunk.strip()) > 100:
                chunk_metadata = doc_dict['metadata'].copy() 
                chunk_metadata['chunk_id'] = f"{chunk_metadata['doc_name']}_p{chunk_metadata['page_number']}_c{i+1}"
                new_doc = Document(page_content=text_chunk.strip(), metadata=chunk_metadata)
                all_split_docs.append(new_doc)
    return all_split_docs

@st.cache_resource
def get_langchain_vectorstore(_langchain_documents, _embedding_model_obj):
    if not _langchain_documents or _embedding_model_obj is None:
        st.error("Cannot create vector store: documents or embedding model missing.")
        return None
    try:
        with st.spinner("🧠 Building vector store with Langchain FAISS..."):
            vectorstore = LangchainFAISS.from_documents(documents=_langchain_documents, embedding=_embedding_model_obj)
        st.caption("Vector store built successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"Error building Langchain FAISS vector store: {e}")
        return None

@st.cache_resource
def get_langchain_conversation_chain(_vectorstore, _llm_obj, _retrieval_k: int = 5):
    if _vectorstore is None or _llm_obj is None:
        st.error("Cannot create conversation chain: vector store or LLM missing.")
        return None
    try:
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )
        retriever = _vectorstore.as_retriever(search_kwargs={"k": _retrieval_k})
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=_llm_obj,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff"
        )
        st.caption("Conversational chain created successfully.")
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating Langchain conversation chain: {e}")
        return None

def handle_user_input_langchain(user_question):
    if st.session_state.conversation_chain is None:
        st.error("Conversation chain not initialized. Please process documents first.")
        return

    try:
        with st.spinner("思考中 (Thinking)..."):
            response = st.session_state.conversation_chain({'question': user_question})
        
        if 'display_chat_history' not in st.session_state:
            st.session_state.display_chat_history = []
        
        st.session_state.display_chat_history.append({"type": "user", "content": user_question})
        st.session_state.display_chat_history.append({"type": "bot", "content": response['answer']})

        if response.get('source_documents'):
            st.session_state.last_sources = response['source_documents'][:3]
        else:
            st.session_state.last_sources = []
                
    except Exception as e:
        st.error(f"Error processing question: {e}")
        st.error("Please try rephrasing your question or check if the models are properly loaded.")