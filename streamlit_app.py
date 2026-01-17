"""
Streamlit UI for Uzbekistan Ministry of Construction RAG System
Professional interface with processing events and status updates
"""

import os
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime
import time
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('punkt_tab', download_dir='/tmp/nltk_data', quiet=True)

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables
load_dotenv()

# Try to get secrets from Streamlit secrets (for cloud deployment)
def get_api_key(key_name: str) -> str:
    """Get API key from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError, AttributeError):
        # Fall back to environment variables (for local development)
        return os.getenv(key_name)

# Page configuration
st.set_page_config(
    page_title="Qurilish Vazirligi - AI Yordamchi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 0.8rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    .stButton button {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        text-align: left;
    }
    
    .stButton button:hover {
        background: #f9fafb;
        border-color: #3b82f6;
        color: #3b82f6;
        transform: translateX(4px);
    }
    
    .streamlit-expanderHeader {
        background: #f9fafb;
        border-radius: 8px;
        font-weight: 500;
        color: #374151;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem 0;
    }
    
    .status-loading {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys():
    """Check if required API keys are configured"""
    openai_key = get_api_key("OPENAI_API_KEY")
    llama_key = get_api_key("LLAMA_CLOUD_API_KEY")
    
    missing_keys = []
    if not openai_key or openai_key == "sk-proj-your-actual-api-key-here":
        missing_keys.append("OPENAI_API_KEY")
    if not llama_key or llama_key == "llx-your-llamaparse-key-here":
        missing_keys.append("LLAMA_CLOUD_API_KEY")
    
    return missing_keys


@st.cache_resource(show_spinner=False)
def initialize_query_system():
    """Initialize and cache the query system"""
    
    try:
        # Check API keys first
        missing = check_api_keys()
        if missing:
            raise ValueError(f"❌ API kalitlari topilmadi: {', '.join(missing)}\n\n"
                           "Iltimos, .env faylini tekshiring!")
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=get_api_key("OPENAI_API_KEY")
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=get_api_key("OPENAI_API_KEY")
        )
        
        # Load index
        persist_dir = os.getenv("PERSIST_DIR", "./storage")
        collection_name = os.getenv("COLLECTION_NAME", "construction_laws")
        
        if not os.path.exists(persist_dir):
            raise ValueError(
                f"❌ Indeks topilmadi: {persist_dir}\n\n"
                "Avval data_ingestion.py ni ishga tushiring!"
            )
        
        # Initialize ChromaDB
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Setup query engine with legal prompt
        qa_prompt_str = """Siz O'zbekiston Respublikasi Qurilish vazirligining yuridik yordamchisisiz.

Sizning vazifangiz:
1. Qurilish qonunlari bo'yicha aniq ma'lumot berish
2. Har doim manba ko'rsatish (hujjat nomi, modda/bo'lim raqami)
3. Majburiy va tavsiya talablarni farqlash
4. Agar ma'lumot yo'q bo'lsa, halollik bilan tan olish

Kontekst ma'lumotlari:
{context_str}

Savol: {query_str}

Javob formatі:
📋 QISQA JAVOB: [1-2 jumla asosiy javob]

📖 BATAFSIL TUSHUNTIRISH:
[To'liq ma'lumot]

📌 MANBALAR:
[Hujjat nomlari va moddalar]

Javob:"""
        
        qa_prompt = PromptTemplate(qa_prompt_str)
        
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=qa_prompt,
            response_mode="compact"
        )
        
        return query_engine, chroma_collection
        
    except Exception as e:
        st.error(f"Xatolik: {str(e)}")
        raise e


def query_documents(query_engine, question: str, status_placeholder) -> Dict:
    """Query the RAG system with status updates"""
    
    try:
        # Step 1: Retrieving relevant chunks
        status_placeholder.markdown(
            '<div class="status-badge status-loading">🔍 1/3 - Tegishli hujjatlar qidirilmoqda...</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.3)  # Brief pause for visual feedback
        
        # Step 2: Processing
        status_placeholder.markdown(
            '<div class="status-badge status-loading">⚙️ 2/3 - Ma\'lumotlar tahlil qilinmoqda...</div>',
            unsafe_allow_html=True
        )
        
        # Query the system
        response = query_engine.query(question)
        
        # Step 3: Generating answer
        status_placeholder.markdown(
            '<div class="status-badge status-loading">✍️ 3/3 - Javob tayyorlanmoqda...</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.2)
        
        # Extract sources
        sources = []
        for i, node in enumerate(response.source_nodes, 1):
            metadata = node.node.metadata
            sources.append({
                'rank': i,
                'file_name': metadata.get('file_name', 'Noma\'lum hujjat'),
                'score': node.score
            })
        
        # Success
        status_placeholder.markdown(
            '<div class="status-badge status-success">✅ Tayyor!</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.5)
        status_placeholder.empty()  # Clear the status
        
        return {
            'answer': str(response),
            'sources': sources,
            'success': True
        }
        
    except Exception as e:
        status_placeholder.markdown(
            f'<div class="status-badge status-error">❌ Xatolik: {str(e)}</div>',
            unsafe_allow_html=True
        )
        return {
            'answer': f"Kechirasiz, javob tayyorlashda xatolik yuz berdi: {str(e)}",
            'sources': [],
            'success': False
        }


def display_sources(sources: List[Dict]):
    """Display source documents in collapsed expander"""
    with st.expander(f"📚 Manba Hujjatlar ({len(sources)} ta)", expanded=False):
        for src in sources:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**📄 {src['file_name']}**")
            with col2:
                st.markdown(f"<span class='badge'>{src['score']:.0%} mos</span>", unsafe_allow_html=True)
            st.markdown("")


def process_question(question: str):
    """Add question to chat and trigger response"""
    if question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending_query = question
        st.session_state.query_count += 1


def main():
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ O'zbekiston Qurilish va Uy-joy Kommunal Xizmati Vazirligi</h1>
        <p>Sun'iy Intellekt Asosidagi Yuridik Yordamchi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    
    # Check API keys before initializing
    missing_keys = check_api_keys()
    if missing_keys:
        st.error(f"""
        ❌ **API Kalitlari topilmadi!**
        
        Quyidagi kalitlar topilmadi:
        {', '.join([f'`{k}`' for k in missing_keys])}
        
        **Streamlit Cloud'da:**
        1. App Settings ⚙️ → Secrets ga o'ting
        2. Quyidagi formatda kiriting:
        ```toml
        OPENAI_API_KEY = "sk-proj-..."
        LLAMA_CLOUD_API_KEY = "llx-..."
        ```
        
        **Local development uchun:**
        1. `.env` faylini yarating
        2. Kalitlarni qo'shing:
        ```
        OPENAI_API_KEY=sk-proj-...
        LLAMA_CLOUD_API_KEY=llx-...
        ```
        """)
        st.stop()
    
    # Initialize query system
    with st.spinner("⏳ Tizim yuklanmoqda..."):
        try:
            query_engine, collection = initialize_query_system()
            doc_count = collection.count()
        except Exception as e:
            st.error(f"❌ Tizimni yuklashda xatolik:\n\n{str(e)}")
            st.info("💡 Iltimos, storage/ papkasining mavjudligini va API kalitlarini tekshiring.")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Tizim Ma'lumotlari")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{doc_count}</div>
                <div class="stat-label">Hujjatlar</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.query_count}</div>
                <div class="stat-label">So'rovlar</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Settings
        st.markdown("## 🎛️ Sozlamalar")
        show_sources = st.checkbox("📚 Manbalarni ko'rsatish", value=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Example questions
        st.markdown("## 💡 Misol Savollar")
        
        example_questions = [
            "Nechta Kompyuter (NP ENVY Desktop — 795-0030qd) bor?",
            "A.I. Ikramov kim?",
            "Qurilish litsenziyasi olish talablari qanday?"
        ]
        
        for q in example_questions:
            # Use unique key and callback
            if st.button(q, key=f"example_{hash(q)}", use_container_width=True):
                process_question(q)
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Suhbatni Tozalash", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.pending_query = None
            st.rerun()
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.8rem; line-height: 1.6;">
            <p><strong>🏛️ O'zbekiston Respublikasi</strong></p>
            <p>Qurilish va Uy-joy Kommunal<br>Xizmati Vazirligi</p>
            <p style="margin-top: 1rem; font-size: 0.7rem;">Powered by GPT-4o</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and show_sources:
                display_sources(message["sources"])
    
    # Handle pending query (from example buttons)
    if st.session_state.pending_query:
        question = st.session_state.pending_query
        st.session_state.pending_query = None
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            result = query_documents(query_engine, question, status_placeholder)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Display sources
            if show_sources and result['sources']:
                display_sources(result['sources'])
            
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
    
    # Chat input
    if prompt := st.chat_input("💬 Savolingizni kiriting..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.query_count += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            result = query_documents(query_engine, prompt, status_placeholder)
            
            # Display answer
            st.markdown(result['answer'])
            
            # Display sources
            if show_sources and result['sources']:
                display_sources(result['sources'])
            
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
    
    # Welcome message if no chat history
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #6b7280;">
            <h3 style="color: #374151; margin-bottom: 1rem;">👋 Xush kelibsiz!</h3>
            <p style="font-size: 1rem; line-height: 1.6;">
                Qurilish qonunlari va me'yorlari haqida savollaringizni bering.<br>
                Misol uchun chap tarafdagi tayyyor savollardan foydalanishingiz mumkin.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()