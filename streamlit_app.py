"""
Streamlit UI for Uzbekistan Ministry of Construction RAG System
Beautiful GPT-like interface for querying construction law documents
"""

import streamlit as st
import os
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime

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

# Page configuration
st.set_page_config(
    page_title="O'zbekiston Qurilish Vazirligi - Yuridik Yordamchi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Source cards */
    .source-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .source-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .source-score {
        color: #10b981;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: white;
    }
    
    /* Stats boxes */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_query_system():
    """Initialize and cache the query system"""
    
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Load index
    persist_dir = "./storage"
    collection_name = "construction_laws"
    
    if not os.path.exists(persist_dir):
        st.error("❌ Indeks topilmadi! Avval data_ingestion.py ni ishga tushiring.")
        st.stop()
    
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

Javob formatı:
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


def query_documents(query_engine, question: str) -> Dict:
    """Query the RAG system and return results"""
    response = query_engine.query(question)
    
    # Extract sources
    sources = []
    for i, node in enumerate(response.source_nodes, 1):
        metadata = node.node.metadata
        sources.append({
            'rank': i,
            'file_name': metadata.get('file_name', 'Noma\'lum hujjat'),
            'score': node.score,
            'text': node.node.text[:500]
        })
    
    return {
        'answer': str(response),
        'sources': sources
    }


def display_sources(sources: List[Dict]):
    """Display source documents in collapsed expander"""
    st.markdown("---")
    
    # Collapsed expander - user can expand if they want to see sources
    with st.expander(f"📚 Manba Hujjatlar ({len(sources)} ta)", expanded=False):
        for src in sources:
            st.markdown(f"**📄 {src['file_name']}**")
            st.caption(f"✓ Mos kelish: {src['score']:.1%}")
            st.markdown("")  # spacing


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ O'zbekiston Qurilish Vazirligi</h1>
        <p>Yuridik Yordamchi - Sun'iy Intellekt Asosida</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    # Initialize query system
    try:
        query_engine, collection = initialize_query_system()
        doc_count = collection.count()
    except Exception as e:
        st.error(f"❌ Tizimni yuklashda xatolik: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Tizim Ma'lumotlari")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{doc_count}</div>
                <div class="stat-label">Hujjat Chunk'lari</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{st.session_state.query_count}</div>
                <div class="stat-label">So'rovlar</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### 🎛️ Sozlamalar")
        show_sources = st.checkbox("Manbalarni ko'rsatish", value=True)
        
        st.markdown("---")
        
        # Example questions
        st.markdown("### 💡 Misol Savol")
        example_questions = [
            "Nechta Kompyuter (NP ENVY Desktop — 795-0030qd) bor??"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Suhbatni Tozalash", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.8rem;">
            <p>🏛️ O'zbekiston Respublikasi</p>
            <p>Qurilish va Uy-joy Kommunal</p>
            <p>Xizmati Vazirligi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and show_sources:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Savolingizni kiriting..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Javob tayyorlanmoqda..."):
                try:
                    result = query_documents(query_engine, prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display sources in collapsed expander
                    if show_sources and result['sources']:
                        display_sources(result['sources'])
                    
                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                    st.session_state.query_count += 1
                    
                except Exception as e:
                    st.error(f"❌ Xatolik: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem;">
        <p>⚡ Powered by LlamaIndex, OpenAI & ChromaDB</p>
        <p>© 2024 O'zbekiston Respublikasi Qurilish va Uy-joy Kommunal Xizmati Vazirligi</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()