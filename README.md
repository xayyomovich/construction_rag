# Construction Law RAG System 🏗️

Construction Law RAG System – Retrieval-Augmented Generation system developed for the Ministry of Construction of Uzbekistan to query construction laws and regulations in Uzbek. 
The system processes .doc/.docx legal documents, builds a semantic vector index, and provides accurate answers with source attribution including document name, article, and section. 
Built with Python using LlamaIndex, LlamaParse, ChromaDB, and OpenAI GPT-4, enabling fast semantic search, reliable citations, and an interactive query interface for legal professionals.
## 🎯 Purpose

This system allows legal professionals and ministry staff to:
- Ask questions about construction laws in Uzbek language
- Get accurate answers with citations (document name, article, section)
- Search across multiple legal documents instantly
- Reference specific regulations without manual document review

## ✨ Features

- **Uzbek Language Support**: Native support for O'zbek tili queries and responses
- **Smart Document Parsing**: Extracts text and tables from .doc/.docx files using LlamaParse
- **Semantic Search**: Finds relevant information even with different wording
- **Source Attribution**: Every answer includes document references
- **Interactive CLI**: User-friendly command-line interface
- **Cost-Effective**: Uses OpenAI GPT-4 for quality responses

## 🏗️ Architecture

```
Documents (.doc) → LlamaParse → Semantic Chunking → Vector Index (ChromaDB)
                                                            ↓
User Query → Embedding → Similarity Search → GPT-4 → Formatted Answer + Sources
```

## 📋 Requirements

- Python 3.9+
- OpenAI API key
- LlamaCloud API key

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd construction-law-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
LLAMA_CLOUD_API_KEY=llx-your-key-here
```

### 3. Add Documents

Place your .doc/.docx files in `construction_docs/` folder.

### 4. Process Documents (One-time)

```bash
python data_ingestion.py
```

This creates a searchable index from your documents (~2-5 minutes).

### 5. Query System

```bash
python query_system.py
```

Choose interactive mode and ask questions in Uzbek:
```
❓ Savolingiz: Iskala xavfsizligi talablari?
```

## 📁 Project Structure

```
construction-law-rag/
├── data_ingestion.py       # Document processing & indexing
├── query_system.py         # Query interface
├── prompt_templates.py     # Advanced prompt engineering (optional)
├── requirements.txt        # Dependencies
├── .env                   # API keys (not in repo)
├── construction_docs/     # Your .doc files
└── storage/              # Generated index (auto-created)
```

## 💡 Usage Examples

**Interactive Mode:**
```bash
python query_system.py
> 1  # Select interactive mode
> Qurilish chiqindilarini utilizatsiya qilish qoidalari?
```

**Response Format:**
```
📋 QISQA JAVOB: Brief answer

📖 BATAFSIL TUSHUNTIRISH:
Detailed explanation

📌 MANBALAR:
1. Document_Name.doc, Article 15, Section 3
2. Regulation_2024.doc, Article 8
```

## 🛠️ Technical Stack

- **LlamaIndex**: RAG framework
- **OpenAI GPT-4**: Answer generation
- **LlamaParse**: Document parsing (tables, complex layouts)
- **ChromaDB**: Vector storage
- **OpenAI Embeddings**: Text embeddings

---

Built for O'zbekiston Respublikasi Qurilish Vazirligi 🇺🇿
