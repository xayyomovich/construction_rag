"""
Data Ingestion Module
Handles document loading, parsing, and index creation
Run this once to process and index your documents
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
import chromadb

# Load environment variables
load_dotenv()


class DocumentProcessor:
    """Handles document loading and index creation"""
    
    def __init__(
        self,
        docs_path: str = "./construction_docs",
        persist_dir: str = "./storage",
        collection_name: str = "construction_laws"
    ):
        """
        Initialize document processor
        
        Args:
            docs_path: Path to folder containing .doc files
            persist_dir: Path to store the index
            collection_name: Name for ChromaDB collection
        """
        self.docs_path = docs_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
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
        
        # Configure LlamaParse for Uzbek documents
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
            language="uz"
        )
        
        print("✓ Document processor initialized")
        print(f"  Documents path: {self.docs_path}")
        print(f"  Storage path: {self.persist_dir}")
    
    def load_documents(self) -> List:
        """Load and parse all .doc and .docx files"""
        print(f"\n{'='*80}")
        print("HUJJATLARNI YUKLASH")
        print(f"{'='*80}")
        
        # Check if directory exists
        if not os.path.exists(self.docs_path):
            raise ValueError(f"Hujjatlar papkasi topilmadi: {self.docs_path}")
        
        # Count files
        doc_files = []
        for ext in [".doc", ".docx"]:
            doc_files.extend(Path(self.docs_path).rglob(f"*{ext}"))
        
        if not doc_files:
            raise ValueError(f"Hech qanday .doc yoki .docx fayl topilmadi: {self.docs_path}")
        
        print(f"\n📄 Topilgan hujjatlar: {len(doc_files)} ta")
        for i, file in enumerate(doc_files, 1):
            print(f"   {i}. {file.name}")
        
        print(f"\n⏳ LlamaParse yordamida tahlil qilinmoqda...")
        print("   (Bu biroz vaqt olishi mumkin - jadvallar va murakkab formatlar tahlil qilinmoqda)")
        
        # Configure file extractor
        file_extractor = {
            ".docx": self.parser,
            ".doc": self.parser
        }
        
        # Load documents
        reader = SimpleDirectoryReader(
            input_dir=self.docs_path,
            required_exts=[".doc", ".docx"],
            file_extractor=file_extractor,
            recursive=True
        )
        
        documents = reader.load_data()
        
        print(f"\n✓ Muvaffaqiyatli yuklandi: {len(documents)} ta hujjat")
        
        # Show document stats
        total_chars = sum(len(doc.text) for doc in documents)
        print(f"✓ Jami belgilar: {total_chars:,}")
        print(f"✓ O'rtacha hujjat hajmi: {total_chars // len(documents):,} belgi")
        
        return documents
    
    def create_index(self, force_rebuild: bool = False):
        """
        Create vector index from documents
        
        Args:
            force_rebuild: If True, rebuild even if index exists
        """
        print(f"\n{'='*80}")
        print("INDEKS YARATISH")
        print(f"{'='*80}")
        
        # Check if index exists
        if not force_rebuild and os.path.exists(self.persist_dir):
            print("\n⚠️  Indeks allaqachon mavjud!")
            print(f"   Path: {self.persist_dir}")
            print("\n   Qayta yaratish uchun: force_rebuild=True qiling")
            return
        
        print("\n📚 Yangi indeks yaratilmoqda...")
        
        # Load documents
        documents = self.load_documents()
        
        # Configure semantic chunking
        print(f"\n⚙️  Semantik chunking sozlanmoqda...")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        
        # Initialize ChromaDB
        print(f"\n💾 ChromaDB sozlanmoqda...")
        db = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = db.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        print(f"\n🔄 Vektorlash va indekslash boshlandi...")
        print("   (Bu jarayon 2-5 daqiqa davom etishi mumkin)")
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True
        )
        
        print(f"\n{'='*80}")
        print("✓ INDEKS MUVAFFAQIYATLI YARATILDI!")
        print(f"{'='*80}")
        print(f"\n📁 Saqlangan joy: {self.persist_dir}")
        print(f"📊 Collection: {self.collection_name}")
        print(f"✓ Endi query_system.py orqali so'rov yuborishingiz mumkin!")
        
        return index
    
    def get_index_info(self):
        """Display information about existing index"""
        if not os.path.exists(self.persist_dir):
            print(f"\n❌ Indeks topilmadi: {self.persist_dir}")
            print("   Avval create_index() ni chaqiring")
            return
        
        print(f"\n{'='*80}")
        print("INDEKS MA'LUMOTLARI")
        print(f"{'='*80}")
        
        db = chromadb.PersistentClient(path=self.persist_dir)
        try:
            collection = db.get_collection(self.collection_name)
            count = collection.count()
            
            print(f"\n✓ Indeks mavjud")
            print(f"📁 Path: {self.persist_dir}")
            print(f"📊 Collection: {self.collection_name}")
            print(f"📄 Chunk'lar soni: {count}")
            
        except Exception as e:
            print(f"\n⚠️  Xatolik: {e}")


def main():
    """Main execution for document processing"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  O'ZBEKISTON QURILISH VAZIRLIGI - HUJJATLARNI QAYTA ISHLASH  ║
║              Document Ingestion System                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize processor
    processor = DocumentProcessor(
        docs_path="./construction_docs",
        persist_dir="./storage"
    )
    
    # Check if index already exists
    if os.path.exists("./storage"):
        print("\n⚠️  DIQQAT: Indeks allaqachon mavjud!")
        print("\nTanlovlar:")
        print("1. Mavjud indeks haqida ma'lumot ko'rish")
        print("2. Indeksni qayta yaratish (barcha ma'lumotlar o'chiriladi)")
        print("3. Chiqish")
        
        choice = input("\nTanlovingiz (1/2/3): ").strip()
        
        if choice == "1":
            processor.get_index_info()
        elif choice == "2":
            confirm = input("\n⚠️  Haqiqatan ham qayta yaratmoqchimisiz? (ha/yo'q): ").strip().lower()
            if confirm == "ha":
                processor.create_index(force_rebuild=True)
            else:
                print("\n❌ Bekor qilindi")
        else:
            print("\n👋 Xayr!")
    else:
        # Create new index
        processor.create_index(force_rebuild=False)
    
    print("\n" + "="*80)
    print("Keyingi qadam: python query_system.py")
    print("="*80)


if __name__ == "__main__":
    main()