"""
Query System Module
Handles querying the pre-built index
Run this to ask questions about construction laws
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

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


class LegalQuerySystem:
    """Handles querying the construction law document index"""
    
    def __init__(
        self,
        persist_dir: str = "./storage",
        collection_name: str = "construction_laws"
    ):
        """
        Initialize query system
        
        Args:
            persist_dir: Path where index is stored
            collection_name: ChromaDB collection name
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.index = None
        self.query_engine = None
        self.chat_history = []
        
        print("✓ Query system initialized")
    
    def load_index(self):
        """Load existing index from storage"""
        if not os.path.exists(self.persist_dir):
            raise ValueError(
                f"❌ Indeks topilmadi: {self.persist_dir}\n"
                "   Avval data_ingestion.py ni ishga tushiring!"
            )
        
        print(f"\n⏳ Indeks yuklanmoqda...")
        
        db = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = db.get_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        
        print(f"✓ Indeks muvaffaqiyatli yuklandi!")
        print(f"  Chunk'lar soni: {chroma_collection.count()}")
    
    def setup_query_engine(self, similarity_top_k: int = 5):
        """
        Setup query engine with legal prompts
        
        Args:
            similarity_top_k: Number of chunks to retrieve
        """
        if self.index is None:
            self.load_index()
        
        # Legal prompt in Uzbek
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
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            text_qa_template=qa_prompt,
            response_mode="compact"
        )
        
        print(f"✓ Query engine tayyor! (Top {similarity_top_k} chunks)")
    
    def query(self, question: str, show_sources: bool = True) -> Dict:
        """
        Query the system
        
        Args:
            question: User question in Uzbek
            show_sources: Whether to display source information
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.query_engine is None:
            self.setup_query_engine()
        
        print(f"\n{'='*80}")
        print(f"❓ SAVOL: {question}")
        print(f"{'='*80}")
        
        # Query the system
        response = self.query_engine.query(question)
        answer = str(response)
        
        # Extract sources
        # sources = []
        # for i, node in enumerate(response.source_nodes, 1):
        #     metadata = node.node.metadata
        #     sources.append({
        #         'rank': i,
        #         'file_name': metadata.get('file_name', 'Noma\'lum hujjat'),
        #         'score': node.score,
        #         'text': node.node.text[:300]
        #     })
        
        # Display answer
        print(f"\n{answer}\n")
        
        # Display sources if requested
        # if show_sources and sources:
        #     print(f"{'='*80}")
        #     print(f"📚 MANBALAR ({len(sources)} ta)")
        #     print(f"{'='*80}")
            
        #     for src in sources:
        #         print(f"\n{src['rank']}. {src['file_name']}")
        #         print(f"   Mos kelish: {src['score']:.1%}")
        #         print(f"   Matn: {src['text'][:150]}...")
        
        # Save to history
        result = {
            'question': question,
            'answer': answer,
            # 'sources': sources,
            # 'num_sources': len(sources)
        }
        self.chat_history.append(result)
        
        return result
    
    def batch_query(self, questions: List[str]):
        """
        Process multiple questions
        
        Args:
            questions: List of questions
        """
        results = []
        
        print(f"\n{'='*80}")
        print(f"📝 BATCH QUERY - {len(questions)} ta savol")
        print(f"{'='*80}")
        
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            result = self.query(q, show_sources=False)
            results.append(result)
            print(f"\n{'-'*80}")
        
        return results
    
    def interactive_mode(self):
        """Interactive query mode"""
        print(f"\n{'='*80}")
        print("💬 INTERAKTIV REJIM")
        print(f"{'='*80}")
        print("\nSavollaringizni bering (chiqish uchun 'exit' yozing)")
        print("Masalan: 'Iskala xavfsizligi talablari?'\n")
        
        while True:
            try:
                question = input("❓ Savolingiz: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'chiqish', 'q']:
                    print("\n👋 Xayr!")
                    break
                
                if question.lower() == 'history':
                    self.show_history()
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Xayr!")
                break
            except Exception as e:
                print(f"\n❌ Xatolik: {e}")
    
    # def show_history(self):
    #     """Display query history"""
    #     if not self.chat_history:
    #         print("\n📝 Hali savol berilmagan")
    #         return
        
    #     print(f"\n{'='*80}")
    #     print(f"📜 SO'ROVLAR TARIXI ({len(self.chat_history)} ta)")
    #     print(f"{'='*80}")
        
    #     for i, item in enumerate(self.chat_history, 1):
    #         print(f"\n{i}. {item['question']}")
    #         print(f"   Javob: {item['answer'][:100]}...")
    #         print(f"   Manbalar: {item['num_sources']} ta")


def main():
    """Main execution for query system"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║    O'ZBEKISTON QURILISH VAZIRLIGI - SO'ROV TIZIMI            ║
║              Legal Query System                               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    query_system = LegalQuerySystem(
        persist_dir="./storage",
        collection_name="construction_laws"
    )
    
    # Load index
    try:
        query_system.load_index()
        query_system.setup_query_engine(similarity_top_k=5)
    except ValueError as e:
        print(f"\n{e}")
        return
    
    print("\n" + "="*80)
    print("Tizim tayyor! Quyidagi rejimlardan birini tanlang:")
    print("="*80)
    print("\n1. Interaktiv rejim (savol-javob)")
    print("2. Test savollari (demo)")
    print("3. Chiqish")
    
    choice = input("\nTanlovingiz (1/2/3): ").strip()
    
    if choice == "1":
        query_system.interactive_mode()
        
    elif choice == "2":
        # Demo questions
        demo_questions = [
             "Nechta Kompyuter (NP ENVY Desktop — 795-0030qd) bor?",
            "A.I. Ikramov kim?"
        ]
        
        print("\n📝 Demo savollar:")
        results = query_system.batch_query(demo_questions)
        
        print(f"\n{'='*80}")
        print(f"✓ {len(results)} ta savol muvaffaqiyatli javob oldi!")
        print(f"{'='*80}")
        
    else:
        print("\n👋 Xayr!")


if __name__ == "__main__":
    main()