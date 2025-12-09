# app/rag/ingest.py

import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

def build_knowledge_base():
    print("--- [RAG BUILDER] Starting Knowledge Base Construction ---")

    # Path Setup
    # We go up two levels from 'app/rag/' to reach the root, then into 'data'
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manuals_dir = os.path.join(base_dir, "data", "manuals")
    chroma_db_dir = os.path.join(base_dir, "data", "chroma_db")

    print(f"1. Reading PDFs from: {manuals_dir}")
    if not os.path.exists(manuals_dir) or not os.listdir(manuals_dir):
        print(f"ERROR: No PDFs found in {manuals_dir}. Please add a file.")
        return

    # 3. Load Documents
    # SimpleDirectoryReader automatically parses PDFs, Text files, etc.
    documents = SimpleDirectoryReader(manuals_dir).load_data()
    print(f"   > Loaded {len(documents)} document pages.")

    # 4. Setup Vector Database (Chroma)
    print(f"2. Initializing ChromaDB at: {chroma_db_dir}")
    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    
    # We create a collection named 'agronomy_manuals'
    chroma_collection = db_client.get_or_create_collection("agronomy_manuals")
    
    # 5. Connect LlamaIndex to Chroma
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Build the Index (This is the heavy lifting)
    print("3. Generating Embeddings (This may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True,
        embed_model=Settings.embed_model
    )

    print("--- [SUCCESS] Knowledge Base Built & Saved! ---")
    print("You can now run 'app/rag/engine.py' to query this data.")

if __name__ == "__main__":
    build_knowledge_base()