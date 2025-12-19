# app/rag/ingest.py
import os
import argparse
import chromadb
from dotenv import load_dotenv

# LlamaIndex & LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse

# Load Env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not GOOGLE_API_KEY or not LLAMA_CLOUD_API_KEY:
    raise ValueError("Missing API Keys. Check your .env file.")


Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)


def get_crop_from_filename(filename: str) -> str:
    """Auto-tags documents based on filename keywords."""
    fname = filename.lower()
    if "tomato" in fname: return "Tomato"
    if "corn" in fname: return "Corn"
    if "rice" in fname: return "Rice"
    if "durian" in fname: return "Durian"
    return "General"


def process_file(filepath: str, parser: LlamaParse):
    """Parses a single file and returns tagged documents."""
    filename = os.path.basename(filepath)
    crop_tag = get_crop_from_filename(filename)
    
    print(f"   📄 Parsing: {filename} (Tag: {crop_tag})...")
    
    # LlamaParse converts PDF -> Markdown
    docs = parser.load_data(filepath)
    
    # Apply Metadata to every chunk
    for doc in docs:
        doc.metadata["crop"] = crop_tag
        doc.metadata["filename"] = filename
        
    return docs


def build_knowledge_base(target_path: str = None):
    print("--- [RAG BUILDER] Starting Ingestion ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_db_dir = os.path.join(base_dir, "data", "chroma_db")
    
    if target_path:
        if os.path.isabs(target_path):
            input_path = target_path
        else:
            input_path = os.path.join(os.getcwd(), target_path)
    else:
        raise "Missing target: --target <file_path>"

    # Init Parser
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True
    )

    documents = []

    # Process Files
    if os.path.isfile(input_path):
        # Single file input
        documents.extend(process_file(input_path, parser))
        
    elif os.path.isdir(input_path):
        # Directory Input
        print(f"   📂 Scanning directory: {input_path}")
        files = [f for f in os.listdir(input_path) if f.lower().endswith(".pdf")]
        
        if not files:
            print("   ⚠️ No PDF files found in directory.")
            return

        for f in files:
            full_path = os.path.join(input_path, f)
            documents.extend(process_file(full_path, parser))
    else:
        print(f"   ❌ Error: Path not found: {input_path}")
        return

    if not documents:
        print("   ⚠️ No documents to ingest.")
        return

    print(f"   📊 Total Parsed Chunks: {len(documents)}")

    # 4. Setup ChromaDB (Persistent Storage)
    print(f"   💾 Saving to ChromaDB at: {chroma_db_dir}")
    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db_client.get_or_create_collection("agronomy_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Indexing (Embeddings)
    # Note: We use 'from_documents' to add to the existing store
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True,
        embed_model=Settings.embed_model
    )

    print("--- [SUCCESS] Knowledge Base Updated! ---")


if __name__ == "__main__":
    # Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Ingest PDF manuals into the RAG Vector Database.")
    parser.add_argument("--target", type=str, help="Path to a specific PDF file or a directory containing PDFs", default=None)
    args = parser.parse_args()
    
    build_knowledge_base(target_path=args.target)