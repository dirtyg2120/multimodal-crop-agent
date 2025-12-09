# app/rag/engine.py

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
# from google.genai.types import EmbedContentConfig

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)
Settings.llm = GoogleGenAI(
    model_name="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)

def get_query_engine():
    """
    Returns a query engine ready to search the vector database.
    """
    # Define Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_db_dir = os.path.join(base_dir, "data", "chroma_db")

    # Connect to existing database
    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db_client.get_or_create_collection("agronomy_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the Index from the Vector Store (No re-embedding!)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model
    )

    # Create Engine
    # similarity_top_k=3 means "Give me the 3 most relevant chunks"
    return index.as_query_engine(similarity_top_k=3)

# Test function for debugging
if __name__ == "__main__":
    engine = get_query_engine()
    response = engine.query("How do I treat Early Blight on tomatoes?")
    print("\n--- RETRIEVED ANSWER ---")
    print(response)