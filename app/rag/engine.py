import os
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.llms.google_genai import GoogleGenAI
from google.genai.types import EmbedContentConfig

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY,
    embedding_config=EmbedContentConfig(output_dimensionality=768)
)
Settings.llm = GoogleGenAI(
    model_name="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY
)

def get_query_engine(crop_filter: str = None):
    """
    Returns a query engine. 
    If crop_filter is provided (e.g., 'Tomato'), it will strictly limit search
    to documents tagged with that crop.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_db_dir = os.path.join(base_dir, "data", "chroma_db")

    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db_client.get_or_create_collection("agronomy_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model
    )

    # Define Filters
    filters = None
    if crop_filter:
        print(f"   ⚙️  [RAG Engine] Applying Filter: crop == {crop_filter}")
        filters = MetadataFilters(
            filters=[MetadataFilter(key="crop", value=crop_filter)]
        )

    return index.as_query_engine(
        similarity_top_k=3,
        filters=filters,
    )

if __name__ == "__main__":
    engine = get_query_engine("Tomato")
    response = engine.query("Treatment Early blight in Tomato")
    print(response)