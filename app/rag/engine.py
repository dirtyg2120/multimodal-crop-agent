import logging
import os
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from google.genai.types import EmbedContentConfig

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY,
    embedding_config=EmbedContentConfig(output_dimensionality=768)
)
# NOTE: Settings.llm intentionally NOT set here.
# LlamaIndex is used only for retrieval — the agent's own LLM does synthesis.


@lru_cache(maxsize=1)
def _get_index() -> VectorStoreIndex:
    """Build the VectorStoreIndex once and cache it for the process lifetime."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_db_dir = os.path.join(base_dir, "data", "chroma_db")
    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db_client.get_or_create_collection("agronomy_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)


def retrieve_chunks(query: str, crop_filter: str = None, top_k: int = 3) -> str:
    """
    Retrieve the top-k relevant text chunks from ChromaDB.
    Returns raw passage text — no LLM synthesis step.
    """
    index = _get_index()

    filters = None
    if crop_filter:
        log.info(f"[RAG] Filter: crop == {crop_filter}")
        filters = MetadataFilters(filters=[MetadataFilter(key="crop", value=crop_filter)])

    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    nodes = retriever.retrieve(query)

    if not nodes:
        return ""

    passages = []
    for i, node in enumerate(nodes, 1):
        score = getattr(node, "score", None)
        score_str = f" (score={score:.3f})" if score is not None else ""
        passages.append(f"[Passage {i}{score_str}]\n{node.get_content().strip()}")

    return "\n\n".join(passages)


# Legacy alias kept for backward compatibility
def get_query_engine(crop_filter: str = None):
    """Deprecated: use retrieve_chunks() directly."""
    log.warning("get_query_engine() is deprecated. Use retrieve_chunks().")
    index = _get_index()
    filters = None
    if crop_filter:
        filters = MetadataFilters(filters=[MetadataFilter(key="crop", value=crop_filter)])
    return index.as_query_engine(similarity_top_k=3, filters=filters)


if __name__ == "__main__":
    result = retrieve_chunks("Treatment Early blight", crop_filter="Tomato")
    log.info(result)