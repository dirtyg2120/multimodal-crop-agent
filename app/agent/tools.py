# app/agent/tools.py

from pydantic_ai import RunContext
from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent 

# --- CONNECTING THE ENGINE ---
# We import the RAG engine we built earlier
from app.rag.engine import get_query_engine

# 1. Initialize the Engine Globaly
# We do this outside the function so we load the database ONCE, not every time we query.
# This makes the agent much faster.
# print("   ⚙️  [System] Loading RAG Engine...")
# _query_engine = get_query_engine()

# @agronomy_agent.tool
# async def consult_ipm_manual(ctx: RunContext[AgronomyDeps], query: str) -> str:
#     """
#     Use this tool to look up treatment plans in the Vector Database (Chroma).
#     Pass a query like 'treatment for Early Blight in Tomato'.
#     """
#     # 2. Context Enhancement (The "Pre-Prompt" logic)
#     # The agent might just ask "how to treat it?", which is bad for search.
#     # We force the crop and disease name into the query to ensure accuracy.
#     enhanced_query = f"{query} for {ctx.deps.disease_label} in {ctx.deps.crop_detected}"
    
#     print(f"   🔎 [RAG Tool] Searching Manuals for: '{enhanced_query}'")
    
#     # 3. REAL RETRIEVAL
#     # This calls ChromaDB via LlamaIndex
#     response = _query_engine.query(enhanced_query)
    
#     # Check if we got a good answer
#     response_text = str(response)
#     if not response_text or "Empty Response" in response_text:
#         return "No specific manual entry found. Recommend general hygiene and monitoring."
    
#     # 4. Return the text chunk to Gemini
#     # We truncate it slightly to prevent overflowing the context window (optional)
#     return response_text[:2000]

@agronomy_agent.tool
async def consult_ipm_manual(ctx: RunContext[AgronomyDeps], query: str) -> str:
    print(f"   >> [RAG Tool] Searching manuals for: {query}...")
    
    # (Paste your mock logic or real LlamaIndex logic here)
    return "Mock Treatment Plan for " + ctx.deps.disease_label