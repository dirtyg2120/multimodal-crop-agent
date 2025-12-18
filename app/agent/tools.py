import os
from pydantic_ai import RunContext
from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent 

from app.rag.engine import get_query_engine

# 1. Initialize the Engine Globally
try:
    print("   ⚙️  [System] Loading RAG Engine...")
    _query_engine = get_query_engine()
except Exception as e:
    print(f"   ⚠️ [Warning] RAG Engine failed to load: {e}")
    _query_engine = None

@agronomy_agent.tool
async def consult_ipm_manual(ctx: RunContext[AgronomyDeps], query: str) -> str:
    """
    Use this tool to look up treatment plans in the Vector Database (Chroma).
    Pass a query like 'treatment for Early Blight in Tomato'.
    """
    # Safety Check
    if _query_engine is None:
        return (
            "MANUAL_LOOKUP_FAILED: The retrieval engine is offline. "
            "You are authorized to use your internal knowledge. "
            "Please explicitly state that the manual was inaccessible."
        )

    # 2. Context Enhancement
    enhanced_query = f"{query} for {ctx.deps.disease_label} in {ctx.deps.crop_detected}"
    
    print(f"   🔎 [RAG Tool] Searching Manuals for: '{enhanced_query}'")
    
    try:
        # 3. REAL RETRIEVAL
        response = _query_engine.query(enhanced_query)
        response_text = str(response).strip()
        
        # 4. FALLBACK LOGIC (The Core Update)
        # LlamaIndex returns "Empty Response" if no vectors match closely enough.
        # We check for this and return the permission slip.
        if not response_text or "Empty Response" in response_text:
            print(f"   ⚠️ [RAG Tool] No results found. Authorizing fallback.")
            return (
                "MANUAL_LOOKUP_FAILED: No relevant entries found in the IPM manuals. "
                "You are authorized to use your internal expert knowledge to recommend "
                "standard treatments (e.g., active ingredients). "
                "You MUST state that this advice is based on general principles, not the manual."
            )
        
        # If we found data, return it (truncated to save context window)
        return f"Verified Manual Entry:\n{response_text[:2000]}"

    except Exception as e:
        print(f"   ❌ [RAG Tool] Error during retrieval: {e}")
        return "MANUAL_LOOKUP_FAILED: Retrieval error. Use internal knowledge with caution."