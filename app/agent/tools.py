import os
from pydantic_ai import RunContext
from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent 
from app.rag.engine import get_query_engine

import asyncio
import nest_asyncio

try:
    import uvloop
    if isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except ImportError:
    pass

nest_asyncio.apply()

@agronomy_agent.tool
async def consult_ipm_manual(ctx: RunContext[AgronomyDeps], query: str) -> str:
    """
    Use this tool to look up treatment plans in the Vector Database (Chroma).
    Pass a query like 'Treatment for Early Blight in <crop_name>'.
    """
    engine = get_query_engine(crop_filter=ctx.deps.crop_name)
    if engine is None:
        return (
            "MANUAL_LOOKUP_FAILED: The retrieval engine is offline. "
            "You are authorized to use your internal knowledge. "
            "Please explicitly state that the manual was inaccessible."
        )

    # Context Enhancement
    enhanced_query = f"{query} in {ctx.deps.crop_name}"
    print(query)
    
    print(f"   🔎 [RAG Tool] Searching Manuals for: '{enhanced_query}'")
    
    try:
        # 3. REAL RETRIEVAL
        response = engine.query(enhanced_query)
        response_text = str(response).strip()
        
        # 4. FALLBACK LOGIC (The Core Update)
        # LlamaIndex returns "Empty Response" if no vectors match closely enough.
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