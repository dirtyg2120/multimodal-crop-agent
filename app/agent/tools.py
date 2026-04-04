import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
import asyncio
import nest_asyncio
from pydantic_ai import RunContext
from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent
from app.rag.engine import get_query_engine

try:
    import uvloop
    if isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except ImportError:
    pass

nest_asyncio.apply()


@agronomy_agent.tool
async def consult_ipm_manual(ctx: RunContext[AgronomyDeps], query: str) -> str:
    """Look up treatment plans in the ChromaDB vector database."""
    if not ctx.deps.enable_rag:
        return (
            "MANUAL_LOOKUP_DISABLED: RAG is off for this run. "
            "Use internal knowledge only. Do NOT reference any manual."
        )

    engine = get_query_engine(crop_filter=ctx.deps.crop_name)
    if engine is None:
        return (
            "MANUAL_LOOKUP_FAILED: Retrieval engine is offline. "
            "Use internal knowledge and state the manual was inaccessible."
        )

    enhanced_query = f"{query} in {ctx.deps.crop_name}"
    log.info(f"🔎 [RAG] Querying: '{enhanced_query}'")

    try:
        response = engine.query(enhanced_query)
        response_text = str(response).strip()

        if not response_text or "Empty Response" in response_text:
            log.info("⚠️ [RAG] No results found.")
            return (
                "MANUAL_LOOKUP_FAILED: No relevant entries found in IPM manuals. "
                "Use internal knowledge. You MUST state advice is based on general principles, not the manual."
            )

        return f"Verified Manual Entry:\n{response_text[:2000]}"

    except Exception as e:
        log.info(f"❌ [RAG] Error: {e}")
        return "MANUAL_LOOKUP_FAILED: Retrieval error. Use internal knowledge with caution."