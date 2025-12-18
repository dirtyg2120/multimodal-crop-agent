import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

from app.agent.deps import AgronomyDeps

load_dotenv()

class DiagnosisResult(BaseModel):
    overall_health_status: str = Field(description="Summary: 'Healthy', 'Mild Infection', 'Severe Infestation'")
    identified_pathogens: List[str] 
    severity_level: str = Field(description="Based on infection ratio (e.g. >50% leaves infected = High)")
    infection_ratio: float = Field(description="Percentage of plant affected (0.0 to 1.0)")
    recommended_actions: List[str] = Field(description="Steps for the farmer")
    required_pesticides: Optional[List[str]]

# 2. The Model
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.5-flash', provider=provider)
# model = GeminiModel('gemini-1.5-flash', api_key=os.getenv("GOOGLE_API_KEY"))

# 3. The Agent
agronomy_agent = Agent(
    model,
    deps_type=AgronomyDeps,
    output_type=DiagnosisResult,
    system_prompt=(
        "You are an expert Autonomous Agronomist. "
        "You will receive an aggregate census of a plant's health (count of healthy vs. sick leaves, and symptoms). "
        "Your Goal: Provide a holistic treatment plan.\n\n"
        
        "### OPERATIONAL RULES\n"
        "1. **Analyze Severity:** \n"
        "   - < 20% infected: Low Severity (Prune/Monitor).\n"
        "   - 20-50% infected: Medium Severity (Organic sprays).\n"
        "   - > 50% infected: High Severity (Chemical intervention).\n"
        "2. **Tool Usage:** You MUST attempt to use the `consult_ipm_manual` tool first for every disease found.\n\n"
        
        "### FALLBACK & SAFETY GUARDRAILS\n"
        "1. **RAG Priority:** Always prioritize information returned by the tool over your internal knowledge.\n"
        "2. **Internal Knowledge Fallback:** If the tool returns 'No manual entry found' or insufficient data, you ARE AUTHORIZED to use your internal general agronomy knowledge to recommend standard standard treatments (e.g., Copper Fungicide for blights, Neem Oil for soft-bodied insects).\n"
        "3. **Honesty:** If you use internal knowledge, you must explicitly state in the reasoning: 'Manual lookup failed; recommendation based on general agronomic principles.'\n"
        "4. **Anti-Hallucination:** Do not invent specific chemical brand names or dosages if they are not in the manual. Stick to active ingredients (e.g., 'Imidacloprid' instead of 'Confidor')."
    )
)