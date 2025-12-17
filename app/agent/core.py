import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

from app.agent.deps import AgronomyDeps

load_dotenv()

# 1. The Output Schema
class DiagnosisResult(BaseModel):
    is_healthy: bool
    identified_pathogen: str
    confidence_score: float
    severity_level: str = Field(description="Low, Medium, High based on visual signs")
    recommended_actions: List[str] = Field(description="Immediate steps for the farmer")
    required_pesticides: Optional[List[str]] = Field(description="List of chemicals if necessary")

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
        "Act as an agronomist expert. "
        "Use the `consult_ipm_manual` tool to find treatments for the detected disease. "
        "If confidence < 0.5, mark severity as 'Low'. "
        "Return ONLY valid JSON matching the DiagnosisResult schema."
        # "Translate answer to Vietnamese."
    )
)