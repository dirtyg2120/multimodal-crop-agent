import os
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

from app.agent.deps import AgronomyDeps

load_dotenv()

class DiagnosisResult(BaseModel):
    # reasoning: str = Field(description="Step-by-step logic used to derive the diagnosis. Explain your analysis of the pest/disease counts here first.")
    reasoning: str = Field(description="Step-by-step reasoning process.")
    overall_health_status: str = Field(description="Summary: 'Healthy', 'Mild Infection', 'Severe Infestation'")
    identified_pathogens: List[str] 
    severity_level: str = Field(description="Based on infection ratio (e.g. >50% leaves infected = High)")
    infection_ratio: float = Field(description="Percentage of plant affected (0.0 to 1.0)")
    recommended_actions: List[str] = Field(description="Steps for the treatment")
    required_pesticides: Optional[List[str]]

    @model_validator(mode='after')
    def check_logical_consistency(self):
        """
        This runs AFTER the LLM generates JSON. 
        If it fails, the error message is sent back to the LLM to 'Try Again'.
        """
        # Rule 1: Don't starve the user
        if not self.recommended_actions:
            raise ModelRetry("You failed to provide any recommended actions. Please suggest at least 2 treatments.")

        # Rule 2: High Severity requires strong action
        if self.severity_level == "High" and (not self.required_pesticides or len(self.required_pesticides) == 0):
            raise ModelRetry("Self-Correction: You marked severity as 'High' but provided no chemical/pesticide names. For high severity, you MUST list specific active ingredients found in the manual.")

        # Rule 3: Math check (Hallucination guard)
        if self.severity_level == "Low" and self.infection_ratio > 0.4:
            raise ModelRetry(f"Self-Correction: Logical Error. You calculated an infection ratio of {self.infection_ratio:.2f} (High) but marked severity as 'Low'. Please fix the severity level.")

        return self

# 2. The Model
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.5-flash', provider=provider)
# model = GeminiModel('gemini-1.5-flash', api_key=os.getenv("GOOGLE_API_KEY"))

# 3. The Agent
agronomy_agent = Agent(
    model,
    deps_type=AgronomyDeps,
    output_type=DiagnosisResult,
    retries=1,
    system_prompt=(
        "You are an expert Autonomous Agronomist. "
        "You will receive an aggregate census of a plant's health. "
        "Your Goal: Provide a holistic treatment plan.\n\n"
        
        "### 1. DISEASE PROTOCOL (Based on Leaves)\n"
        "   - < 20% infected: Low Severity (Prune/Monitor).\n"
        "   - 20-50% infected: Medium Severity (Organic sprays).\n"
        "   - > 50% infected: High Severity (Chemical intervention).\n"
        "   - **Tool Usage:** You MUST use `consult_ipm_manual` for every disease found (leaf only).\n\n"

        "### 2. PEST PROTOCOL (Based on 'pest_counts')\n"
        "   - **Beneficial Insects:** (e.g., Ladybug, Bee, Spider, Wasp, Dragonfly)\n"
        "     -> **ACTION:** PROTECT. Do NOT recommend pesticides. State that they help control other pests.\n"
        "   - **Harmful Pests:** (e.g., Aphid, Whitefly, Mite, Beetle, Caterpillar, Worm)\n"
        "     -> **Low Population (< 3 detected):** Recommend mechanical removal (hand-picking) or water spray or something else.\n"
        "     -> **High Population (>= 3 detected):** Recommend chemical/organic intervention (Neem Oil, Insecticidal Soap, ...).\n"
        "   - **Conflict Rule:** If BOTH Beneficial and Harmful pests are present, prioritize NON-CHEMICAL methods to avoid killing the 'good guys'.\n\n"
        
        "### 3. FALLBACK & SAFETY\n"
        "   - **RAG Priority:** Prioritize tool outputs over internal knowledge.\n"
        "   - **Anti-Hallucination:** Do not invent chemical names. Stick to active ingredients (e.g., 'Imidacloprid')."

        # "CRITICAL VALIDATION RULES (Do not violate these or you will be rejected):\n"
        # "- If `detected_beneficials` is True, `is_chemical_recommended` MUST be False.\n"
        # "- You MUST provide a `reasoning_trace` before your final plan."
    )
)