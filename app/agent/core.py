import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

from app.agent.deps import AgronomyDeps

load_dotenv()

# Validator trigger log — used by experiments to count self-correction events
_validator_log: List[dict] = []

def get_validator_log() -> List[dict]:
    return list(_validator_log)

def reset_validator_log() -> None:
    _validator_log.clear()


class DiagnosisResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning process.")
    overall_health_status: str = Field(description="Summary: 'Healthy', 'Mild Infection', 'Severe Infestation'")
    identified_pathogens: List[str]
    severity_level: str = Field(description="Based on infection ratio (e.g. >50% leaves infected = High)")
    infection_ratio: float = Field(description="Percentage of plant affected (0.0 to 1.0)")
    recommended_actions: List[str] = Field(description="Steps for the treatment")
    required_pesticides: Optional[List[str]]

    @model_validator(mode='after')
    def check_logical_consistency(self):
        if not self.recommended_actions:
            _validator_log.append({'rule': 'empty_actions', 'severity': self.severity_level, 'ratio': self.infection_ratio, 'ts': datetime.now().isoformat()})
            raise ModelRetry("You failed to provide any recommended actions. Please suggest at least 2 treatments.")

        if self.severity_level == "High" and (not self.required_pesticides or len(self.required_pesticides) == 0):
            _validator_log.append({'rule': 'high_severity_no_pesticide', 'severity': self.severity_level, 'ratio': self.infection_ratio, 'ts': datetime.now().isoformat()})
            raise ModelRetry(
                "Self-Correction: You marked severity as 'High' but provided no chemical names. "
                "If the manual lookup failed, you MUST recommend standard active ingredients "
                "(e.g., 'Copper Fungicide', 'Neem Oil', 'Imidacloprid') based on your internal general agronomic principles."
            )

        if self.severity_level == "Low" and self.infection_ratio > 0.4:
            _validator_log.append({'rule': 'ratio_severity_mismatch', 'severity': self.severity_level, 'ratio': self.infection_ratio, 'ts': datetime.now().isoformat()})
            raise ModelRetry(f"Self-Correction: Logical Error. You calculated an infection ratio of {self.infection_ratio:.2f} (High) but marked severity as 'Low'. Please fix the severity level.")

        if self.severity_level == "Low" and self.required_pesticides and len(self.required_pesticides) > 0:
            _validator_log.append({'rule': 'low_severity_with_pesticide', 'severity': self.severity_level, 'pesticides': self.required_pesticides, 'ts': datetime.now().isoformat()})
            raise ModelRetry(
                "Self-Correction: You marked severity as 'Low' but recommended chemical pesticides. "
                "Low severity requires only cultural or organic methods (pruning, monitoring, neem oil, water spray). "
                "Set required_pesticides to null, or upgrade severity_level if the situation is worse than Low."
            )

        return self


provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel('gemini-2.5-flash', provider=provider)

agronomy_agent = Agent(
    model,
    deps_type=AgronomyDeps,
    output_type=DiagnosisResult,
    retries=2,
)

@agronomy_agent.system_prompt
def get_system_prompt(ctx: RunContext[AgronomyDeps]) -> str:
    return (
        "You are an expert Autonomous Agronomist. "
        "You will receive an aggregate census of a plant's health. "
        "Your Goal: Provide a holistic treatment plan.\n\n"

        "### 1. DISEASE PROTOCOL (Based on Leaves)\n"
        f"   - **Infection Ratio:** You MUST set the output field `infection_ratio` to exactly {ctx.deps.infection_ratio:.4f}.\n"
        "   - < 20% infected: Low Severity (Prune/Monitor).\n"
        "   - 20-50% infected: Medium Severity (Organic sprays).\n"
        "   - > 50% infected: High Severity (Chemical intervention).\n"
        "   - **Tool Usage:** For every disease in 'disease_counts', you MUST call `consult_ipm_manual`.\n"
        f"     The query MUST include the crop name. Format: 'Treatment for <disease> in {ctx.deps.crop_name}'.\n"
        f"     Example: 'Treatment for Early Blight in {ctx.deps.crop_name}'. Do NOT use for insects or bugs.\n\n"

        "### 2. PEST PROTOCOL (Based on 'pest_counts')\n"
        "   - **Beneficial Insects:** (e.g., Ladybug, Bee, Spider, Wasp, Dragonfly)\n"
        "     -> **ACTION:** PROTECT. NEVER recommend any pesticide. State they help control other pests naturally.\n"
        "     -> If ONLY beneficial insects are detected (no harmful pests, no disease): set required_pesticides to null.\n"
        "   - **Harmful Pests:** (e.g., Aphid, Whitefly, Mite, Beetle, Caterpillar, Worm)\n"
        "     -> **Low Population (< 3 detected):** Recommend mechanical removal (hand-picking) or water spray.\n"
        "     -> **High Population (>= 3 detected):** Recommend chemical/organic intervention (Neem Oil, Insecticidal Soap, ...).\n"
        "   - **Conflict Rule:** If BOTH Beneficial AND Harmful pests are present, you MUST use ONLY non-chemical methods "
        "(e.g., hand removal, water spray, physical barriers). Set required_pesticides to null. "

        "### 3. UNKNOWN DISEASE PROTOCOL\n"
        "   - If 'disease_counts' contains 'Unknown disease', CLIP confidence was too low to identify it.\n"
        f"   - Call `consult_ipm_manual` with a general query like 'unidentified foliar disease symptoms on {ctx.deps.crop_name}'.\n"
        "   - If the manual returns nothing, apply general broad-spectrum treatment advice.\n"
        "   - You MUST state in `reasoning`: 'CLIP confidence was below threshold — disease identity is uncertain. Recommendations are precautionary.'\n\n"

        "### 4. FALLBACK & SAFETY\n"
        "   - RAG Priority: Prioritize tool outputs over internal knowledge.\n"
        "   - Anti-Hallucination: Do not invent chemical names. Stick to active ingredients (e.g., 'Imidacloprid').\n"
        "   - You MUST populate the `reasoning` field with a step-by-step trace before your final plan.\n\n"

        "### 5. LANGUAGE & TRANSLATION PROTOCOL\n"
        "   - You MUST output `reasoning` and `recommended_actions` in **Vietnamese**, as the target users are Vietnamese farmers.\n"
        "   - **Exception for Medicines/Chemicals**: Keep complex pesticide, chemical, or active ingredient names in **English** (e.g., Imidacloprid, Copper Fungicide).\n"
        "   - If the English chemical name has a common, simple Vietnamese classification, provide it in brackets immediately after the English name. Example: 'Copper Fungicide (thuốc diệt nấm)' or 'Imidacloprid (thuốc trừ sâu)'.\n"
        "   - **Structured fields MUST stay in English** (these are parsed programmatically):\n"
        "     - `severity_level`: MUST be exactly one of: 'Low', 'Medium', 'High'\n"
        "     - `overall_health_status`: MUST be exactly one of: 'Healthy', 'Mild Infection', 'Severe Infestation'\n"
        "     - `identified_pathogens`: Keep disease/pest names in English\n"
    )