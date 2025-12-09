import asyncio
import os
from dotenv import load_dotenv

# Import the core components we split up
from app.agent.core import agronomy_agent
from app.agent.deps import AgronomyDeps
from app.vision.clip_labels import CLIP_LABEL_MAP

# CRITICAL: Import tools so they register with the agent!
import app.agent.tools 

# Load env for API keys (Gemini)
load_dotenv()

async def run_pipeline(clip_class_id: int, confidence: float):
    print("--- 🚜 Starting Agronomy Agent Pipeline 🚜 ---")
    
    # 1. VISUAL PERCEPTION LAYER (Simulated Raw Input)
    # The camera gives us an ID (e.g., 33), not a string.
    if clip_class_id not in CLIP_LABEL_MAP:
        raise ValueError(f"Unknown CLIP Class ID: {clip_class_id}")
    
    full_label = CLIP_LABEL_MAP[clip_class_id]
    
    # Logic to parse "Tomato leaf with Early blight"
    parts = full_label.split(" with ")
    if len(parts) == 2:
        crop_name = parts[0].replace(" leaf", "")
        disease_name = parts[1]
    else:
        # Handle "Healthy Tomato leaf" case
        crop_name = full_label.replace("Healthy ", "").replace(" leaf", "")
        disease_name = "Healthy"

    print(f"   👁️  [Vision] Detected ID {clip_class_id} -> '{full_label}'")
    print(f"   📊 [Vision] Confidence: {confidence:.4f}")

    # 2. CONTEXT BUILDING (Dependency Injection)
    # We package this data safely for the Agent
    deps = AgronomyDeps(
        user_id="farmer_01",
        crop_detected=crop_name,
        disease_label=disease_name,
        clip_confidence=confidence
    )

    # 3. AGENT REASONING (The "Brain")
    # We construct a prompt that forces the agent to look at the data
    print(full_label)
    user_prompt = (
        f"The vision system detected {full_label} with {confidence*100:.1f}% confidence. "
        "Verify the severity and recommend a treatment plan."
    )

    print(f"   🧠 [Agent] Reasoning with Gemini...")
    result = await agronomy_agent.run(user_prompt, deps=deps)
    # print(result)

    # 4. ACTUATION (IoT Output)
    print("\n--- 📝 FINAL JSON OUTPUT (Send to IoT Gateway) ---")
    print(result.output.model_dump_json(indent=2))
    return result.output

if __name__ == "__main__":
    # Test Case: Class 33 is 'Tomato leaf with Early blight'
    asyncio.run(run_pipeline(clip_class_id=34, confidence=0.88))