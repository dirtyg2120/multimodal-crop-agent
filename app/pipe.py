import torch
import numpy as np
from PIL import Image
from collections import Counter

from app.agent.deps import AgronomyDeps, DetectedObject
from app.agent.core import agronomy_agent
from app.vision.clip_labels import CLIP_LABEL_MAP
from app import sample

# load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def classify_crop_clip(image_crop: np.ndarray, model, processor, device="cpu"):
    labels_text = list(CLIP_LABEL_MAP.values())
    pil_image = Image.fromarray(image_crop)
    
    inputs = processor(
        text=labels_text,
        images=pil_image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    
    top_prob, top_idx = probs[0].max(dim=0)
    return top_idx.item(), top_prob.item()


# async def run_pipeline(clip_class_id: int, confidence: float):
async def analyze_full_plant(crops_data: list, clip_model, clip_processor, device="cpu"):
    print("--- 🚜 Starting Agronomy Agent Pipeline 🚜 ---")
    
    detected_objects = []
    disease_tally = Counter()
    healthy_count = 0
    
    # CLIP Loop
    for i, item in enumerate(crops_data):
        class_id, confidence = classify_crop_clip(item['crop'], clip_model, clip_processor)
        full_label = CLIP_LABEL_MAP.get(class_id, "Unknown")
        
        # Add to detailed list
        detected_objects.append(DetectedObject(
            label=full_label,
            confidence=confidence,
            box=item['bbox'],
            crop_id=i
        ))
        
        # Update Statistics
        if "Healthy" in full_label:
            healthy_count += 1
        else:
            disease_name = full_label.split(" with ")[-1] if " with " in full_label else full_label
            disease_tally[disease_name] += 1

    # DINO Results (Pests)
    # e.g., dino_results = {"leaf": 7, "beetle": 2}
    # pest_tally = {k: v for k, v in dino_results.items() if k != 'leaf'}

    # 2. CONTEXT BUILDING (Dependency Injection)
    deps = AgronomyDeps(
        user_id="stream_user",
        total_leaves=len(detected_objects),
        healthy_count=healthy_count,
        disease_counts=disease_tally,
        pest_counts=0,
        detailed_detections=detected_objects
    )

    # 3. AGENT REASONING (Construct Whole-Plant Prompt)
    # We summarize the situation for the LLM so it doesn't have to do math.
    summary_text = (
        f"Analysis Report:\n"
        f"- Total Leaves: {deps.total_leaves}\n"
        f"- Healthy: {deps.healthy_count}\n"
        f"- Diseases: {deps.disease_counts}\n"
        f"- Pests: {deps.pest_counts}\n"
    )
    
    user_prompt = (
        f"Here is the aggregate data for a crop image. \n{summary_text}\n"
        "Assess the overall severity based on the ratio of infected leaves. "
        "If mixed infections (pests + disease) are present, prioritize the most severe threat."
    )

    print(f"   🧠 [Agent] Reasoning ...")

    testing = True
    if not testing:
        result = await agronomy_agent.run(user_prompt, deps=deps)
        output = result.output
        print(output.model_dump_json(indent=4))
    else:
        output = sample.output
    
    return {
        "detections": detected_objects,
        "stats": deps,
        "agent_response": output
    }

if __name__ == "__main__":
    # Test Case: Class 33 is 'Tomato leaf with Early blight'
    # asyncio.run(run_pipeline(clip_class_id=34, confidence=0.88))
    print("nothing")