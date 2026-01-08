import torch
import numpy as np
from PIL import Image
from collections import Counter
from transformers import CLIPModel, CLIPProcessor
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.agent.deps import AgronomyDeps, DetectedObject
from app.agent.core import agronomy_agent
from app.vision.clip_labels import CLIP_LABEL_MAP, INSECT_LABELS
from app import sample
import app.agent.tools


# load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. DEFINE THE CLASS (Do not instantiate it here) ---
class VisionSystem:
    def __init__(self):
        print("   🚜 Loading Vision Models...")
        # Plant Disease Model
        self.plant_model = CLIPModel.from_pretrained("Keetawan/clip-vit-large-patch14-plant-disease-finetuned").to(DEVICE)
        self.plant_processor = CLIPProcessor.from_pretrained("Keetawan/clip-vit-large-patch14-plant-disease-finetuned")
        
        # Insect Model (BioCLIP)
        # self.insect_model = CLIPModel.from_pretrained("imageomics/bioclip").to(DEVICE)
        # self.insect_processor = CLIPProcessor.from_pretrained("imageomics/bioclip")
        self.insect_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
        self.insect_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        print("   ✅ Models Loaded.")

    def classify_leaf(self, image_crop: np.ndarray):
        return self._run_clip(
            image_crop, list(CLIP_LABEL_MAP.values()), self.plant_model, self.plant_processor
        )

    def classify_pest(self, image_crop: np.ndarray):
        return self._run_clip(
            image_crop, INSECT_LABELS, self.insect_model, self.insect_processor
        )

    def _run_clip(self, image_crop, labels, model, processor):
        pil_image = Image.fromarray(image_crop)
        inputs = processor(text=labels, images=pil_image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        top_prob, top_idx = probs[0].max(dim=0)
        return labels[top_idx.item()], top_prob.item()


# async def run_pipeline(clip_class_id: int, confidence: float):
async def analyze_full_plant(crops_data: list, vision_system: VisionSystem):
    print("--- 🚜 Starting Agronomy Agent Pipeline 🚜 ---")
    
    detected_objects = []
    disease_tally = Counter()
    pest_tally = Counter()
    healthy_count = 0
    
    # CLIP Loop
    for i, item in enumerate(crops_data):
        label_lower = item['label'].lower()

        if "leaf" in label_lower:
            full_label, conf = vision_system.classify_leaf(item['crop'])
            if "Healthy" in full_label:
                healthy_count += 1
            else:
                disease_name = full_label.split(" with ")[-1] if " with " in full_label else full_label
                disease_tally[disease_name] += 1
            
            detected_objects.append(DetectedObject(
                label=full_label, confidence=conf, box=item['bbox'], crop_id=i
            ))
        else:
            pest_name, conf = vision_system.classify_pest(item['crop'])
            pest_tally[pest_name] += 1
            
            detected_objects.append(DetectedObject(
                label=f"Pest: {pest_name}", confidence=conf, box=item['bbox'], crop_id=i
            ))

    # DINO Results (Pests)
    # e.g., dino_results = {"leaf": 7, "beetle": 2}
    # pest_tally = {k: v for k, v in dino_results.items() if k != 'leaf'}

    dominant_crop = "General"
    if detected_objects:
        # "Tomato leaf with..." -> "Tomato"
        # "Healthy Tomato leaf" -> "Tomato"
        for obj in detected_objects:
            label = obj.label
            if " leaf " in label:
                dominant_crop = label.split(" leaf ")[0]
                break
            elif "Healthy" in label:
                dominant_crop = label.replace("Healthy ", "").replace(" leaf", "")
                break
            else:
                continue

    # 2. CONTEXT BUILDING (Dependency Injection)
    deps = AgronomyDeps(
        user_id="user",
        crop_name=dominant_crop,
        total_leaves=len(detected_objects),
        healthy_count=healthy_count,
        disease_counts=dict(disease_tally),
        pest_counts=dict(pest_tally),
        detailed_detections=None
    )

    # 3. AGENT REASONING (Construct Whole-Plant Prompt)
    # We summarize the situation for the LLM so it doesn't have to do math.
    summary_text = (
        f"Analysis Report:\n"
        f"- Total detected objects: {deps.total_leaves}\n"
        f"- Healthy: {deps.healthy_count}\n"
        f"- Diseases: {deps.disease_counts}\n"
        f"- Pests: {deps.pest_counts}\n"
    )

    user_prompt = (
        f"Here is the aggregate data for a crop image: \n{summary_text}\n"
        "TASKS:\n"
        "1. **Pest Analysis:** Categorize detected pests as 'Beneficial' or 'Harmful'. Apply the Pest Protocol rules.\n"
        "2. **Disease Analysis:** Assess severity based on infection ratio.\n"
        "3. **Plan:** Provide an integrated plan. If mixed infections (pests + disease) exists, prioritize the most severe threat but protect beneficial insects."
    )

    print(f"   🧠 [Agent] Reasoning ...")

    try:
        testing = False
        if not testing:
            result = await agronomy_agent.run(user_prompt, deps=deps)
            output = result.output
            print(output.model_dump_json(indent=4))
        else:
            output = sample.output
    except UnexpectedModelBehavior as e:
        print(f"DEBUG INFO: {e}")
    else:
        return {
            "detections": detected_objects,
            "stats": deps,
            "agent_response": output
        }

if __name__ == "__main__":
    print("nothing")