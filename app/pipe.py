import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_CONFIDENCE_THRESHOLD = 0.5  # below this → "Unknown" (open-set rejection)


class VisionSystem:
    def __init__(self):
        log.info("🚜 Loading Vision Models...")
        self.plant_model = CLIPModel.from_pretrained("Keetawan/clip-vit-large-patch14-plant-disease-finetuned").to(DEVICE)
        self.plant_processor = CLIPProcessor.from_pretrained("Keetawan/clip-vit-large-patch14-plant-disease-finetuned")
        self.insect_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
        self.insect_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        log.info("✅ Models Loaded.")

    def classify_leaf(self, image_crop: np.ndarray):
        return self._run_clip(image_crop, list(CLIP_LABEL_MAP.values()), self.plant_model, self.plant_processor)

    def classify_pest(self, image_crop: np.ndarray):
        return self._run_clip(image_crop, INSECT_LABELS, self.insect_model, self.insect_processor)

    def _run_clip(self, image_crop, labels, model, processor):
        pil_image = Image.fromarray(image_crop)
        inputs = processor(text=labels, images=pil_image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)
        top_prob, top_idx = probs[0].max(dim=0)
        top_prob_val = top_prob.item()
        if top_prob_val < CLIP_CONFIDENCE_THRESHOLD:
            return "Unknown", top_prob_val
        return labels[top_idx.item()], top_prob_val


async def analyze_full_plant(crops_data: list, vision_system: VisionSystem, few_shot_engine=None):
    """
    few_shot_engine: optional FewShotEngine — if provided, Unknown leaves are
    re-checked against registered prototypes before being flagged as Unknown.
    """
    log.info("🚜 Starting Agronomy Agent Pipeline")

    detected_objects = []
    disease_tally = Counter()
    pest_tally = Counter()
    healthy_count = 0
    unknown_count = 0  # tracked separately so agent can compute ratio correctly

    for i, item in enumerate(crops_data):
        label_lower = item['label'].lower()
        if "leaf" in label_lower:
            full_label, conf = vision_system.classify_leaf(item['crop'])

            # --- Few-shot fallback for Unknown leaves ---
            if full_label == "Unknown" and few_shot_engine and few_shot_engine.prototypes:
                pil = Image.fromarray(item['crop'])
                fs_label, fs_score, matched = few_shot_engine.identify(pil)
                if matched:
                    full_label = fs_label
                    conf = fs_score
                    log.info(f"🧬 Leaf {i}: few-shot matched '{full_label}' ({conf:.2f})")

            if full_label == "Unknown":
                unknown_count += 1
                log.info(f"⚠️  Leaf {i}: low confidence ({conf:.2f}) → flagged as Unknown")
            elif "Healthy" in full_label:
                healthy_count += 1
            else:
                disease_name = full_label.split(" with ")[-1] if " with " in full_label else full_label
                disease_tally[disease_name] += 1

            detected_objects.append(DetectedObject(label=full_label, confidence=conf, box=item['bbox'], crop_id=i))
        else:
            pest_name, conf = vision_system.classify_pest(item['crop'])
            pest_tally[pest_name] += 1
            detected_objects.append(DetectedObject(label=pest_name, confidence=conf, box=item['bbox'], crop_id=i))

    # Determine dominant crop from first identified leaf
    dominant_crop = "General"
    for obj in detected_objects:
        if " leaf " in obj.label:
            dominant_crop = obj.label.split(" leaf ")[0]
            break
        elif "Healthy" in obj.label:
            dominant_crop = obj.label.replace("Healthy ", "").replace(" leaf", "")
            break

    # Total leaves only (exclude pests) for accurate infection ratio
    total_leaves = healthy_count + sum(disease_tally.values()) + unknown_count

    deps = AgronomyDeps(
        user_id="user",
        crop_name=dominant_crop,
        total_leaves=len(detected_objects),
        healthy_count=healthy_count,
        disease_counts=dict(disease_tally),
        pest_counts=dict(pest_tally),
        detailed_detections=None
    )

    # Bug fix: pass unknown_count separately so agent computes infection_ratio
    # from confirmed leaves only — not inflated by unidentified ones.
    summary_text = (
        f"Analysis Report:\n"
        f"- Total leaf objects detected: {total_leaves}\n"
        f"- Healthy leaves: {healthy_count}\n"
        f"- Diseased leaves (confirmed): {dict(disease_tally)}\n"
        f"- Unknown condition leaves (CLIP uncertain, identity unknown): {unknown_count}\n"
        f"- Pests detected: {dict(pest_tally)}\n"
        f"\nIMPORTANT: Compute infection_ratio as confirmed_diseased_leaves / "
        f"(healthy + confirmed_diseased). Do NOT count Unknown leaves as diseased.\n"
    )

    user_prompt = (
        f"Here is the aggregate data for a crop image: \n{summary_text}\n"
        "TASKS:\n"
        "1. **Pest Analysis:** Categorize detected pests as 'Beneficial' or 'Harmful'. Apply the Pest Protocol rules.\n"
        "2. **Disease Analysis:** Assess severity based on infection ratio.\n"
        "3. **Plan:** Provide an integrated plan. If mixed infections (pests + disease) exists, prioritize the most severe threat but protect beneficial insects."
    )

    log.info("🧠 [Agent] Reasoning...")

    try:
        testing = False
        if not testing:
            result = await agronomy_agent.run(user_prompt, deps=deps)
            output = result.output
            log.info(output.model_dump_json(indent=4))
        else:
            output = sample.output
    except UnexpectedModelBehavior as e:
        log.info(f"DEBUG INFO: {e}")
    else:
        return {"detections": detected_objects, "stats": deps, "agent_response": output}