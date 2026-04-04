import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
import pickle
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Keetawan/clip-vit-large-patch14-plant-disease-finetuned"
DEFAULT_THRESHOLD = 0.82
PROTOTYPES_PATH = "./data/few_shot_prototypes.pkl"


class FewShotEngine:
    """Centroid-based few-shot classifier using frozen CLIP features."""

    def __init__(self):
        log.info(f"🔬 Loading Few-Shot CLIP on {DEVICE}...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.prototypes: dict[str, np.ndarray] = {}
        self._load()
        log.info(f"✅ Few-Shot Engine ready. ({len(self.prototypes)} prototypes loaded)")

    def _embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
        feat = raw.pooler_output if hasattr(raw, "pooler_output") else raw
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()

    def register(self, label: str, support_images: list) -> None:
        """Build centroid from PIL images and save to disk."""
        embeddings = np.array([self._embed(img) for img in support_images])
        centroid = embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        self.prototypes[label] = centroid
        self._save()
        log.info(f"✅ Registered '{label}' ({len(support_images)} images) → saved.")

    def identify(self, query_image: Image.Image, threshold: float = DEFAULT_THRESHOLD):
        """Returns (best_label, score, is_matched)."""
        if not self.prototypes:
            return None, 0.0, False
        query_feat = self._embed(query_image).reshape(1, -1)
        best_label, best_score = None, -1.0
        for label, centroid in self.prototypes.items():
            score = float(cosine_similarity(query_feat, centroid)[0][0])
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score, best_score >= threshold

    def _save(self):
        os.makedirs(os.path.dirname(PROTOTYPES_PATH), exist_ok=True)
        with open(PROTOTYPES_PATH, "wb") as f:
            pickle.dump(self.prototypes, f)
            print("save done!")

    def _load(self):
        if os.path.exists(PROTOTYPES_PATH):
            with open(PROTOTYPES_PATH, "rb") as f:
                self.prototypes = pickle.load(f)
