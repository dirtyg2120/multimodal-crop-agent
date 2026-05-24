import os
import sys
from typing import List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.35
from app.vision.clip_labels import DISEASE_LABELS

class CLIPClassifier:
    """CLIP-based classifier with unknown rejection"""

    def __init__(self, model_name: str = "Keetawan/clip-vit-large-patch14-plant-disease-finetuned"):
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.known_labels = None

    def set_known_labels(self, labels: List[str]):
        self.known_labels = DISEASE_LABELS

    def predict(self, image: Image.Image, threshold: float = CONFIDENCE_THRESHOLD) -> Tuple[str, float, bool]:
        """
        Returns:
            label: Predicted label or "Unknown"
            confidence: Prediction confidence
            is_unknown: True if rejected as unknown
        """
        if self.known_labels is None:
            raise ValueError("Known labels not set. Call set_known_labels() first.")

        inputs = self.processor(
            text=self.known_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        top_prob, top_idx = probs[0].max(dim=0)
        top_prob = top_prob.item()
        top_idx = top_idx.item()

        if top_prob < threshold:
            return "Unknown", top_prob, True
        else:
            return self.known_labels[top_idx], top_prob, False
