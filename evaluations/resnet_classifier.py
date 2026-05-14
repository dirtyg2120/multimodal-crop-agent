import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from app.vision.clip_labels import RESNET_TO_CLIP


def resnet_to_clip_label(resnet_label: str) -> str:
    return RESNET_TO_CLIP.get(resnet_label, f"[UNMAPPED] {resnet_label}")


class ResNetClassifier:
    def __init__(self, model_name: str = "A2H0H0R1/resnet-50-plant-disease"):
        print(f"Loading ResNet model: {model_name}")
        self.model = ResNetForImageClassification.from_pretrained(model_name).to(DEVICE)
        # Specify size explicitly to avoid ConvNeXt processor config mismatch
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            size={"shortest_edge": 224}
        )
        self.known_labels = None

    def set_known_labels(self, labels: List[str]):
        self.known_labels = labels

    def predict(self, image: Image.Image, threshold: float = 0.35) -> Tuple[str, float, bool]:
        inputs = self.feature_extractor(images=np.array(image), return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=1)

        top_prob, top_idx = probs[0].max(dim=0)
        top_prob = top_prob.item()
        top_idx = top_idx.item()

        resnet_label = self.model.config.id2label[top_idx]
        clip_label = resnet_to_clip_label(resnet_label)

        # ResNet rarely rejects unknowns - it always picks from its 38 classes
        if top_prob < threshold:
            return "Unknown", top_prob, True
        
        return clip_label, top_prob, False
