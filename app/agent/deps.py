from dataclasses import dataclass

@dataclass
class AgronomyDeps:
    user_id: str
    crop_detected: str
    disease_label: str
    clip_confidence: float