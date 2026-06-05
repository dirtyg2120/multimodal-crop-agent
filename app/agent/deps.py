from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class DetectedObject:
    label: str          
    confidence: float   
    box: List[int]      
    crop_id: int        # To link back to the specific image crop in the UI

@dataclass
class AgronomyDeps:
    crop_name: str
    total_leaves: int
    healthy_count: int
    disease_counts: Dict[str, int]
    pest_counts: Dict[str, int]
    detailed_detections: Optional[List[DetectedObject]]
    user_id: str = "default_user"

    # Component toggles (used by exp_component_analysis.py)
    enable_rag: bool = True
    infection_ratio: float = 0.0

    def __post_init__(self):
        denom = self.healthy_count + sum(self.disease_counts.values())
        self.infection_ratio = float(sum(self.disease_counts.values()) / denom) if denom > 0 else 0.0

    def to_dict(self):
        return asdict(self)