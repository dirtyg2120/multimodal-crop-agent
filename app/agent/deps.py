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
    user_id: str
    crop_name: str
    total_leaves: int
    healthy_count: int
    disease_counts: Dict[str, int]
    pest_counts: Dict[str, int]
    detailed_detections: Optional[List[DetectedObject]]

    # Component toggles (used by exp_component_analysis.py)
    enable_rag: bool = True
    enable_validators: bool = True

    def to_dict(self):
        return asdict(self)