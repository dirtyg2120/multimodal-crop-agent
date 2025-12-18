from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

@dataclass
class DetectedObject:
    label: str          
    confidence: float   
    box: List[int]      
    crop_id: int        # To link back to the specific image crop in the UI

@dataclass
class AgronomyDeps:
    user_id: str
    total_leaves: int
    healthy_count: int
    disease_counts: Dict[str, int]
    pest_counts: Dict[str, int]
    detailed_detections: List[DetectedObject]

    def to_dict(self):
        return asdict(self)