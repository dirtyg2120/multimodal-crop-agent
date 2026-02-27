import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import random

def load_plantvillage_dataset(
    root_dir: str = "data/plantvillage",
    known_classes: List[str] = None,
    unknown_classes: List[str] = None,
    samples_per_class: int = 50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load PlantVillage dataset split into known and unknown
    
    Returns:
        (known_samples, unknown_samples)
    """
    if known_classes is None:
        known_classes = [
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Bacterial_spot",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___healthy"
        ]
    
    if unknown_classes is None:
        unknown_classes = [
            "Tomato___Target_Spot",
            "Tomato___Tomato_mosaic_virus",
            "Pepper___Bacterial_spot",
            "Potato___Early_blight"
        ]
    
    def load_class_images(class_name: str, max_samples: int) -> List[Dict]:
        """Load images from a class folder"""
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Class not found: {class_name}")
            return []
        
        image_files = list(Path(class_dir).glob("*.jpg")) + \\
                     list(Path(class_dir).glob("*.JPG"))
        
        # Randomly sample
        sampled = random.sample(image_files, min(len(image_files), max_samples))
        
        return [{
            'image': Image.open(str(img_path)),
            'label': class_name.replace('___', ' ').replace('_', ' '),
            'is_unknown': False,
            'path': str(img_path)
        } for img_path in sampled]
    
    # Load known samples
    known_samples = []
    for cls in known_classes:
        samples = load_class_images(cls, samples_per_class)
        known_samples.extend(samples)
    
    # Load unknown samples
    unknown_samples = []
    for cls in unknown_classes:
        samples = load_class_images(cls, samples_per_class)
        for sample in samples:
            sample['is_unknown'] = True
        unknown_samples.extend(samples)
    
    print(f"✅ Loaded {len(known_samples)} known samples")
    print(f"✅ Loaded {len(unknown_samples)} unknown samples")
    
    return known_samples, unknown_samples


if __name__ == "__main__":
    known, unknown = load_plantvillage_dataset()
    print(f"Known classes: {len(known)}")
    print(f"Unknown classes: {len(unknown)}")