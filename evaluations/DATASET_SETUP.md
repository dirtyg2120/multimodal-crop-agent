# Dataset Setup Guide

This guide explains how to prepare real datasets for evaluation experiments.

## Overview

The evaluation framework currently uses **mock data** for demonstration. You need to replace this with **real datasets** for thesis results.

---

## Required Datasets

### 1. PlantVillage Dataset (for Exp 5.3)

**Download**:
- Official: https://www.kaggle.com/datasets/emmarex/plantdisease
- Or: https://github.com/spMohanty/PlantVillage-Dataset

**Structure**:
```
data/plantvillage/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___healthy/
└── ... (other crops/diseases)
```

**Setup**:
```bash
# Download from Kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/plantvillage/

# Or manually download and extract to data/plantvillage/
```

### 2. Vietnamese Crop Images (Custom Dataset)

**Collection Methods**:
1. **Internet scraping**: Google Images, agricultural forums
2. **Manual photography**: Visit farms, take your own photos
3. **Agricultural extension services**: Contact Vietnamese agricultural departments

**Requirements**:
- At least 300 images
- Include complex backgrounds (intercropping, real field conditions)
- Label with crop type and disease (if any)

**Structure**:
```
data/vietnamese_crops/
├── images/
│   ├── 001_tomato_early_blight.jpg
│   ├── 002_durian_leaf_spot.jpg
│   └── ...
└── labels.json  # Annotations
```

**Labels format** (`labels.json`):
```json
{
  "001_tomato_early_blight.jpg": {
    "crop": "Tomato",
    "disease": "Early blight",
    "severity": "High",
    "has_pests": false
  },
  "002_durian_leaf_spot.jpg": {
    "crop": "Durian",
    "disease": "Leaf Spot",
    "severity": "Medium",
    "has_pests": true,
    "pests": ["Aphid"]
  }
}
```

### 3. Unknown Disease Classes (for Open-Set Testing)

You need images of diseases **NOT in PlantVillage** to test unknown rejection.

**Options**:
1. Use crops not in PlantVillage (e.g., Durian, Dragon Fruit)
2. Use rare disease variants
3. Use images from different datasets (iNaturalist, PNAS)

**Example**:
```
data/unknown_diseases/
├── rice_blast/
├── corn_gray_leaf_spot/
├── durian_anthracnose/
└── novel_disease/
```

---

## Dataset Preparation Scripts

### Script 1: Load PlantVillage for Classification

Create `evaluations/data/load_plantvillage.py`:

```python
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
```

### Script 2: Load Vietnamese Dataset

Create `evaluations/data/load_vietnamese.py`:

```python
import json
import os
from PIL import Image
from typing import List, Dict

def load_vietnamese_dataset(
    images_dir: str = "data/vietnamese_crops/images",
    labels_file: str = "data/vietnamese_crops/labels.json"
) -> List[Dict]:
    """
    Load Vietnamese crop dataset
    
    Returns:
        List of samples with images and labels
    """
    if not os.path.exists(labels_file):
        print(f"⚠️  Labels file not found: {labels_file}")
        return []
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    samples = []
    for filename, annotation in labels.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        samples.append({
            'image': Image.open(img_path),
            'crop': annotation.get('crop'),
            'disease': annotation.get('disease'),
            'severity': annotation.get('severity'),
            'has_pests': annotation.get('has_pests', False),
            'pests': annotation.get('pests', []),
            'path': img_path
        })
    
    print(f"✅ Loaded {len(samples)} Vietnamese samples")
    return samples


if __name__ == "__main__":
    samples = load_vietnamese_dataset()
    print(f"Total samples: {len(samples)}")
```

---

## Updating Experiments to Use Real Data

### Update exp_classification.py

Replace the `create_mock_test_dataset()` function:

```python
# OLD (line ~200):
def create_mock_test_dataset(config, samples_per_class=10):
    # ... mock implementation

# NEW:
def create_real_test_dataset(config):
    """Load real PlantVillage data"""
    from evaluations.data.load_plantvillage import load_plantvillage_dataset
    
    known_samples, unknown_samples = load_plantvillage_dataset(
        known_classes=[label.replace(' ', '___') for label in config['known_classes']],
        samples_per_class=50
    )
    
    # Combine and return
    return known_samples + unknown_samples
```

Then update the main function:

```python
# Line ~230:
# test_dataset = create_mock_test_dataset(config, samples_per_class=10)
test_dataset = create_real_test_dataset(config)  # Use real data
```

### Update exp_component analysis.py

For component analysis study with real images, create test cases with actual image paths:

```python
def create_test_case_with_image(image_path: str, ground_truth: Dict) -> Dict:
    """Create test case from real image"""
    from app.pipe import VisionSystem, analyze_full_plant
    
    # Run detection + classification pipeline
    # ... (you'll need to integrate the full pipeline)
    
    return test_case
```

---

## Ground Truth Creation

### For Classification (Exp 5.3)

Use existing PlantVillage labels - they're already annotated!

### For Ablation/RAG (Exp 5.4, 5.6)

You need **expert validation**. Options:

1. **Literature-based**: Use textbook recommendations
2. **Expert consultation**: Get agronomist to review 30-50 cases
3. **Self-annotation**: Carefully research each disease and create ground truth

**Template** (`evaluations/data/ground_truth_template.json`):

```json
{
  "TC001": {
    "image": "data/test_images/tomato_early_blight_high.jpg",
    "expected_diagnosis": {
      "severity_level": "High",
      "overall_health_status": "Severe Infestation",
      "should_have_pesticides": true,
      "key_treatment": "copper fungicide",
      "scoring_rubric": {
        "5": "Recommends copper fungicide with correct dosage",
        "3": "Recommends generic fungicide",
        "1": "Wrong treatment or hallucinated chemicals"
      }
    }
  }
}
```

---

## Validation Checklist

Before running final experiments:

- [] Downloaded PlantVillage dataset
- [ ] Verified dataset structure matches expected format
- [ ] Created known/unknown class split
- [ ] Collected Vietnamese crop images (at least 100)
- [ ] Created labels.json for Vietnamese dataset
- [ ] Updated experiment scripts to load real data
- [ ] Created ground truth for at least 30 test cases
- [ ] Tested dataset loading scripts
- [ ] Verified images load correctly
- [ ] Checked that all paths are absolute or relative to project root

---

## Quick Test

```bash
# Test dataset loading
cd evaluations/data
python load_plantvillage.py
python load_vietnamese.py

# If both succeed, you're ready to run real experiments!
```

---

## Notes

1. **Image preprocessing**: PIL will handle most formats, but ensure:
   - RGB format (not RGBA or grayscale)
   - Reasonable size (224x224 to 512x512)
   - Valid image files (not corrupted)

2. **Label mapping**: Ensure your label format matches what CLIP expects:
   - PlantVillage uses: `"Tomato___Early_blight"`
   - CLIP expects: `"Tomato leaf with Early blight"`
   - You may need mapping functions

3. **Data splits**: Use consistent train/test splits if comparing with other work

4. **Ethics**: If using web-scraped images, ensure you have rights to use them

---

For questions or issues, check:
- PlantVillage GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- LlamaIndex docs: https://docs.llamaindex.ai/
- Your project README: `evaluations/README.md`
