"""
Experiment 5.4: Few-Shot Engine Threshold Sensitivity

Evaluates the Cosine Similarity threshold used to match new "Unknown"
diseases to registered prototypes.

Methodology:
1. Support Set: Take 5 images of an unknown disease and register it.
2. Positive Query Set: Test remaining images of the same disease (Should Match).
3. Negative Query Set: Test images of a completely different unknown disease (Should Reject).
4. Sweep thresholds from 0.70 to 0.95 to find the optimal F1 score.

Design Note (Thesis Limitation):
  The target disease (Squash Powdery Mildew) is used because it is morphologically
  very different from the Tomato/Grape diseases the CLIP backbone specifically focused on.
  However, as it is sourced from PlantVillage, the base CLIP model may have seen Squash
  during pre-training. For fully rigorous evaluation, a disease from a completely
  separate dataset would be ideal. This is acknowledged as a stated limitation.
"""

import glob
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.few_shot_engine import FewShotEngine, DEFAULT_THRESHOLD

# Configuration
PLANTVILLAGE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'PlantDoc-Dataset-master', 'train')

# Target disease: visually distinct from known Tomato/Grape classes, so CLIP embeddings
# are less saturated for it than for Potato (which shares the nightshade family).
TARGET_DISEASE = "Squash___Powdery mildew"
# Negative disease: completely different crop type — should NOT match the squash prototype.
NEGATIVE_DISEASE = "Blueberry leaf"

K_SHOT = 5         # Number of images to build the centroid
QUERY_SIZE = 50    # Number of images to test per class

def load_images_from_folder(folder_name: str, num_images: int) -> list:
    class_dir = os.path.join(PLANTVILLAGE_DIR, folder_name)
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(class_dir, ext)))
    
    if not image_files:
        print(f"⚠️  No images found in {class_dir}")
        return []
        
    sampled_files = random.sample(image_files, min(len(image_files), num_images))
    return [Image.open(f).convert('RGB') for f in sampled_files]

def run_few_shot_experiment(i):
    print("\n" + "="*60)
    print("EXPERIMENT 5.4: FEW-SHOT THRESHOLD ANALYSIS")
    print("="*60)

    # 1. Initialize Engine (Isolated from production)
    engine = FewShotEngine()
    engine.prototypes = {} # Clear loaded prototypes to ensure isolation
    # Prevent saving over production data during evaluation
    engine._save = lambda: None 

    # 2. Load Data
    print(f"\n📂 Loading {K_SHOT} support images for {TARGET_DISEASE}...")
    target_all_images = load_images_from_folder(TARGET_DISEASE, QUERY_SIZE + K_SHOT)
    
    if len(target_all_images) < K_SHOT + 1:
        print("Not enough images to run experiment.")
        return

    support_images = target_all_images[:K_SHOT]
    positive_queries = target_all_images[K_SHOT:]
    
    print(f"📂 Loading {QUERY_SIZE} negative images from {NEGATIVE_DISEASE}...")
    negative_queries = load_images_from_folder(NEGATIVE_DISEASE, QUERY_SIZE)

    # 3. Register Prototype
    print(f"\n🧠 Registering Prototype ({K_SHOT}-shot)...")
    engine.register("Squash Powdery Mildew", support_images)

    # 4. Evaluate Thresholds
    thresholds = [x / 100.0 for x in range(70, 96, 2)] # 0.70, 0.72... 0.94
    results = []

    print(f"\n🔬 Evaluating {len(positive_queries)} Positives and {len(negative_queries)} Negatives...")
    
    # Get scores for all queries once to save time
    positive_scores = []
    for img in tqdm(positive_queries, desc="Scoring Positives"):
        _, score, _ = engine.identify(img, threshold=0.0)
        positive_scores.append(score)
        
    negative_scores = []
    for img in tqdm(negative_queries, desc="Scoring Negatives"):
        _, score, _ = engine.identify(img, threshold=0.0)
        negative_scores.append(score)

    print(f"\n{'Threshold':<12} {'TPR (Match)':<14} {'FPR (False Alarm)':<18} {'F1 Score'}")
    print("-" * 60)

    for thresh in thresholds:
        # True Positives: Target images scored above threshold
        tp = sum(1 for s in positive_scores if s >= thresh)
        fn = len(positive_scores) - tp
        
        # False Positives: Negative images mistakenly scored above threshold
        fp = sum(1 for s in negative_scores if s >= thresh)
        tn = len(negative_scores) - fp
        
        tpr = tp / len(positive_scores) if positive_scores else 0
        fpr = fp / len(negative_scores) if negative_scores else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'tpr': tpr,
            'fpr': fpr,
            'f1': f1
        })
        with open(f"evaluations/results/few_shot/results_{i}.txt", "w") as f:
            for result in results:
                f.write(f"{result}\n")
        
        print(f"  {thresh:<10.2f} {100*tpr:<14.1f} {100*fpr:<18.1f} {f1:.3f}")

    # 5. Save Plot
    output_dir = "evaluations/results/few_shot"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    tpr_list = [r['tpr'] for r in results]
    fpr_list = [r['fpr'] for r in results]
    f1_list = [r['f1'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr_list, marker='o', label='True Positive Rate (Correct Matches)', color='blue')
    plt.plot(thresholds, fpr_list, marker='s', label='False Positive Rate (Wrong Matches)', color='red')
    plt.plot(thresholds, f1_list, marker='^', label='F1 Score', color='purple', linestyle='--')
    
    # Optional: Plot the default 0.8
    plt.axvline(x=DEFAULT_THRESHOLD, color='green', linestyle=':', label='Current Default (0.82)')
    
    plt.title('Few-Shot Cosine Similarity Threshold Sensitivity')
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"few_shot_threshold_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"\n📊 Saved threshold sensitivity plot to: {plot_path}")
    print("✅ Experiment complete!")

if __name__ == "__main__":
    for i in range(1):
        run_few_shot_experiment(i)
