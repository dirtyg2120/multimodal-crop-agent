"""
Experiment 5.3: Classification Quality Benchmark

Tests CLIP vs. ResNet-50 on:
1. Known Accuracy: Performance on standard disease classes
2. Unknown Rejection: Ability to detect out-of-distribution samples

This is THE KEY experiment that justifies using CLIP over traditional classifiers.

Dataset Requirements:
- Known classes: 10 common diseases from PlantVillage
- Unknown classes: 5 diseases NOT in CLIP's training set
"""

import glob
import os
import random
import sys
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations.eval_utils import EvaluationResult, Timer, calculate_classification_metrics, print_metrics
from evaluations.clip_classifier import CLIPClassifier
from evaluations.resnet_classifier import ResNetClassifier

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5  # For unknown rejection

# Dataset paths - Edit if needed
PLANTVILLAGE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'plantvillage')

# Label to folder mapping
LABEL_TO_FOLDER = {
    "Tomato leaf with Early blight": "Tomato___Early_blight",
    "Tomato leaf with Late blight": "Tomato___Late_blight",
    "Potato leaf with Late blight": "Potato___Late_blight",
}




def create_test_dataset_config() -> Dict:
    # Known classes: Common diseases CLIP was trained on
    known_classes = [
        "Tomato leaf with Early blight",
        "Tomato leaf with Late blight",
        # "Tomato leaf with Bacterial spot",
        # "Tomato leaf with Leaf Mold",
        # "Tomato leaf with Septoria leaf spot",
        # "Healthy Tomato leaf",
        # "Potato leaf with Early blight",
        # "Potato leaf with Late blight",
        # "Healthy Potato leaf",
        # "Durian leaf with Leaf Blight",
        # "Durian leaf with Leaf Spot",
        # "Healthy Durian leaf",
    ]
    
    # Unknown classes: Diseases NOT in CLIP's training
    unknown_classes = [
        # "Tomato leaf with Tomato Yellow Leaf Curl Virus",  # Might be in training
        # "Corn leaf with Gray leaf spot",  # Different crop
        # "Rice leaf with Blast",  # Different crop
        # "Pepper leaf with Bacterial spot",  # Different crop
        # "Novel disease not in dataset",  # Completely new
        "Potato leaf with Late blight"
    ]
    
    return {
        'known_classes': known_classes,
        'unknown_classes': unknown_classes
    }


def evaluate_classifier(classifier, test_images: List[Dict], is_clip: bool = True) -> Dict:
    """
    Evaluate classifier on known and unknown samples
    
    Args:
        classifier: CLIPClassifier or ResNetClassifier
        test_images: List of dicts with 'image', 'label', 'is_unknown'
        is_clip: Whether classifier supports unknown rejection
    
    Returns:
        Metrics dictionary
    """
    y_true = []
    y_pred = []
    confidences = []
    
    # Separate metrics for known and unknown
    known_correct = 0
    known_total = 0
    unknown_rejected = 0
    unknown_total = 0
    false_rejections = 0  # Known samples incorrectly rejected
    
    print(f"\n🔬 Evaluating on {len(test_images)} samples...")
    
    for item in tqdm(test_images):
        image = item['image']
        true_label = item['label']
        is_unknown_sample = item['is_unknown']
        
        # Get prediction
        pred_label, confidence, is_rejected = classifier.predict(image)
        
        confidences.append(confidence)
        
        if is_unknown_sample:
            # Unknown sample
            unknown_total += 1
            if is_rejected:
                unknown_rejected += 1
            y_true.append(-1)  # Special label for unknown
            y_pred.append(-1 if is_rejected else 0)
        else:
            # Known sample
            known_total += 1
            if is_rejected:
                false_rejections += 1
            else:
                # Check if prediction is correct
                if pred_label == true_label:
                    known_correct += 1
            
            # For sklearn metrics
            true_idx = classifier.known_labels.index(true_label) if true_label in classifier.known_labels else 0
            pred_idx = classifier.known_labels.index(pred_label) if pred_label in classifier.known_labels else 0
            y_true.append(true_idx)
            y_pred.append(pred_idx)
    
    # Calculate metrics
    metrics = {
        # Known class accuracy
        'known_accuracy': known_correct / known_total if known_total > 0 else 0,
        'known_total': known_total,
        
        # Unknown rejection rate
        'unknown_rejection_rate': unknown_rejected / unknown_total if unknown_total > 0 else 0,
        'unknown_total': unknown_total,
        
        # False rejection rate (should be low)
        'false_rejection_rate': false_rejections / known_total if known_total > 0 else 0,
        
        # F1 score for unknown detection
        'unknown_precision': unknown_rejected / (unknown_rejected + false_rejections) if (unknown_rejected + false_rejections) > 0 else 0,
        'unknown_recall': unknown_rejected / unknown_total if unknown_total > 0 else 0,
        
        # Average confidence
        'avg_confidence': np.mean(confidences),
    }
    
    # Calculate F1 for unknown detection
    if metrics['unknown_precision'] + metrics['unknown_recall'] > 0:
        metrics['unknown_f1'] = 2 * (metrics['unknown_precision'] * metrics['unknown_recall']) / \
                                (metrics['unknown_precision'] + metrics['unknown_recall'])
    else:
        metrics['unknown_f1'] = 0
    
    return metrics


def load_dataset(config: Dict, samples_per_class: int = 50) -> List[Dict]:
    test_images = []
    print(f"\n📂 Loading from: {PLANTVILLAGE_DIR}")
    
    # Load known classes
    for clip_label in config['known_classes']:
        folder_name = LABEL_TO_FOLDER.get(clip_label)
        class_dir = os.path.join(PLANTVILLAGE_DIR, folder_name)
        
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        sampled = random.sample(image_files, min(len(image_files), samples_per_class))
        print(f"  ✅ {len(sampled)} from {folder_name}")
        
        for img_path in sampled:
            test_images.append({
                'image': Image.open(img_path).convert('RGB'),
                'label': clip_label,
                'is_unknown': False,
                'path': img_path
            })
    
    # Load unknown classes
    for clip_label in config['unknown_classes']:
        
        folder_name = LABEL_TO_FOLDER.get(clip_label)
        class_dir = os.path.join(PLANTVILLAGE_DIR, folder_name)
        
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        sampled = random.sample(image_files, min(len(image_files), samples_per_class))
        print(f"  ✅ {len(sampled)} unknown from {folder_name}")
        
        for img_path in sampled:
            test_images.append({
                'image': Image.open(img_path).convert('RGB'),
                'label': clip_label,
                'is_unknown': True,
                'path': img_path
            })
    
    print(f"✅ Total: {len(test_images)}\n")
    return test_images



def run_experiment():
    """Main experiment runner"""
    print("\n" + "="*60)
    print("🧪 EXPERIMENT 5.3: CLASSIFICATION BENCHMARK")
    print("="*60)
    
    # Get dataset configuration
    config = create_test_dataset_config()
    print(f"\n📋 Known classes: {len(config['known_classes'])}")
    print(f"📋 Unknown classes: {len(config['unknown_classes'])}")
    
    # Create test dataset - Try to load real data first
    test_dataset = load_dataset(config, samples_per_class=100)
    
    # Initialize classifiers
    print("\n" + "="*60)
    print("🤖 Initializing Classifiers")
    print("="*60)
    
    clip_classifier = CLIPClassifier()
    clip_classifier.set_known_labels(config['known_classes'])
    
    resnet_classifier = ResNetClassifier()
    resnet_classifier.set_known_labels(config['known_classes'])
    
    # Evaluate CLIP
    print("\n" + "="*60)
    print("📊 Evaluating CLIP")
    print("="*60)
    
    with Timer("CLIP Evaluation"):
        clip_metrics = evaluate_classifier(clip_classifier, test_dataset, is_clip=True)
    
    print_metrics(clip_metrics, "CLIP Results")
    
    # Evaluate ResNet
    print("\n" + "="*60)
    print("📊 Evaluating ResNet-50 (MOCK)")
    print("="*60)
    
    with Timer("ResNet Evaluation"):
        resnet_metrics = evaluate_classifier(resnet_classifier, test_dataset, is_clip=False)
    
    print_metrics(resnet_metrics, "ResNet-50 Results")
    
    # Comparison
    print("\n" + "="*60)
    print("🔍 COMPARISON")
    print("="*60)
    
    comparison = {
        'Metric': ['Known Accuracy', 'Unknown Rejection', 'Unknown F1', 'False Rejection'],
        'CLIP': [
            f"{100*clip_metrics['known_accuracy']:.1f}%",
            f"{100*clip_metrics['unknown_rejection_rate']:.1f}%",
            f"{clip_metrics['unknown_f1']:.3f}",
            f"{100*clip_metrics['false_rejection_rate']:.1f}%"
        ],
        'ResNet-50': [
            f"{100*resnet_metrics['known_accuracy']:.1f}%",
            f"{100*resnet_metrics['unknown_rejection_rate']:.1f}%",
            f"{resnet_metrics['unknown_f1']:.3f}",
            f"{100*resnet_metrics['false_rejection_rate']:.1f}%"
        ]
    }
    
    import pandas as pd
    df_comp = pd.DataFrame(comparison)
    print("\n" + df_comp.to_string(index=False))
    
    # Save results
    output_dir = "evaluations/results/classification"
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CLIP results
    clip_result = EvaluationResult(
        experiment_name="exp_5_3_classification_clip",
        timestamp=timestamp,
        config={'model': 'CLIP', 'threshold': CONFIDENCE_THRESHOLD},
        metrics=clip_metrics
    )
    clip_result.save(output_dir)
    
    # ResNet results
    resnet_result = EvaluationResult(
        experiment_name="exp_5_3_classification_resnet",
        timestamp=timestamp,
        config={'model': 'ResNet-50', 'threshold': CONFIDENCE_THRESHOLD},
        metrics=resnet_metrics
    )
    resnet_result.save(output_dir)
    
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    run_experiment()
