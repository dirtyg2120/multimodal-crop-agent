"""
Evaluation Utilities

Common functions for all evaluation experiments.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class EvaluationResult:
    """Standard result format for all experiments"""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None
    
    def save(self, output_dir: str):
        """Save results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        print(f"✅ Results saved to: {filepath}")
        return filepath


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.3f}s")


def calculate_classification_metrics(y_true: List[int], y_pred: List[int], 
                                     y_scores: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate standard classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction confidence scores (optional)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_scores is not None:
        metrics['avg_confidence'] = np.mean(y_scores)
    
    return metrics


def calculate_detection_metrics(pred_boxes: List[List[float]], 
                                gt_boxes: List[List[float]], 
                                iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection metrics (simplified IoU-based)
    
    Args:
        pred_boxes: List of predicted boxes [[x1, y1, x2, y2], ...]
        gt_boxes: List of ground truth boxes
        iou_threshold: IoU threshold for considering a detection correct
    
    Returns:
        Dictionary of metrics
    """
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # Match predictions to ground truth
    true_positives = 0
    matched_gt = set()
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(pred_boxes) if pred_boxes else 0
    recall = true_positives / len(gt_boxes) if gt_boxes else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': len(pred_boxes) - true_positives,
        'false_negatives': len(gt_boxes) - true_positives
    }


def load_ground_truth(filepath: str) -> Dict[str, Any]:
    """Load ground truth annotations from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_results_summary(results: List[EvaluationResult], output_path: str):
    """Create a summary table from multiple experiment results"""
    import pandas as pd
    
    rows = []
    for result in results:
        row = {
            'Experiment': result.experiment_name,
            'Timestamp': result.timestamp,
            **result.config,
            **result.metrics
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"📊 Summary saved to: {output_path}")
    return df


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
    print(f"{'='*50}\n")
