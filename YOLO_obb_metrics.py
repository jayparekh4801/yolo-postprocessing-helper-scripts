import numpy as np
import os
from pathlib import Path
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm

def parse_obb_line(line):
    """Parse OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 [confidence]"""
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:9]]
    points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    confidence = float(parts[9]) if len(parts) > 9 else 1.0
    return class_id, points, confidence

def calculate_rotated_iou(box1_points, box2_points):
    """Calculate IoU between two oriented bounding boxes"""
    try:
        poly1 = Polygon(box1_points)
        poly2 = Polygon(box2_points)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0

def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold):
    """
    Match predictions to ground truth using greedy algorithm
    Returns: list of (gt_idx, pred_idx, iou, confidence) for matches
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_rotated_iou(pred['points'], gt['points'])
    
    # Sort predictions by confidence (descending)
    pred_indices = sorted(range(len(pred_boxes)), 
                         key=lambda i: pred_boxes[i]['confidence'], 
                         reverse=True)
    
    matched_gt = set()
    matches = []
    
    for pred_idx in pred_indices:
        # Find best IoU with unmatched GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx
        
        # Match if IoU >= threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matches.append({
                'gt_idx': best_gt_idx,
                'pred_idx': pred_idx,
                'iou': best_iou,
                'confidence': pred_boxes[pred_idx]['confidence']
            })
            matched_gt.add(best_gt_idx)
    
    return matches

def calculate_ap(gt_boxes, pred_boxes, iou_threshold):
    """Calculate Average Precision at a specific IoU threshold"""
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate IoU for all pairs
    tp = np.zeros(len(pred_boxes_sorted))
    fp = np.zeros(len(pred_boxes_sorted))
    matched_gt = set()
    
    for pred_idx, pred in enumerate(pred_boxes_sorted):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_rotated_iou(pred['points'], gt['points'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_idx] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # Cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision and Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes)
    
    # Add endpoints for interpolation
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    
    # Ensure precision is monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap

def process_files(gt_folder, pred_folder, target_class=0):
    """Process all file pairs and calculate metrics"""
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    
    # Get all GT files
    gt_files = sorted(gt_path.glob('*.txt'))
    
    # Aggregate data across all images
    all_gt_boxes = []
    all_pred_boxes = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_instances = 0
    
    print(f"Processing {len(gt_files)} files...")
    
    for gt_file in tqdm(gt_files):
        pred_file = pred_path / gt_file.name

        gt_boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id, points, _ = parse_obb_line(line)
                    if class_id == target_class:
                        gt_boxes.append({'points': points})

        pred_boxes = []
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id, points, confidence = parse_obb_line(line)
                        if class_id == target_class:
                            pred_boxes.append({'points': points, 'confidence': confidence})
        
        # Match predictions to GT (IoU 0.5 for mAP calculation)
        matches = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5)
        
        # Count TP, FP, FN for this image
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_instances += len(gt_boxes)
        
        # Store for mAP calculation
        all_gt_boxes.extend(gt_boxes)
        all_pred_boxes.extend(pred_boxes)
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mAP@50
    map50 = calculate_ap(all_gt_boxes, all_pred_boxes, iou_threshold=0.5)
    
    # Calculate mAP@50-95 (average across IoU thresholds)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(all_gt_boxes, all_pred_boxes, iou_threshold=iou_thresh)
        aps.append(ap)
    map50_95 = np.mean(aps)
    
    return {
        'class_id': target_class,
        'gt_instances': total_gt_instances,
        'detected_tp': total_tp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'map50': map50,
        'map50_95': map50_95
    }

def main():
    # Set your paths here
    gt_folder = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/natural_images_test_dataset/labels"
    pred_folder = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/xai_results/body_postprocessing_methods_results"
    output_csv = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/xai_results/body_postprocessing_metrics_results.csv"
    
    print("Calculating metrics for class 0...")
    results = process_files(gt_folder, pred_folder, target_class=0)
    
    # Create DataFrame
    df = pd.DataFrame([results])
    df.columns = ['Class ID', 'GT Instances', 'Detected (TP)', 'Precision', 
                  'Recall', 'F1-Score', 'mAP@50', 'mAP@50-95']
    
    # Save to CSV
    df.to_csv(output_csv, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print(f"Results for Class {results['class_id']}:")
    print(f"{'='*70}")
    print(f"Ground Truth Instances: {results['gt_instances']}")
    print(f"Detected (TP):          {results['detected_tp']}")
    print(f"Precision:              {results['precision']:.4f}")
    print(f"Recall:                 {results['recall']:.4f}")
    print(f"F1-Score:               {results['f1_score']:.4f}")
    print(f"mAP@50:                 {results['map50']:.4f}")
    print(f"mAP@50-95:              {results['map50_95']:.4f}")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_csv}")

if __name__ == "__main__":
    main()