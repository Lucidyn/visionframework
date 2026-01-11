"""
Detection evaluation utilities
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ...data.detection import Detection


class DetectionEvaluator:
    """Evaluate detection performance"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detection evaluator
        
        Args:
            iou_threshold: IoU threshold for matching detections
        """
        self.iou_threshold = iou_threshold
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(
        self,
        pred_detections: List[Detection],
        gt_detections: List[Detection]
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        """Match predicted detections with ground truth"""
        if len(pred_detections) == 0 or len(gt_detections) == 0:
            return [], [], []
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(pred_detections), len(gt_detections)))
        for i, pred in enumerate(pred_detections):
            for j, gt in enumerate(gt_detections):
                if pred.class_id == gt.class_id:
                    iou_matrix[i, j] = self._calculate_iou(pred.bbox, gt.bbox)
        
        # Greedy matching
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        match_candidates = []
        for i in range(len(pred_detections)):
            for j in range(len(gt_detections)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    match_candidates.append((iou_matrix[i, j], i, j))
        
        match_candidates.sort(reverse=True)
        
        for iou, i, j in match_candidates:
            if i not in matched_pred and j not in matched_gt:
                matches.append((i, j))
                matched_pred.add(i)
                matched_gt.add(j)
        
        return list(matched_pred), list(matched_gt), matches
    
    def calculate_metrics(
        self,
        pred_detections: List[Detection],
        gt_detections: List[Detection]
    ) -> Dict[str, float]:
        """Calculate detection metrics"""
        matched_pred, matched_gt, matches = self.match_detections(pred_detections, gt_detections)
        
        tp = len(matches)
        fp = len(pred_detections) - len(matched_pred)
        fn = len(gt_detections) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    def calculate_map(
        self,
        all_pred_detections: List[List[Detection]],
        all_gt_detections: List[List[Detection]],
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate mAP (mean Average Precision)"""
        if num_classes is None:
            all_class_ids = set()
            for dets in all_gt_detections:
                for det in dets:
                    all_class_ids.add(det.class_id)
            num_classes = len(all_class_ids) if all_class_ids else 1
        
        class_aps = []
        
        for class_id in range(num_classes):
            pred_scores = []
            gt_count = 0
            
            for pred_dets, gt_dets in zip(all_pred_detections, all_gt_detections):
                pred_class = [d for d in pred_dets if d.class_id == class_id]
                gt_class = [d for d in gt_dets if d.class_id == class_id]
                
                gt_count += len(gt_class)
                
                if len(pred_class) == 0:
                    continue
                
                matched_pred, matched_gt, matches = self.match_detections(pred_class, gt_class)
                
                for i, det in enumerate(pred_class):
                    pred_scores.append((det.confidence, i in matched_pred))
            
            if len(pred_scores) == 0:
                class_aps.append(0.0)
                continue
            
            pred_scores.sort(reverse=True)
            
            tp = 0
            fp = 0
            precisions = []
            recalls = []
            
            for score, is_matched in pred_scores:
                if is_matched:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / gt_count if gt_count > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP using 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                p = max([p for r, p in zip(recalls, precisions) if r >= t], default=0.0)
                ap += p / 11.0
            
            class_aps.append(ap)
        
        map_score = np.mean(class_aps) if class_aps else 0.0
        
        return {
            "mAP": map_score,
            "AP_per_class": {i: ap for i, ap in enumerate(class_aps)},
            "num_classes": num_classes
        }

