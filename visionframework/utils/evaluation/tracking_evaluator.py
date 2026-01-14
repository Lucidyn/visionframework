"""
Tracking evaluation utilities

Provides standard Multiple Object Tracking (MOT) metrics calculation:
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- IDF1 (ID F1 Score)
- Precision, Recall
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackingEvaluator:
    """
    Tracking performance evaluator
    
    Calculates standard MOT metrics including MOTA, MOTP, IDF1, Precision, and Recall.
    
    Example:
        ```python
        evaluator = TrackingEvaluator()
        
        # Assuming pred_tracks and gt_tracks are lists of Track objects
        metrics = evaluator.calculate_mota(pred_tracks, gt_tracks)
        idf1_score = evaluator.calculate_idf1(pred_tracks, gt_tracks)
        ```
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize tracking evaluator
        
        Args:
            iou_threshold: IoU threshold for detection matching (default: 0.5)
        """
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def _compute_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
        """Compute IoU between two bounding boxes
        
        Args:
            box1: {x1, y1, x2, y2} format or {x, y, w, h} format
            box2: {x1, y1, x2, y2} format or {x, y, w, h} format
        
        Returns:
            IoU score (0-1)
        """
        # Assume boxes are in [x1, y1, x2, y2] format
        x1_min = max(box1['x1'], box2['x1'])
        y1_min = max(box1['y1'], box2['y1'])
        x2_max = min(box1['x2'], box2['x2'])
        y2_max = min(box1['y2'], box2['y2'])
        
        intersection = max(0, x2_max - x1_min) * max(0, y2_max - y1_min)
        
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detections(self, pred_boxes: List[Dict], gt_boxes: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match predicted boxes with ground truth boxes using IoU
        
        Args:
            pred_boxes: List of predicted bounding boxes
            gt_boxes: List of ground truth bounding boxes
        
        Returns:
            (matched_pairs, unmatched_pred_indices, unmatched_gt_indices)
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self._compute_iou(pred_box, gt_box)
        
        # Hungarian algorithm for optimal matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        
        matched_pairs = []
        unmatched_pred = set(range(len(pred_boxes)))
        unmatched_gt = set(range(len(gt_boxes)))
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if iou_matrix[pred_idx, gt_idx] > self.iou_threshold:
                matched_pairs.append((pred_idx, gt_idx))
                unmatched_pred.discard(pred_idx)
                unmatched_gt.discard(gt_idx)
        
        return matched_pairs, list(unmatched_pred), list(unmatched_gt)
    
    def calculate_mota(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate MOTA (Multiple Object Tracking Accuracy)
        
        MOTA = 1 - (FN + FP + ID_sw) / GT_total
        where:
            - FN: False Negatives (missed detections)
            - FP: False Positives (wrong detections)
            - ID_sw: ID Switches (track ID changes)
            - GT_total: Total ground truth objects
        
        Args:
            pred_tracks: List of {track_id, boxes: [{x1, y1, x2, y2}, ...]}
            gt_tracks: List of {track_id, boxes: [{x1, y1, x2, y2}, ...]}
        
        Returns:
            Dictionary with MOTA and related metrics
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        matched_frames = 0
        
        # Ensure we process the same number of frames
        num_frames = min(len(pred_tracks), len(gt_tracks))
        
        prev_pred_id_to_gt_id = {}  # Track ID mapping for detecting switches
        
        for frame_idx in range(num_frames):
            pred_frame_tracks = pred_tracks[frame_idx]
            gt_frame_tracks = gt_tracks[frame_idx]
            
            total_gt += len(gt_frame_tracks)
            
            # Extract bounding boxes
            pred_boxes = [t.get('bbox', {}) for t in pred_frame_tracks]
            gt_boxes = [t.get('bbox', {}) for t in gt_frame_tracks]
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            
            # Match predictions to ground truth
            matched_pairs, unmatched_pred, unmatched_gt = self._match_detections(
                pred_boxes, gt_boxes
            )
            
            matched_frames += 1
            
            # Count false positives and false negatives
            total_fp += len(unmatched_pred)
            total_fn += len(unmatched_gt)
            
            # Count ID switches (simplified)
            current_id_mapping = {}
            for pred_idx, gt_idx in matched_pairs:
                pred_id = pred_frame_tracks[pred_idx].get('track_id', -1)
                gt_id = gt_frame_tracks[gt_idx].get('track_id', -1)
                current_id_mapping[pred_id] = gt_id
                
                if pred_id in prev_pred_id_to_gt_id:
                    if prev_pred_id_to_gt_id[pred_id] != gt_id:
                        total_id_switches += 1
            
            prev_pred_id_to_gt_id = current_id_mapping
        
        mota = (1.0 - (total_fp + total_fn + total_id_switches) / total_gt) if total_gt > 0 else 0.0
        
        return {
            "MOTA": mota,
            "total_gt": total_gt,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_id_switches": total_id_switches,
            "precision": (total_gt - total_fn) / (total_gt - total_fn + total_fp) if (total_gt - total_fn + total_fp) > 0 else 0.0,
            "recall": (total_gt - total_fn) / total_gt if total_gt > 0 else 0.0,
        }
    
    def calculate_motp(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate MOTP (Multiple Object Tracking Precision)
        
        MOTP = (sum of all distances) / (number of matched pairs)
        
        Args:
            pred_tracks: List of predicted tracks
            gt_tracks: List of ground truth tracks
        
        Returns:
            Dictionary with MOTP and related metrics
        """
        total_distance = 0.0
        total_matched_pairs = 0
        
        num_frames = min(len(pred_tracks), len(gt_tracks))
        
        for frame_idx in range(num_frames):
            pred_frame = pred_tracks[frame_idx]
            gt_frame = gt_tracks[frame_idx]
            
            pred_boxes = [t.get('bbox', {}) for t in pred_frame]
            gt_boxes = [t.get('bbox', {}) for t in gt_frame]
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            matched_pairs, _, _ = self._match_detections(pred_boxes, gt_boxes)
            
            for pred_idx, gt_idx in matched_pairs:
                # Compute center distance
                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]
                
                pred_center_x = (pred_box.get('x1', 0) + pred_box.get('x2', 0)) / 2
                pred_center_y = (pred_box.get('y1', 0) + pred_box.get('y2', 0)) / 2
                
                gt_center_x = (gt_box.get('x1', 0) + gt_box.get('x2', 0)) / 2
                gt_center_y = (gt_box.get('y1', 0) + gt_box.get('y2', 0)) / 2
                
                distance = np.sqrt((pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2)
                total_distance += distance
                total_matched_pairs += 1
        
        motp = total_distance / total_matched_pairs if total_matched_pairs > 0 else float('inf')
        
        return {
            "MOTP": motp,
            "total_matched_pairs": total_matched_pairs,
            "total_distance": total_distance
        }
    
    def calculate_idf1(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate IDF1 (ID F1 Score)
        
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        where:
            - IDTP: Correct ID detections
            - IDFP: Wrong ID detections  
            - IDFN: Missed detections
        
        Args:
            pred_tracks: List of predicted tracks
            gt_tracks: List of ground truth tracks
        
        Returns:
            Dictionary with IDF1 and related metrics
        """
        idtp = 0
        idfp = 0
        idfn = 0
        
        num_frames = min(len(pred_tracks), len(gt_tracks))
        
        # Build track ID to detection lists
        pred_id_to_detections = {}  # {track_id: [(frame, idx), ...]}
        gt_id_to_detections = {}
        
        for frame_idx in range(num_frames):
            pred_frame = pred_tracks[frame_idx]
            gt_frame = gt_tracks[frame_idx]
            
            # Group by track ID
            for idx, track in enumerate(pred_frame):
                track_id = track.get('track_id', -1)
                if track_id not in pred_id_to_detections:
                    pred_id_to_detections[track_id] = []
                pred_id_to_detections[track_id].append((frame_idx, idx))
            
            for idx, track in enumerate(gt_frame):
                track_id = track.get('track_id', -1)
                if track_id not in gt_id_to_detections:
                    gt_id_to_detections[track_id] = []
                gt_id_to_detections[track_id].append((frame_idx, idx))
        
        # Simplified IDF1: match predictions and GT by spatial proximity
        for frame_idx in range(num_frames):
            pred_frame = pred_tracks[frame_idx]
            gt_frame = gt_tracks[frame_idx]
            
            pred_boxes = [t.get('bbox', {}) for t in pred_frame]
            gt_boxes = [t.get('bbox', {}) for t in gt_frame]
            
            matched_pairs, unmatched_pred, unmatched_gt = self._match_detections(
                pred_boxes, gt_boxes
            )
            
            # Check if matched predictions have same ID as GT
            for pred_idx, gt_idx in matched_pairs:
                pred_id = pred_frame[pred_idx].get('track_id', -1)
                gt_id = gt_frame[gt_idx].get('track_id', -1)
                
                if pred_id == gt_id:
                    idtp += 1
                else:
                    idfp += 1
            
            idfp += len(unmatched_pred)
            idfn += len(unmatched_gt)
        
        idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
        
        return {
            "IDF1": idf1,
            "IDTP": idtp,
            "IDFP": idfp,
            "IDFN": idfn
        }
    
    def evaluate(
        self,
        pred_tracks: List[Dict[str, Any]],
        gt_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with all metrics
        
        Args:
            pred_tracks: List of predicted tracks
            gt_tracks: List of ground truth tracks
        
        Returns:
            Dictionary containing MOTA, MOTP, IDF1 and all sub-metrics
        """
        mota_result = self.calculate_mota(pred_tracks, gt_tracks)
        motp_result = self.calculate_motp(pred_tracks, gt_tracks)
        idf1_result = self.calculate_idf1(pred_tracks, gt_tracks)
        
        return {
            "MOTA": mota_result["MOTA"],
            "MOTP": motp_result["MOTP"],
            "IDF1": idf1_result["IDF1"],
            "precision": mota_result["precision"],
            "recall": mota_result["recall"],
            "details": {
                "MOTA": mota_result,
                "MOTP": motp_result,
                "IDF1": idf1_result
            }
        }

