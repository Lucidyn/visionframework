"""
Example demonstrating TrackingEvaluator usage

Shows how to evaluate tracking results using MOTA, MOTP, and IDF1 metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from visionframework import TrackingEvaluator


def main():
    """Example of evaluating tracking performance"""
    
    # Initialize evaluator
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # Example data: 3 frames of predictions vs ground truth
    # Each track is represented as:
    # {
    #     "track_id": int,
    #     "bbox": {"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max}
    # }
    
    pred_tracks = [
        # Frame 0: 2 predictions
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # Frame 1: 2 predictions (track 1 moves, track 2 stays)
        [
            {
                "track_id": 1,
                "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # Frame 2: 1 prediction (track 2 missing)
        [
            {
                "track_id": 1,
                "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}
            }
        ]
    ]
    
    gt_tracks = [
        # Ground truth Frame 0
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # Ground truth Frame 1
        [
            {
                "track_id": 1,
                "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # Ground truth Frame 2
        [
            {
                "track_id": 1,
                "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 105, "y1": 105, "x2": 155, "y2": 155}
            }
        ]
    ]
    
    # Calculate MOTA
    print("=" * 60)
    print("Tracking Evaluation Results")
    print("=" * 60)
    
    mota_result = evaluator.calculate_mota(pred_tracks, gt_tracks)
    print(f"\n MOTA (Multiple Object Tracking Accuracy):")
    print(f"  - MOTA:          {mota_result['MOTA']:.4f}")
    print(f"  - Precision:     {mota_result['precision']:.4f}")
    print(f"  - Recall:        {mota_result['recall']:.4f}")
    print(f"  - Total GT:      {mota_result['total_gt']}")
    print(f"  - False Pos:     {mota_result['total_fp']}")
    print(f"  - False Neg:     {mota_result['total_fn']}")
    print(f"  - ID Switches:   {mota_result['total_id_switches']}")
    
    # Calculate MOTP
    motp_result = evaluator.calculate_motp(pred_tracks, gt_tracks)
    print(f"\n MOTP (Multiple Object Tracking Precision):")
    print(f"  - MOTP:          {motp_result['MOTP']:.4f} pixels")
    print(f"  - Matched Pairs: {motp_result['total_matched_pairs']}")
    
    # Calculate IDF1
    idf1_result = evaluator.calculate_idf1(pred_tracks, gt_tracks)
    print(f"\n IDF1 (ID F1 Score):")
    print(f"  - IDF1:          {idf1_result['IDF1']:.4f}")
    print(f"  - IDTP:          {idf1_result['IDTP']}")
    print(f"  - IDFP:          {idf1_result['IDFP']}")
    print(f"  - IDFN:          {idf1_result['IDFN']}")
    
    # Comprehensive evaluation
    print(f"\n Comprehensive Evaluation:")
    result = evaluator.evaluate(pred_tracks, gt_tracks)
    print(f"  - MOTA:          {result['MOTA']:.4f}")
    print(f"  - MOTP:          {result['MOTP']:.4f} pixels")
    print(f"  - IDF1:          {result['IDF1']:.4f}")
    print(f"  - Precision:     {result['precision']:.4f}")
    print(f"  - Recall:        {result['recall']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
