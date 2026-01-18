"""
roi_example.py
Demonstrate creating an `ROIDetector` from config and testing ROI membership.

Usage:
  python examples/roi_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import ROIDetector


def main():
    cfg = {
      "rois": [
        {"name": "zone1", "type": "rectangle", "points": [[50, 50], [200, 200]]}
      ]
    }
    roi = ROIDetector(cfg)
    roi.initialize()

    # Test a point inside and outside
    inside = roi.is_inside((100, 100))
    outside = roi.is_inside((10, 10))
    print(f"Point (100,100) inside zone1: {inside}")
    print(f"Point (10,10) inside zone1: {outside}")

    # Example: filter detections by ROI membership
    # Suppose we have a detection at (60,60)-(120,120)
    from visionframework.data.detection import Detection
    det = Detection((60, 60, 120, 120), 0.9, 0, "person")
    filtered = roi.process([det])
    print(f"Detections inside ROIs: {len(filtered)}")


if __name__ == "__main__":
    main()
