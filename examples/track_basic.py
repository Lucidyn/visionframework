"""
track_basic.py
Minimal tracker example using the unified `Tracker` class.

Usage:
  python examples/track_basic.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import Tracker
from visionframework.data.detection import Detection


def main():
    # Create a Tracker configured for IOU tracking. Common options:
    #  - tracker_type: 'iou', 'bytetrack', or 'reid'
    #  - max_age: frames to keep lost tracks
    #  - min_hits: confirmations needed for a track
    tracker_cfg = {"tracker_type": "iou", "max_age": 20, "min_hits": 2}
    tracker = Tracker(tracker_cfg)
    tracker.initialize()

    # Create two dummy detections
    dets = [
        Detection((10, 10, 50, 50), 0.9, 0, "person"),
        Detection((100, 100, 150, 150), 0.85, 1, "car"),
    ]

    # Update tracker with detections (image optional). Returns list of Track objects.
    tracks = tracker.process(dets)
    print(f"Tracker returned {len(tracks)} tracks")
    for t in tracks:
        print(f" - track_id={t.track_id}, class={t.class_name}, bbox={t.bbox}")


if __name__ == "__main__":
    main()
