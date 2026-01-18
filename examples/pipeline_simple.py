"""
pipeline_simple.py
Minimal VisionPipeline example: detection (optionally enable tracking).

Usage:
  python examples/pipeline_simple.py
"""
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import VisionPipeline, Visualizer


def main():
    # Pipeline configuration demonstrates common options:
    #  - enable_tracking: 是否启用跟踪
    #  - detector_config: 传递给 Detector 的子配置
    #  - tracker_config: 传递给 Tracker 的配置（如果启用跟踪）
    config = {
        "enable_tracking": False,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25,
            # Optional: categories can be set per-pipeline
            # "categories": ["person", "car"],
        },
        # "tracker_config": {"tracker_type": "iou", "max_age": 30}
    }

    pipeline = VisionPipeline(config)
    pipeline.initialize()

    # Dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)

    results = pipeline.process(frame)
    # results typically contains keys: 'detections' and optionally 'tracks'
    detections = results.get("detections", [])
    tracks = results.get("tracks", [])
    print(f"Pipeline returned {len(detections)} detections, {len(tracks)} tracks")

    vis = Visualizer()
    out = vis.draw_detections(frame, detections)
    cv2.imwrite("output_pipeline_simple.jpg", out)
    print("Saved output_pipeline_simple.jpg")


if __name__ == "__main__":
    main()
