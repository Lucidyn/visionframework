"""
pose_example.py
Minimal pose estimator example. Initializes the estimator and runs on a dummy image.

Usage:
  python examples/pose_example.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import PoseEstimator, Visualizer


def main():
    # PoseEstimator config options: model_path, keypoint_threshold, device
    cfg = {"model_path": "yolov8n-pose.pt", "keypoint_threshold": 0.5, "device": "cpu"}
    pose = PoseEstimator(cfg)
    pose.initialize()

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = pose.process(img)
    print(f"PoseEstimator returned {len(poses)} poses")

    # Visualize poses (if any)
    vis = Visualizer()
    out = vis.draw_poses(img, poses)
    import cv2
    cv2.imwrite("output_pose_example.jpg", out)
    print("Saved output_pose_example.jpg")


if __name__ == "__main__":
    main()
