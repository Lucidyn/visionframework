"""
reid_example.py
Simple ReID feature extraction example.

Usage:
  python examples/reid_example.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import ReIDExtractor


def main():
    # ReIDExtractor config: device, model_path, use_pretrained
    cfg = {"device": "cpu", "use_pretrained": True}
    reid = ReIDExtractor(cfg)
    reid.initialize()

    img = np.zeros((128, 64, 3), dtype=np.uint8)
    # The process method typically accepts an image and list of bboxes
    features = reid.process(img, bboxes=[(0, 0, 64, 128)])
    print(f"Extracted features type: {type(features)}")
    # If ndarray, print feature vector length
    try:
      import numpy as np
      if isinstance(features, (list, tuple)):
        print("Number of feature vectors:", len(features))
      elif isinstance(features, np.ndarray):
        print("Feature shape:", features.shape)
    except Exception:
      pass


if __name__ == "__main__":
    main()
