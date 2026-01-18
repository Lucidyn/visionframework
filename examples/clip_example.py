"""
clip_example.py
Show how to initialize CLIPExtractor and compute zero-shot scores (optional dependency).

Usage:
  python examples/clip_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    from visionframework import CLIPExtractor
except Exception:
    CLIPExtractor = None


def main():
    if CLIPExtractor is None:
        print("CLIPExtractor not available in this environment.")
        return

    clip = CLIPExtractor({"device": "cpu"})
    clip.initialize()

    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    labels = ["a person", "a car", "a dog"]
    scores = clip.zero_shot_classify(img, labels)
    print(list(zip(labels, [f"{s:.4f}" for s in scores])))


if __name__ == "__main__":
    main()
