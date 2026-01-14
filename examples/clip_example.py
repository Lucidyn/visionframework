"""Example showing CLIP usage with the framework's CLIP wrapper."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image
from visionframework.core.processors.clip_extractor import CLIPExtractor


def main():
    clip = CLIPExtractor({"device": "cpu"})
    clip.initialize()

    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    labels = ["a photo of a cat", "a photo of a dog", "a person"]
    scores = clip.zero_shot_classify(img, labels)
    for lbl, s in zip(labels, scores):
        print(lbl, s)


if __name__ == "__main__":
    main()
