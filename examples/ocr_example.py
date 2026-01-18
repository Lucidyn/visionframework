"""
ocr_example.py
Simple OCR example using pytesseract if available.

Requirements (optional): `pip install pytesseract Pillow`
Tesseract binary must be installed separately for pytesseract to work.

Usage:
  python examples/ocr_example.py
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

from visionframework import Detector


def main():
    # Use detector to find text-like regions (here we just detect all objects)
    detector = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
    detector.initialize()

    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)

    detections = detector.detect(img)
    print(f"Found {len(detections)} detections (will attempt OCR on each)")

    if not TESSERACT_AVAILABLE:
        print("pytesseract not available — install with: pip install pytesseract Pillow")
        return

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = map(int, d.bbox)
        crop = img[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil)
        print(f"Detection {i}: {d.class_name} -> OCR text: {text.strip()}")


if __name__ == "__main__":
    main()
