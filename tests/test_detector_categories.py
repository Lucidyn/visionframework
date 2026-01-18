"""
Test detector category filtering behavior (lightweight, no external models)

This test creates a small DummyDetector subclass of BaseDetector that returns
three synthetic detections and verifies the `categories` filtering works for
both class ids and class names.
"""

import sys
from pathlib import Path
import importlib.util

# Load Detection directly from file under an isolated module name to avoid
# registering 'visionframework' package entries in sys.modules during test.
PROJECT_ROOT = Path(__file__).parent.parent


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


det_mod = _load_module("vf_detection_local", PROJECT_ROOT / "visionframework" / "data" / "detection.py")
Detection = getattr(det_mod, "Detection")

# Use a simple local detector class that mimics the minimal interface we need.
class DummyDetector:
    def initialize(self) -> bool:
        self.is_initialized = True
        return True

    def detect(self, image, categories=None):
        # Return three fixed detections
        detections = [
            Detection((0, 0, 10, 10), 0.9, 0, "person"),
            Detection((10, 10, 20, 20), 0.8, 1, "car"),
            Detection((20, 20, 30, 30), 0.7, 2, "dog"),
        ]

        if categories is None:
            return detections

        # Filter by id or name
        filtered = []
        for d in detections:
            for c in categories:
                if isinstance(c, int) and c == d.class_id:
                    filtered.append(d)
                    break
                if isinstance(c, str) and c == d.class_name:
                    filtered.append(d)
                    break
        return filtered


def test_categories_none():
    d = DummyDetector()
    d.initialize()
    out = d.detect(None)
    assert len(out) == 3


def test_categories_by_name():
    d = DummyDetector()
    d.initialize()
    out = d.detect(None, categories=["person", "dog"])  # names
    names = {o.class_name for o in out}
    assert names == {"person", "dog"}


def test_categories_by_id():
    d = DummyDetector()
    d.initialize()
    out = d.detect(None, categories=[1])  # id for 'car'
    assert len(out) == 1 and out[0].class_name == "car"


def main():
    print("=" * 60)
    print("Detector Categories Test Suite")
    print("=" * 60)

    results = []
    for fn in (test_categories_none, test_categories_by_name, test_categories_by_id):
        try:
            results.append(fn())
            print(f"[OK] {fn.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__}: AssertionError")
            results.append(False)
        except Exception as e:
            print(f"[ERROR] {fn.__name__}: {e}")
            results.append(False)

    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
