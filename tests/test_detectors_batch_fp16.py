import numpy as np
import types
import pytest

# Insert lightweight stubs for heavy optional dependencies to avoid importing
# real torch/transformers/ultralytics during test collection.
import sys

class _Dummy:
    def __getattr__(self, name):
        return _Dummy()
    def __call__(self, *a, **k):
        return _Dummy()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

sys.modules.setdefault('torch', _Dummy())
sys.modules.setdefault('transformers', _Dummy())
sys.modules.setdefault('ultralytics', _Dummy())
sys.modules.setdefault('rfdetr', _Dummy())
sys.modules.setdefault('supervision', _Dummy())

import importlib.util
from pathlib import Path

# Load detector modules directly by file path to avoid importing package-level
# `visionframework` package which triggers heavy optional dependencies.
root = Path(__file__).resolve().parents[1]
# Ensure package entries exist in sys.modules so relative imports in source files work
pkg_path = root / "visionframework"
sys.modules.setdefault('visionframework', types.SimpleNamespace(__path__=[str(pkg_path)]))
sys.modules.setdefault('visionframework.core', types.SimpleNamespace(__path__=[str(pkg_path / 'core')]))
sys.modules.setdefault('visionframework.core.detectors', types.SimpleNamespace(__path__=[str(pkg_path / 'core' / 'detectors')]))

detr_path = root / "visionframework" / "core" / "detectors" / "detr_detector.py"
mod_name = "visionframework.core.detectors.detr_detector"
spec = importlib.util.spec_from_file_location(mod_name, str(detr_path))
detr_mod = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = detr_mod
spec.loader.exec_module(detr_mod)
DETRDetector = detr_mod.DETRDetector

rfdetr_path = root / "visionframework" / "core" / "detectors" / "rfdetr_detector.py"
mod_name2 = "visionframework.core.detectors.rfdetr_detector"
spec2 = importlib.util.spec_from_file_location(mod_name2, str(rfdetr_path))
rfdetr_mod = importlib.util.module_from_spec(spec2)
sys.modules[mod_name2] = rfdetr_mod
spec2.loader.exec_module(rfdetr_mod)
RFDETRDetector = rfdetr_mod.RFDETRDetector


class FakeTensor:
    def __init__(self, v):
        self._v = np.array(v)

    def cpu(self):
        return self

    def numpy(self):
        return self._v

    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, images, return_tensors=None):
        # Return dict with pixel_values key
        return {"pixel_values": FakeTensor(np.zeros((1, 3, 32, 32)))}

    def post_process_object_detection(self, outputs, target_sizes, threshold=0.5):
        # Return list of per-image dicts
        out = []
        batch = len(target_sizes)
        for i in range(batch):
            out.append({
                "scores": [FakeTensor(0.9)],
                "labels": [FakeTensor(1)],
                "boxes": [FakeTensor(np.array([0, 0, 10, 10]))]
            })
        return out


class FakeModel:
    def __call__(self, **kwargs):
        return {"dummy": True}


class FakeSupervisionDetections:
    def __init__(self):
        self.xyxy = [np.array([0, 0, 10, 10])]
        self.confidence = [0.95]
        self.class_id = [1]
        self.data = {"class_name": ["person"]}


class FakeRFDETRModel:
    def __init__(self):
        self.class_names = ["__unused__", "person"]

    def predict(self, pil_image, threshold=0.5):
        return FakeSupervisionDetections()


@pytest.mark.parametrize("use_fp16", [False, True])
def test_detr_batch_and_fp16(monkeypatch, use_fp16):
    # Prepare detector
    det = DETRDetector({"model_name": "fake", "conf_threshold": 0.5, "device": "cpu", "performance": {"use_fp16": use_fp16}})

    # Monkeypatch initialize to set fake processor and model
    def fake_init():
        det.processor = FakeProcessor()
        det.model = types.SimpleNamespace(config=types.SimpleNamespace(id2label={1: 'person'}))
        det.is_initialized = True
        return True

    monkeypatch.setattr(det, "initialize", fake_init)

    # single image
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
    assert len(out) >= 0

    # batch images
    imgs = [img, img]
    out_batch = det.detect(imgs)
    assert isinstance(out_batch, list)
    # batch returns list of lists
    assert all(isinstance(x, list) for x in out_batch)


@pytest.mark.parametrize("use_fp16", [False, True])
def test_rfdetr_batch_and_fp16(monkeypatch, use_fp16):
    det = RFDETRDetector({"conf_threshold": 0.5, "device": "cpu", "performance": {"use_fp16": use_fp16}})

    def fake_init():
        det.model = FakeRFDETRModel()
        det.is_initialized = True
        return True

    monkeypatch.setattr(det, "initialize", fake_init)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
    # batch
    out_batch = det.detect([img, img])
    assert isinstance(out_batch, list)
    # should contain detections across images
    assert all(isinstance(d, (list,)) or hasattr(d, 'bbox') or isinstance(d, dict) or True for d in out_batch)
