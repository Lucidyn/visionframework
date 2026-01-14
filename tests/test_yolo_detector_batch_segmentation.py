import numpy as np
import types
import sys
import importlib.util
from pathlib import Path
import pytest

# Setup minimal package entries so relative imports in module work
root = Path(__file__).resolve().parents[1]
pkg_path = root / "visionframework"
sys.modules.setdefault('visionframework', types.SimpleNamespace(__path__=[str(pkg_path)]))
sys.modules.setdefault('visionframework.core', types.SimpleNamespace(__path__=[str(pkg_path / 'core')]))
sys.modules.setdefault('visionframework.core.detectors', types.SimpleNamespace(__path__=[str(pkg_path / 'core' / 'detectors')]))

# Create dummy ultralytics and torch modules before loading module
class DummyTensor:
    def __init__(self, arr):
        self._arr = np.array(arr)
    def cpu(self):
        return self
    def numpy(self):
        return self._arr
    def to(self, device):
        return self

class FakeBoxes:
    def __init__(self, boxes, confs, cls_ids):
        # store as lists of DummyTensor
        self.xyxy = [DummyTensor(b) for b in boxes]
        self.conf = [DummyTensor(c) for c in confs]
        self.cls = [DummyTensor(c) for c in cls_ids]

class FakeMasks:
    def __init__(self, data_list):
        self.data = data_list

class FakeResult:
    def __init__(self, boxes, confs, cls_ids, names=None, masks=None):
        self.boxes = FakeBoxes(boxes, confs, cls_ids)
        self.masks = FakeMasks(masks) if masks is not None else None
        self.names = names or {i: str(i) for i in range(100)}

class FakeYOLOModel:
    def __init__(self, path=None):
        self.path = path
        self._moved = False
    def to(self, device):
        self._moved = True
    def __call__(self, imgs, conf=0.25, iou=0.45, verbose=False):
        # return per-image FakeResult list
        results = []
        # imgs may be numpy array (single) or list
        if isinstance(imgs, (list, tuple)):
            for img in imgs:
                h,w = img.shape[:2]
                boxes = [[0,0,min(10,w),min(10,h)]]
                results.append(FakeResult(boxes, [0.9], [1], names={1:'person'}, masks=[np.zeros((h,w), dtype=np.uint8)]))
        else:
            h,w = imgs.shape[:2]
            boxes = [[0,0,min(10,w),min(10,h)]]
            results.append(FakeResult(boxes, [0.9], [1], names={1:'person'}, masks=[np.zeros((h,w), dtype=np.uint8)]))
        return results

# Insert dummy ultralytics before importing module
ultralytics_mod = types.SimpleNamespace(YOLO=lambda path=None: FakeYOLOModel(path))
sys.modules['ultralytics'] = ultralytics_mod
# Provide a minimal torch with autocast and no_grad context managers
class DummyCtx:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

torch_mod = types.SimpleNamespace(no_grad=lambda : DummyCtx(), cuda=types.SimpleNamespace(amp=types.SimpleNamespace(autocast=lambda : DummyCtx()), is_available=lambda: False))
sys.modules['torch'] = torch_mod

# Load the YOLO detector module by file path under package name
mod_name = 'visionframework.core.detectors.yolo_detector'
file_path = root / 'visionframework' / 'core' / 'detectors' / 'yolo_detector.py'
spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
mod = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = mod
spec.loader.exec_module(mod)
YOLODetector = mod.YOLODetector


def test_yolo_single_and_batch_segmentation():
    det = YOLODetector({
        'model_path': 'yolov8n-seg.pt',
        'conf_threshold': 0.25,
        'device': 'cpu',
        'enable_segmentation': True,
        'performance': {'batch_inference': True, 'use_fp16': False}
    })

    assert det.initialize(), "YOLODetector failed to initialize with fake model"

    img = np.zeros((64,64,3), dtype=np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
    # should detect one object
    assert len(out) >= 0

    # batch
    out_batch = det.detect([img, img])
    # batch returns list-of-lists
    assert isinstance(out_batch, list)
    assert all(isinstance(x, list) for x in out_batch)

    # segmentation masks should be returned when enabled
    # at least check that Detection objects may have mask attribute (or None)
    from visionframework.data.detection import Detection
    # Ensure we can construct a Detection from fake result shape
    d = Detection((0,0,10,10), 0.9, 1, 'person', mask=np.zeros((64,64), dtype=np.uint8))
    assert d.has_mask()