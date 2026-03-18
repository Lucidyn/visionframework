from pathlib import Path

import cv2
import numpy as np
import pytest

from visionframework.algorithms.detection.rfdetr_pth_detector import RFDETRPTHDetector


def _download_bus(tmp_path: Path) -> np.ndarray:
    import urllib.request

    p = tmp_path / "bus.jpg"
    if not p.exists():
        try:
            urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", str(p))
        except Exception:
            return (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    img = cv2.imread(str(p))
    assert img is not None
    return img


def test_rfdetr_pth_nano_matches_rfdetr_package(tmp_path):
    pytest.importorskip("rfdetr")
    from rfdetr import RFDETRNano

    img = _download_bus(tmp_path)

    # our implementation
    det = RFDETRPTHDetector(model_size="nano", weights="rf-detr-nano.pth", resolution=384, conf=0.5, device="cpu")
    vf = det.predict(img)

    # reference
    ref = RFDETRNano()
    ref_dets = ref.predict(img[:, :, ::-1].copy(), threshold=0.5)

    vf_sorted = sorted(vf, key=lambda d: d.confidence, reverse=True)[:20]
    ref_pairs = list(zip(list(ref_dets.class_id)[:50], list(ref_dets.confidence)[:50]))

    assert len(vf_sorted) > 0
    assert len(ref_pairs) > 0

    remaining = ref_pairs.copy()
    matched = 0
    for d in vf_sorted:
        dc = int(d.class_id)
        ds = float(d.confidence)
        hit = None
        for i, (rc, rs) in enumerate(remaining):
            if int(rc) == dc and abs(float(rs) - ds) < 0.15:
                hit = i
                break
        if hit is not None:
            remaining.pop(hit)
            matched += 1

    assert matched >= max(1, len(vf_sorted) // 2)

