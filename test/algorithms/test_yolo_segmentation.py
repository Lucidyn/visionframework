"""YOLO11 / YOLO26 实例分割：原生 PyTorch 推理测试（无需 ``ultralytics``）。

默认 ``pytest`` 通过 ``addopts`` 排除本模块（``-m 'not yolo_seg'``）。
需可下载或已缓存官方 ``*-seg.pt`` 权重后运行::

    pytest -m yolo_seg test/algorithms/test_yolo_segmentation.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import visionframework.algorithms.segmentation.yolo_segmenter  # noqa: F401 — register ALGORITHMS
from visionframework.core.registry import ALGORITHMS

REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_BUS = REPO_ROOT / "test" / "fixtures" / "bus.jpg"
_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
_ASSETS_Y26 = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


@pytest.fixture(scope="module")
def bus_bgr(tmp_path_factory):
    """测试图：优先 ``test_bus.jpg`` / ``test/fixtures/bus.jpg``，否则尝试下载一次（模块级缓存）。"""
    import urllib.request

    for candidate in (REPO_ROOT / "test_bus.jpg", _FIXTURE_BUS):
        if candidate.is_file():
            img = cv2.imread(str(candidate))
            if img is not None:
                return img

    cache = tmp_path_factory.mktemp("yolo_seg_data") / "bus.jpg"
    if not cache.is_file():
        try:
            urllib.request.urlretrieve(
                "https://ultralytics.com/images/bus.jpg", str(cache)
            )
        except OSError as e:
            pytest.skip(f"无法下载测试图且本地无 bus.jpg: {e}")
    img = cv2.imread(str(cache))
    assert img is not None
    return img


def _ensure_seg_pt(name: str, tmp_path: Path) -> Path:
    """下载或复用 ``*-seg.pt`` 到临时目录。"""
    import urllib.request

    dest = tmp_path / name
    if dest.is_file():
        return dest
    urls = (
        [f"{_ASSETS_Y26}/{name}", f"{_ASSETS_BASE}/{name}"]
        if name.startswith("yolo26")
        else [f"{_ASSETS_BASE}/{name}", f"{_ASSETS_Y26}/{name}"]
    )
    last_err: OSError | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(dest))
            last_err = None
            break
        except OSError as e:
            last_err = e
            continue
    if last_err is not None:
        pytest.skip(f"无法下载权重 {name}: {last_err}")
    if not dest.is_file() or dest.stat().st_size < 1000:
        pytest.skip(f"权重无效或下载失败: {name}")
    return dest


def _seg_yaml_for_hub(hub_id: str, cls_name: str) -> str:
    stem = Path(hub_id).stem
    base = stem.replace("-seg", "")
    family = "yolo11" if cls_name == "YOLO11Segmenter" else "yolo26"
    rel = f"configs/segmentation/{family}/{base}_seg.yaml"
    p = REPO_ROOT / rel
    if not p.is_file():
        pytest.skip(f"缺少模型配置: {rel}")
    return rel


YOLO_SEG_CASES = [
    ("YOLO11Segmenter", "yolo11n-seg.pt"),
    ("YOLO11Segmenter", "yolo11s-seg.pt"),
    ("YOLO11Segmenter", "yolo11m-seg.pt"),
    ("YOLO11Segmenter", "yolo11l-seg.pt"),
    ("YOLO11Segmenter", "yolo11x-seg.pt"),
    ("YOLO26Segmenter", "yolo26n-seg.pt"),
    ("YOLO26Segmenter", "yolo26s-seg.pt"),
    ("YOLO26Segmenter", "yolo26m-seg.pt"),
    ("YOLO26Segmenter", "yolo26l-seg.pt"),
    ("YOLO26Segmenter", "yolo26x-seg.pt"),
]


pytestmark = pytest.mark.yolo_seg


def _assert_visualization_ok(img: np.ndarray, result_dict: dict) -> np.ndarray:
    """Visualizer 对分割结果绘图：尺寸一致、uint8、叠加后与原图有差异。返回绘制图。"""
    from visionframework import Visualizer

    vis = Visualizer()
    drawn = vis.draw(img.copy(), result_dict)
    assert drawn.shape == img.shape
    assert drawn.dtype == np.uint8
    assert np.any(drawn != img), "可视化应在原图上产生可见叠加"
    return drawn


def _assert_png_roundtrip(drawn: np.ndarray, path: Path) -> None:
    """PNG 写入与再解码，保证可视化结果可持久化。"""
    assert cv2.imwrite(str(path), drawn)
    reload = cv2.imread(str(path))
    assert reload is not None
    assert reload.shape == drawn.shape


@pytest.mark.parametrize("cls_name,hub_id", YOLO_SEG_CASES)
def test_yolo_seg_all_sizes_load_and_predict_bus(cls_name, hub_id, bus_bgr, tmp_path):
    cls = ALGORITHMS.get(cls_name)
    seg_yaml = _seg_yaml_for_hub(hub_id, cls_name)
    wpt = _ensure_seg_pt(hub_id, tmp_path)
    seg = cls(
        weights=str(wpt),
        device="cpu",
        conf=0.25,
        model_yaml=str(REPO_ROOT / seg_yaml),
    )
    dets = seg.predict(bus_bgr)
    assert isinstance(dets, list)
    assert len(dets) >= 1, "bus 图应至少检出 1 个实例"
    for d in dets:
        assert d.mask is not None
        assert d.mask.ndim == 2
        assert d.mask.shape[0] == bus_bgr.shape[0] and d.mask.shape[1] == bus_bgr.shape[1]

    drawn = _assert_visualization_ok(bus_bgr, {"detections": dets})
    safe = hub_id.replace(".pt", "").replace("-", "_")
    _assert_png_roundtrip(drawn, tmp_path / f"{cls_name}_{safe}_seg.png")


@pytest.mark.parametrize("cls_name,hub_id", YOLO_SEG_CASES)
def test_taskrunner_segmentation_runtime_yaml(cls_name, hub_id, monkeypatch, tmp_path, bus_bgr):
    monkeypatch.chdir(REPO_ROOT)
    from visionframework import TaskRunner

    seg_yaml = _seg_yaml_for_hub(hub_id, cls_name)
    wpt = _ensure_seg_pt(hub_id, tmp_path)

    import yaml

    run = tmp_path / "run.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "segmentation",
                "models": {"segmenter": seg_yaml},
                "weights": str(wpt),
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    r = task.process(bus_bgr)
    assert "detections" in r
    for d in r["detections"]:
        assert d.has_mask()

    drawn = _assert_visualization_ok(bus_bgr, r)
    safe = hub_id.replace(".pt", "").replace("-", "_")
    _assert_png_roundtrip(drawn, tmp_path / f"taskrunner_{cls_name}_{safe}_seg.png")
