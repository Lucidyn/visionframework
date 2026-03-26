"""YOLO11 / YOLO26 实例分割：Ultralytics 全尺寸权重推理测试。

默认 ``pytest`` 通过 ``addopts`` 排除本模块（``-m 'not yolo_seg'``）。
安装 ``ultralytics`` 后运行（含推理、``Visualizer`` 叠加与 PNG 落盘）::

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


def _skip_if_corrupt_ultralytics_weight(exc: BaseException, hub_id: str) -> None:
    """不完整下载或损坏的 .pt 会触发 torch zip 读失败，跳过并提示清理缓存。"""
    msg = str(exc).lower()
    if isinstance(exc, RuntimeError) and (
        "zip" in msg
        or "central directory" in msg
        or "pytorchstreamreader" in msg
        or "failed finding" in msg
    ):
        pytest.skip(
            f"权重文件可能损坏或下载不完整（请删除缓存中的 {hub_id} 后重试）: {exc}"
        )


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
    pytest.importorskip("ultralytics")
    cls = ALGORITHMS.get(cls_name)
    img = bus_bgr
    try:
        seg = cls(weights=hub_id, device="cpu", conf=0.25)
        dets = seg.predict(img)
    except RuntimeError as e:
        _skip_if_corrupt_ultralytics_weight(e, hub_id)
        raise
    assert isinstance(dets, list)
    assert len(dets) >= 1, "bus 图应至少检出 1 个实例"
    for d in dets:
        assert d.mask is not None
        assert d.mask.ndim == 2
        assert d.mask.shape[0] == img.shape[0] and d.mask.shape[1] == img.shape[1]

    drawn = _assert_visualization_ok(img, {"detections": dets})
    safe = hub_id.replace(".pt", "").replace("-", "_")
    _assert_png_roundtrip(drawn, tmp_path / f"{cls_name}_{safe}_seg.png")


@pytest.mark.parametrize("cls_name,hub_id", YOLO_SEG_CASES)
def test_taskrunner_segmentation_runtime_yaml(cls_name, hub_id, monkeypatch, tmp_path, bus_bgr):
    pytest.importorskip("ultralytics")
    monkeypatch.chdir(REPO_ROOT)
    from visionframework import TaskRunner

    seg_rel = (
        "configs/segmentation/yolo11/yolo11_seg.yaml"
        if cls_name == "YOLO11Segmenter"
        else "configs/segmentation/yolo26/yolo26_seg.yaml"
    )
    assert (REPO_ROOT / seg_rel).is_file()

    import yaml

    run = tmp_path / "run.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "segmentation",
                "models": {"segmenter": seg_rel},
                "weights": hub_id,
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    try:
        task = TaskRunner(run)
        r = task.process(bus_bgr)
    except RuntimeError as e:
        _skip_if_corrupt_ultralytics_weight(e, hub_id)
        raise
    assert "detections" in r
    for d in r["detections"]:
        assert d.has_mask()

    drawn = _assert_visualization_ok(bus_bgr, r)
    safe = hub_id.replace(".pt", "").replace("-", "_")
    _assert_png_roundtrip(drawn, tmp_path / f"taskrunner_{cls_name}_{safe}_seg.png")
