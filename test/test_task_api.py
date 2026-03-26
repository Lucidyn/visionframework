"""TaskRunner integration: detection algorithms, strict_weights, tracking, Path entry."""

from __future__ import annotations

import yaml
import numpy as np
import pytest
import torch

from pathlib import Path

from visionframework import TaskRunner
from visionframework.core.builder import build_model
from visionframework.core.config import resolve_config
from visionframework.task_api import _build_detection_algorithm

REPO_ROOT = Path(__file__).resolve().parents[1]

_DETECTION_CASES = [
    ("Detector", "configs/detection/yolo11/yolo11s.yaml"),
    ("DETRDetector", "configs/detection/detr/detr_r50.yaml"),
    ("RTDETRDetector", "configs/detection/rtdetr/rtdetr_l.yaml"),
]


@pytest.mark.parametrize("algorithm,model_cfg_rel", _DETECTION_CASES)
def test_taskrunner_detection_algorithms_smoke(algorithm, model_cfg_rel, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / model_cfg_rel
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "rand.pth"
    torch.save(m.state_dict(), wpath)

    run = tmp_path / "run.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "detection",
                "algorithm": algorithm,
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "device": "cpu",
                "fp16": False,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = task.process(img)
    assert "detections" in out
    assert isinstance(out["detections"], list)


def test_taskrunner_accepts_path(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "r.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "detection",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    assert task.pipeline is not None


def test_build_model_strict_weights_missing_raises(tmp_path):
    cfg = {
        "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
        "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
        "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
    }
    with pytest.raises(FileNotFoundError):
        build_model(cfg, weights=str(tmp_path / "nope.pth"), strict_weights=True)


def test_taskrunner_strict_weights_from_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    run = tmp_path / "run.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "detection",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str((tmp_path / "missing.pth").resolve()).replace("\\", "/"),
                "strict_weights": True,
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        TaskRunner(run)


def test_taskrunner_tracking_builds(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    tracker_inline = {"type": "bytetrack", "track_thresh": 0.5}
    run = tmp_path / "track.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "tracking",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "tracker": tracker_inline,
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    r = task.process(img)
    assert "tracks" in r and "detections" in r


@pytest.mark.parametrize(
    "tracker_inline",
    (
        {"type": "iou", "min_hits": 0},
        {"type": "centroid", "min_hits": 0, "max_distance": 200.0},
        {"type": "sort", "min_hits": 0},
        {"type": "deepsort", "min_hits": 0},
    ),
)
def test_taskrunner_tracking_tracker_types(monkeypatch, tmp_path, tracker_inline):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "track.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "tracking",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "tracker": tracker_inline,
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    r = task.process(img)
    assert "tracks" in r and "detections" in r


def test_taskrunner_unknown_tracker_type(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "bad_track.yaml"
    run.write_text(
        yaml.safe_dump(
            {
                "pipeline": "tracking",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "tracker": {"type": "not_a_real_tracker"},
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown tracker type"):
        TaskRunner(run)


@pytest.mark.parametrize("algorithm", ("Detector", "DETRDetector", "RTDETRDetector"))
def test_build_detection_algorithm_factory_matches_registry(algorithm, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rel = {
        "Detector": "configs/detection/yolo11/yolo11s.yaml",
        "DETRDetector": "configs/detection/detr/detr_r50.yaml",
        "RTDETRDetector": "configs/detection/rtdetr/rtdetr_l.yaml",
    }[algorithm]
    cfg_path = REPO_ROOT / rel
    model_cfg = resolve_config(cfg_path)
    model = build_model(model_cfg, weights=None)
    runtime = {"algorithm": algorithm, "device": "cpu", "fp16": False}
    det = _build_detection_algorithm(model, model_cfg, runtime)
    assert det.__class__.__name__ == (
        "Detector" if algorithm == "Detector" else algorithm
    )


def test_process_batch_and_collect_max_frames(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "run.yaml"
    run.write_text(
        __import__("yaml").safe_dump(
            {
                "pipeline": "detection",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    imgs = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    batch_out = task.process_batch(imgs)
    assert len(batch_out) == 3
    assert all("detections" in r for r in batch_out)

    rows = task.collect_results(imgs, max_frames=2)
    assert len(rows) == 2


def test_process_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    img_path = tmp_path / "a.jpg"
    cv2 = __import__("cv2")
    cv2.imwrite(str(img_path), np.zeros((32, 32, 3), dtype=np.uint8))
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "run.yaml"
    run.write_text(
        __import__("yaml").safe_dump(
            {
                "pipeline": "detection",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    task = TaskRunner(run)
    pairs = task.process_paths([img_path])
    assert len(pairs) == 1
    assert pairs[0][0] == str(img_path.resolve())
    assert "detections" in pairs[0][1]


def test_vf_run_cli_smoke(tmp_path, monkeypatch):
    import cv2

    from visionframework.tools.run_inference import main

    monkeypatch.chdir(REPO_ROOT)
    img_path = tmp_path / "x.jpg"
    cv2.imwrite(str(img_path), np.zeros((40, 40, 3), dtype=np.uint8))
    cfg_path = REPO_ROOT / "configs/detection/yolo11/yolo11s.yaml"
    cfg = resolve_config(cfg_path)
    m = build_model(cfg, weights=None)
    wpath = tmp_path / "w.pth"
    torch.save(m.state_dict(), wpath)
    run = tmp_path / "run.yaml"
    run.write_text(
        __import__("yaml").safe_dump(
            {
                "pipeline": "detection",
                "models": {"detector": str(cfg_path.resolve()).replace("\\", "/")},
                "weights": str(wpath.resolve()).replace("\\", "/"),
                "device": "cpu",
            },
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    rc = main(
        [
            "--config",
            str(run),
            "--source",
            str(img_path),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert any(out_dir.glob("*.jpg"))
