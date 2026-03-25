"""RT-DETR HG: TaskRunner smoke tests (random weights). Optional official ``.pt`` via env vars."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from visionframework import TaskRunner
from visionframework.core.builder import build_model
from visionframework.core.config import resolve_config
from visionframework.tools.convert_ultralytics_rtdetr_hg import convert_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_BUS = REPO_ROOT / "test" / "fixtures" / "bus.jpg"

RTDETR_HG_VARIANTS = [
    ("l", "rtdetr_l.yaml", "RTDETR_L_PT"),
    ("x", "rtdetr_x.yaml", "RTDETR_X_PT"),
]


def _bench_bgr() -> np.ndarray:
    if FIXTURE_BUS.is_file():
        img = cv2.imread(str(FIXTURE_BUS))
        if img is not None:
            return img
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def bench_bgr():
    return _bench_bgr()


@pytest.mark.parametrize("variant,cfg_name,env_key", RTDETR_HG_VARIANTS)
def test_rtdetr_taskrunner_image_smoke(bench_bgr, variant, cfg_name, env_key, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = f"configs/detection/rtdetr/{cfg_name}"
    m = build_model(resolve_config(cfg_path), weights=None)
    ckpt = tmp_path / f"rand_{variant}.pth"
    torch.save(m.state_dict(), ckpt)
    det_cfg = (REPO_ROOT / "configs" / "detection" / "rtdetr" / cfg_name).resolve()
    run_yaml = tmp_path / f"run_{variant}.yaml"
    run_yaml.write_text(
        "\n".join(
            [
                "pipeline: detection",
                "algorithm: RTDETRDetector",
                "models:",
                f"  detector: {det_cfg.as_posix()}",
                f"weights: {ckpt.resolve().as_posix()}",
                "device: cpu",
                "fp16: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    task = TaskRunner(str(run_yaml))
    out = task.process(bench_bgr)
    assert "detections" in out
    assert isinstance(out["detections"], list)


@pytest.mark.rtdetr_official
@pytest.mark.parametrize("variant,cfg_name,env_key", RTDETR_HG_VARIANTS)
def test_rtdetr_official_pt_bus_detections(variant, cfg_name, env_key, tmp_path, monkeypatch):
    pt = os.environ.get(env_key, "").strip()
    if not pt or not Path(pt).is_file():
        pytest.skip(f"未设置或找不到 {env_key}（官方 rtdetr-{variant}.pt 路径）")
    if not FIXTURE_BUS.is_file():
        pytest.skip(f"缺少测试图: {FIXTURE_BUS}")

    monkeypatch.chdir(REPO_ROOT)
    vf_ckpt = tmp_path / f"{variant}_official_vf.pth"
    convert_checkpoint(pt, str(vf_ckpt), variant=variant, verify=True, verify_config=f"configs/detection/rtdetr/{cfg_name}")

    det_cfg = (REPO_ROOT / "configs" / "detection" / "rtdetr" / cfg_name).resolve()
    run_yaml = tmp_path / f"run_{variant}_official.yaml"
    run_yaml.write_text(
        "\n".join(
            [
                "pipeline: detection",
                "algorithm: RTDETRDetector",
                "models:",
                f"  detector: {det_cfg.as_posix()}",
                f"weights: {vf_ckpt.resolve().as_posix()}",
                "device: cpu",
                "fp16: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    img = cv2.imread(str(FIXTURE_BUS))
    task = TaskRunner(str(run_yaml))
    out = task.process(img)
    dets = out.get("detections", [])
    assert len(dets) >= 1, "COCO 预训练应在 bus 图上至少检出 1 个目标"
