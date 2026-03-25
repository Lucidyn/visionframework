"""RT-DETR HG: build, checkpoint round-trip, optional Ultralytics numeric parity."""

import tempfile
from pathlib import Path

import pytest
import torch

from visionframework.core.builder import build_model
from visionframework.core.config import resolve_config
from visionframework.tools.convert_ultralytics_rtdetr_hg import (
    convert_checkpoint,
    ultralytics_key_to_vf,
    vf_key_to_ultralytics,
)

_VARIANTS = [
    ("l", "configs/detection/rtdetr/rtdetr_l.yaml"),
    ("x", "configs/detection/rtdetr/rtdetr_x.yaml"),
]


@pytest.mark.parametrize("variant,vf_cfg", _VARIANTS)
def test_rtdetr_build_and_forward(variant, vf_cfg):
    cfg = resolve_config(vf_cfg)
    m = build_model(cfg, weights=None)
    m.eval()
    x = torch.randn(1, 3, 640, 640)
    y = m(x)
    if isinstance(y, tuple):
        y = y[0]
    assert y.shape == (1, 300, 84)


@pytest.mark.parametrize("variant,vf_cfg", _VARIANTS)
def test_checkpoint_roundtrip_strict(variant, vf_cfg, tmp_path):
    cfg = resolve_config(vf_cfg)
    m0 = build_model(cfg, weights=None)
    ul_sd = {}
    for k, v in m0.state_dict().items():
        uk = vf_key_to_ultralytics(k, variant)
        if uk is not None:
            ul_sd[uk] = v
    raw = tmp_path / "fake_ultra.pt"
    torch.save(ul_sd, raw)
    out = tmp_path / "vf_roundtrip.pth"
    convert_checkpoint(str(raw), str(out), variant=variant, verify=True, verify_config=vf_cfg)


@pytest.mark.parametrize("variant,vf_cfg", _VARIANTS)
def test_optional_ultralytics_output_parity(variant, vf_cfg):
    """When ``ultralytics`` is installed, outputs must match ``RTDETRDetectionModel`` bit-exact (same seed)."""
    pytest.importorskip("ultralytics")
    import ultralytics  # noqa: E402
    from ultralytics.nn.tasks import RTDETRDetectionModel  # noqa: E402

    yml = f"rtdetr-{variant}.yaml"
    cfg_path = str(Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "rt-detr" / yml)
    um = RTDETRDetectionModel(cfg_path, ch=3, nc=80, verbose=False)
    vf_sd = {ultralytics_key_to_vf(k, variant): v for k, v in um.state_dict().items() if ultralytics_key_to_vf(k, variant)}
    cfg = resolve_config(vf_cfg)
    vm = build_model(cfg, weights=None)
    vm.load_state_dict(vf_sd, strict=True)
    um.eval()
    vm.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        yu = um(x)[0]
        yv = vm(x)
        if isinstance(yv, tuple):
            yv = yv[0]
    assert yu.shape == yv.shape
    assert (yu - yv).abs().max().item() == 0.0
