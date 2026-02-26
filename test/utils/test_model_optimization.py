"""
模型优化测试（量化、剪枝、蒸馏）。
"""

import pytest
import torch
import torch.nn as nn

from visionframework import (
    QuantizationConfig,
    quantize_model,
    PruningConfig,
    prune_model,
    DistillationConfig,
)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------

def test_quantization_config_defaults():
    cfg = QuantizationConfig()
    assert cfg.quantization_type in ("dynamic", "static", "qat")
    assert cfg.verbose is False


def test_quantization_config_custom():
    cfg = QuantizationConfig(quantization_type="dynamic", backend="fbgemm", verbose=True)
    assert cfg.quantization_type == "dynamic"
    assert cfg.backend == "fbgemm"


@pytest.mark.parametrize("backend", ["fbgemm", "qnnpack"])
def test_quantize_model_dynamic(backend):
    model = _TinyNet().eval()
    cfg = QuantizationConfig(quantization_type="dynamic", backend=backend, verbose=False)
    try:
        q_model = quantize_model(model, cfg)
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"量化后端 '{backend}' 不可用：{e}")
    assert isinstance(q_model, nn.Module)
    x = torch.randn(1, 8)
    with torch.no_grad():
        out = q_model(x)
    assert out.shape == (1, 2)


# ---------------------------------------------------------------------------
# 剪枝
# ---------------------------------------------------------------------------

def test_pruning_config_defaults():
    cfg = PruningConfig()
    assert cfg.amount > 0
    assert cfg.verbose is False


def test_pruning_config_custom():
    cfg = PruningConfig(
        pruning_type="l1_unstructured",
        amount=0.3,
        target_modules=[nn.Linear],
        global_pruning=False,
        verbose=False,
    )
    assert cfg.amount == 0.3


def test_prune_model_l1():
    model = _TinyNet()
    cfg = PruningConfig(
        pruning_type="l1_unstructured",
        amount=0.2,
        target_modules=[nn.Linear],
        global_pruning=False,
        verbose=False,
    )
    pruned = prune_model(model, cfg)
    assert isinstance(pruned, nn.Module)
    x = torch.randn(2, 8)
    out = pruned(x)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_prune_model_random():
    model = _TinyNet()
    cfg = PruningConfig(
        pruning_type="random_unstructured",
        amount=0.1,
        target_modules=[nn.Linear],
        global_pruning=False,
        verbose=False,
    )
    pruned = prune_model(model, cfg)
    assert isinstance(pruned, nn.Module)


def test_prune_model_global():
    model = _TinyNet()
    cfg = PruningConfig(
        pruning_type="l1_unstructured",
        amount=0.1,
        target_modules=[nn.Linear],
        global_pruning=True,
        verbose=False,
    )
    pruned = prune_model(model, cfg)
    assert isinstance(pruned, nn.Module)


# ---------------------------------------------------------------------------
# 蒸馏配置
# ---------------------------------------------------------------------------

def test_distillation_config_import():
    cfg = DistillationConfig()
    assert isinstance(cfg, DistillationConfig)


def test_distillation_config_defaults():
    cfg = DistillationConfig()
    assert cfg.temperature > 0
    assert 0 < cfg.alpha <= 1
    assert cfg.epochs > 0
    assert cfg.batch_size > 0
