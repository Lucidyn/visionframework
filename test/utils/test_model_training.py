"""
ModelFineTuner 与 FineTuningConfig 测试。
"""

import pytest
import torch
import torch.nn as nn

from visionframework import FineTuningConfig, FineTuningStrategy, ModelFineTuner


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# FineTuningConfig
# ---------------------------------------------------------------------------

def test_fine_tuning_config_defaults():
    cfg = FineTuningConfig()
    assert isinstance(cfg, FineTuningConfig)
    assert cfg.strategy in (
        FineTuningStrategy.FULL,
        FineTuningStrategy.FREEZE,
        FineTuningStrategy.LORA,
        FineTuningStrategy.QLORA,
    )


def test_fine_tuning_config_full():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FULL, learning_rate=1e-4)
    assert cfg.strategy == FineTuningStrategy.FULL
    assert cfg.learning_rate == 1e-4


def test_fine_tuning_config_freeze():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FREEZE)
    assert cfg.strategy == FineTuningStrategy.FREEZE


def test_fine_tuning_config_lora():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.LORA)
    assert cfg.strategy == FineTuningStrategy.LORA


def test_fine_tuning_config_qlora():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.QLORA)
    assert cfg.strategy == FineTuningStrategy.QLORA


# ---------------------------------------------------------------------------
# ModelFineTuner
# ---------------------------------------------------------------------------

def test_model_fine_tuner_creation_full():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FULL)
    tuner = ModelFineTuner(cfg)
    assert isinstance(tuner, ModelFineTuner)


def test_model_fine_tuner_creation_freeze():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FREEZE)
    tuner = ModelFineTuner(cfg)
    assert isinstance(tuner, ModelFineTuner)


def test_model_fine_tuner_creation_lora():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.LORA)
    tuner = ModelFineTuner(cfg)
    assert isinstance(tuner, ModelFineTuner)


def test_model_fine_tuner_creation_qlora():
    cfg = FineTuningConfig(strategy=FineTuningStrategy.QLORA)
    tuner = ModelFineTuner(cfg)
    assert isinstance(tuner, ModelFineTuner)


def test_model_fine_tuner_prepare_model_full():
    model = _TinyNet()
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FULL)
    tuner = ModelFineTuner(cfg)
    prepared = tuner._prepare_model(model, FineTuningStrategy.FULL)
    assert isinstance(prepared, nn.Module)
    assert any(p.requires_grad for p in prepared.parameters())


def test_model_fine_tuner_prepare_model_freeze():
    model = _TinyNet()
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FREEZE)
    tuner = ModelFineTuner(cfg)
    prepared = tuner._prepare_model(model, FineTuningStrategy.FREEZE)
    assert isinstance(prepared, nn.Module)


def test_model_fine_tuner_prepare_model_lora():
    model = _TinyNet()
    cfg = FineTuningConfig(strategy=FineTuningStrategy.LORA)
    tuner = ModelFineTuner(cfg)
    prepared = tuner._prepare_model(model, FineTuningStrategy.LORA)
    assert isinstance(prepared, nn.Module)


def test_model_fine_tuner_create_optimizer():
    model = _TinyNet()
    cfg = FineTuningConfig(strategy=FineTuningStrategy.FULL, learning_rate=1e-3)
    tuner = ModelFineTuner(cfg)
    optimizer = tuner._create_optimizer(model)
    assert isinstance(optimizer, torch.optim.Optimizer)
