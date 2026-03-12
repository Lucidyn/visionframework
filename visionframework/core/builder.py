"""
Builder utilities — assemble models, algorithms and pipelines from config dicts.

权重加载机制：在模型配置中添加 ``weights`` 字段即可自动加载预训练权重。
支持 ``.pt`` / ``.pth`` 格式，兼容完整 checkpoint 和纯 state_dict。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .registry import BACKBONES, NECKS, HEADS, ALGORITHMS, PIPELINES
from .config import load_config, resolve_config

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    """Backbone → Neck → Head 的薄包装层。"""

    def __init__(self, backbone: nn.Module, neck: Optional[nn.Module], head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return self.head(feats)


def _load_weights(model: nn.Module, weights_path: str, strict: bool = False) -> nn.Module:
    """Load pretrained weights into *model*.

    Parameters
    ----------
    weights_path : str
        Path to a ``.pt`` / ``.pth`` file. Can be either a raw ``state_dict``
        or a checkpoint dict containing a ``model`` or ``state_dict`` key.
    strict : bool
        If ``True`` every key must match exactly.  Default ``False`` to
        allow partial loading (e.g. loading backbone-only weights into a
        full model).
    """
    if not os.path.isfile(weights_path):
        logger.warning("权重文件不存在，跳过加载: %s", weights_path)
        return model

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "ema" in ckpt and ckpt["ema"] is not None:
            ema = ckpt["ema"]
            state_dict = ema.state_dict() if isinstance(ema, nn.Module) else ema
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if isinstance(ckpt, nn.Module) else ckpt

    result = model.load_state_dict(state_dict, strict=strict)
    n_loaded = len(state_dict) - len(result.unexpected_keys) if hasattr(result, 'unexpected_keys') else len(state_dict)
    logger.info(
        "已加载权重 %s  (matched=%d, missing=%d, unexpected=%d)",
        Path(weights_path).name,
        n_loaded,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    return model


def build_model(cfg: Dict[str, Any], weights: Optional[str] = None) -> nn.Module:
    """Build a ``ModelWrapper`` from a model-config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``backbone`` and ``head`` sub-dicts (each with ``type``).
        Optionally ``neck``.  A top-level ``weights`` key specifies a
        pretrained checkpoint path.
    weights : str, optional
        Override weights path (takes precedence over ``cfg["weights"]``).
    """
    cfg = cfg.copy()
    weights_path = weights or cfg.pop("weights", None)
    strict = cfg.pop("weights_strict", False)

    backbone = BACKBONES.build(cfg["backbone"])
    neck = NECKS.build(cfg["neck"]) if "neck" in cfg and cfg["neck"] else None
    head = HEADS.build(cfg["head"])
    model = ModelWrapper(backbone, neck, head)

    if weights_path:
        _load_weights(model, weights_path, strict=strict)

    return model


def build_model_from_file(path: str, weights: Optional[str] = None) -> nn.Module:
    """Convenience: load a YAML model config and build the model."""
    cfg = resolve_config(path)
    return build_model(cfg, weights=weights)


def build_algorithm(cfg: Dict[str, Any]) -> Any:
    """Build a registered algorithm from a config dict with ``type``."""
    return ALGORITHMS.build(cfg)


def build_pipeline(cfg: Dict[str, Any]) -> Any:
    """Build a registered pipeline from a config dict with ``type``."""
    return PIPELINES.build(cfg)
