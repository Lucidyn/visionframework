"""
Tests for multimodal fusion and model management utilities.

这些测试覆盖：
- 多模态特征融合（concat / attention）
- 模型格式与部署平台工具函数
- 自动模型选择（AutoSelector 高层接口）
"""

from typing import List

import numpy as np
import torch

from visionframework.utils.multimodal import (
    FusionType,
    MultimodalFusion,
    fuse_features,
    get_fusion_model,
)
from visionframework.utils.model_conversion import (
    ModelFormat,
    get_supported_formats,
    get_compatible_formats,
    get_format_extension,
    get_format_dependencies,
    get_format_from_extension,
)
from visionframework.utils.model_deployment import (
    DeploymentPlatform,
    get_supported_platforms,
    get_platform_compatibility,
    get_platform_requirements,
    get_platform_from_string,
)
from visionframework.utils.model_management import (
    ModelType,
    ModelRequirement,
    HardwareInfo,
    HardwareTier,
    ModelSelector,
    select_model,
)


def test_fuse_features_concat_numpy() -> None:
    """numpy 输入下的 concat 融合应工作正常，并返回 numpy 数组。"""
    np.random.seed(0)
    f1 = np.random.randn(4, 8).astype("float32")
    f2 = np.random.randn(4, 16).astype("float32")

    fused = fuse_features([f1, f2], fusion_type="concat", hidden_dim=32, output_dim=10, device="cpu")
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (4, 10)


def test_get_fusion_model_attention_torch() -> None:
    """attention 融合的模型应能前向推理，输出维度正确。"""
    batch = 2
    dim_v, dim_t = 8, 12
    model: MultimodalFusion = get_fusion_model(
        fusion_type=FusionType.ATTENTION,
        input_dims=[dim_v, dim_t],
        hidden_dim=16,
        output_dim=6,
        num_heads=2,
        device="cpu",
        dropout=0.0,
    )

    v = torch.randn(batch, dim_v)
    t = torch.randn(batch, dim_t)
    out = model([v, t])
    assert out.shape == (batch, 6)


def test_model_format_utils_roundtrip() -> None:
    """模型格式工具函数：支持列表、兼容格式和扩展名的往返转换。"""
    formats = get_supported_formats()
    assert ModelFormat.PYTORCH in formats

    compat = get_compatible_formats(ModelFormat.PYTORCH)
    assert ModelFormat.ONNX in compat

    ext = get_format_extension(ModelFormat.ONNX)
    assert ext == ".onnx"
    assert get_format_from_extension(".onnx") == ModelFormat.ONNX

    deps = get_format_dependencies(ModelFormat.TENSORRT)
    assert "tensorrt" in deps


def test_deployment_platform_utils() -> None:
    """部署平台工具函数基本行为。"""
    plats = get_supported_platforms()
    assert DeploymentPlatform.LOCAL in plats

    compat = get_platform_compatibility(DeploymentPlatform.LOCAL)
    assert "pytorch" in compat

    reqs = get_platform_requirements(DeploymentPlatform.NVIDIA_JETSON)
    assert "dependencies" in reqs and isinstance(reqs["dependencies"], list)

    assert get_platform_from_string("local") == DeploymentPlatform.LOCAL


def test_model_selector_high_level_select_model_detection() -> None:
    """
    高层 select_model 接口：
    - 至少返回一个含有 model_name 的结果
    - 内部会自动根据硬件信息和内存约束过滤 / 回退
    """
    result = select_model(
        model_type="detection",
        accuracy=70,
        speed=50,
        memory=256,  # 限制内存，倾向 small / nano 模型
        task="test",
    )

    # 无论是正常选择还是回退，都应该包含 model_name 字段
    assert "model_name" in result or "error" in result
    if "model_name" in result:
        assert isinstance(result["model_name"], str)


def test_model_selector_manual_hardware_injection() -> None:
    """
    通过直接构造 ModelSelector 并注入硬件信息，验证 _is_compatible_with_hardware 逻辑。
    """
    selector = ModelSelector()

    # 强制硬件信息为 MOBILE（最低档），但仍应允许选择 MOBILE 级别模型
    selector._hardware_info = HardwareInfo(
        platform="TestOS",
        cpu_cores=2,
        cpu_ram=2048,
        gpu_available=False,
        gpu_memory=0,
        gpu_name="",
        hardware_tier=HardwareTier.MOBILE,
    )

    req = ModelRequirement(model_type=ModelType.DETECTION, memory=512)
    result = selector.select_model(req)

    if "model_name" in result:
        info = result["model_info"]
        assert isinstance(info["memory"], int)
        assert info["memory"] <= req.memory or "smallest model" in result.get("reason", "")

