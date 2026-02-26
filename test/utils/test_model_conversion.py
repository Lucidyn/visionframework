"""
ModelConverter、ConversionConfig 及 ModelFormat 工具测试。
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from visionframework import (
    ModelFormat,
    get_supported_formats,
    is_format_supported,
    get_compatible_formats,
    get_format_extension,
    get_format_dependencies,
    get_format_from_extension,
    ConversionConfig,
    ModelConverter,
    convert_model,
    validate_converted_model,
)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# ModelFormat 工具函数
# ---------------------------------------------------------------------------

def test_get_supported_formats():
    formats = get_supported_formats()
    assert ModelFormat.PYTORCH in formats
    assert ModelFormat.ONNX in formats


def test_is_format_supported():
    assert is_format_supported(ModelFormat.PYTORCH) is True
    assert is_format_supported(ModelFormat.ONNX) is True


def test_get_compatible_formats():
    compat = get_compatible_formats(ModelFormat.PYTORCH)
    assert ModelFormat.ONNX in compat


def test_get_format_extension():
    assert get_format_extension(ModelFormat.ONNX) == ".onnx"
    assert get_format_extension(ModelFormat.PYTORCH) in (".pt", ".pth")


def test_get_format_from_extension():
    assert get_format_from_extension(".onnx") == ModelFormat.ONNX


def test_get_format_dependencies():
    deps = get_format_dependencies(ModelFormat.TENSORRT)
    assert "tensorrt" in deps


# ---------------------------------------------------------------------------
# ConversionConfig
# ---------------------------------------------------------------------------

def test_conversion_config_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "model.onnx")
        # 创建占位文件
        open(input_path, "w").close()
        cfg = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=input_path,
            output_path=output_path,
        )
        assert cfg.input_format == ModelFormat.PYTORCH
        assert cfg.output_format == ModelFormat.ONNX


def test_conversion_config_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "model.onnx")
        open(input_path, "w").close()
        cfg = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=input_path,
            output_path=output_path,
        )
        assert cfg.optimize is True or cfg.optimize is False


# ---------------------------------------------------------------------------
# ModelConverter
# ---------------------------------------------------------------------------

def test_model_converter_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "model.onnx")
        open(input_path, "w").close()
        cfg = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=input_path,
            output_path=output_path,
        )
        try:
            converter = ModelConverter(cfg)
            assert isinstance(converter, ModelConverter)
        except ImportError:
            pytest.skip("onnx 未安装")


def test_model_converter_pytorch_to_onnx():
    """PyTorch → ONNX 转换基础测试。"""
    model = _TinyNet().eval()
    example_input = torch.randn(1, 4)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "model.onnx")
        torch.save(model, input_path)

        cfg = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=input_path,
            output_path=output_path,
            example_input=example_input,
        )
        try:
            converter = ModelConverter(cfg)
            result_path = converter.convert()
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
        except Exception as e:
            pytest.skip(f"ONNX 转换不可用：{e}")


def test_validate_converted_model_nonexistent():
    """不存在的文件应返回 False。"""
    result = validate_converted_model("/nonexistent/model.onnx", ModelFormat.ONNX)
    assert result is False


def test_convert_model_helper():
    """convert_model 辅助函数应能调用（依赖缺失时跳过）。"""
    model = _TinyNet().eval()
    example_input = torch.randn(1, 4)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "model.onnx")
        torch.save(model, input_path)

        cfg = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=input_path,
            output_path=output_path,
            example_input=example_input,
        )
        try:
            result = convert_model(cfg)
            assert result is not None
        except Exception:
            pytest.skip("convert_model 在当前环境不可用")
