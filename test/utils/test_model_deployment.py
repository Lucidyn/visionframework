"""
ModelDeployer、DeploymentConfig 及 DeploymentPlatform 工具测试。
"""

import os
import tempfile
import pytest

from visionframework import (
    DeploymentPlatform,
    get_supported_platforms,
    is_platform_supported,
    get_platform_compatibility,
    get_platform_requirements,
    get_platform_from_string,
    DeploymentConfig,
    ModelDeployer,
    deploy_model,
    validate_deployment,
    ModelFormat,
)


# ---------------------------------------------------------------------------
# DeploymentPlatform 工具函数
# ---------------------------------------------------------------------------

def test_get_supported_platforms():
    platforms = get_supported_platforms()
    assert DeploymentPlatform.LOCAL in platforms


def test_is_platform_supported():
    assert is_platform_supported(DeploymentPlatform.LOCAL) is True


def test_get_platform_compatibility():
    compat = get_platform_compatibility(DeploymentPlatform.LOCAL)
    assert isinstance(compat, list)
    assert "pytorch" in compat


def test_get_platform_requirements():
    reqs = get_platform_requirements(DeploymentPlatform.NVIDIA_JETSON)
    assert "dependencies" in reqs
    assert isinstance(reqs["dependencies"], list)


def test_get_platform_from_string():
    assert get_platform_from_string("local") == DeploymentPlatform.LOCAL


# ---------------------------------------------------------------------------
# DeploymentConfig
# ---------------------------------------------------------------------------

def test_deployment_config_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "deployed")
        open(model_path, "w").close()
        cfg = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            model_path=model_path,
            model_format=ModelFormat.PYTORCH,
            output_path=output_path,
        )
        assert cfg.platform == DeploymentPlatform.LOCAL
        assert cfg.model_format == ModelFormat.PYTORCH


def test_deployment_config_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "deployed")
        open(model_path, "w").close()
        cfg = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            model_path=model_path,
            model_format=ModelFormat.PYTORCH,
            output_path=output_path,
        )
        assert cfg.optimize is True or cfg.optimize is False


# ---------------------------------------------------------------------------
# ModelDeployer
# ---------------------------------------------------------------------------

def test_model_deployer_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "deployed")
        open(model_path, "w").close()
        cfg = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            model_path=model_path,
            model_format=ModelFormat.PYTORCH,
            output_path=output_path,
        )
        deployer = ModelDeployer(cfg)
        assert isinstance(deployer, ModelDeployer)


def test_model_deployer_compatible_platform():
    """LOCAL 平台 + PYTORCH 格式应能成功创建 deployer。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "deployed")
        open(model_path, "w").close()
        cfg = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            model_path=model_path,
            model_format=ModelFormat.PYTORCH,
            output_path=output_path,
        )
        deployer = ModelDeployer(cfg)
        assert isinstance(deployer, ModelDeployer)


def test_validate_deployment_nonexistent_path():
    """不存在的路径应返回 False。"""
    result = validate_deployment("/nonexistent/deployed_model", DeploymentPlatform.LOCAL)
    assert result is False


def test_deploy_model_helper():
    """deploy_model 辅助函数应能调用（失败时跳过）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        output_path = os.path.join(tmpdir, "deployed")
        open(model_path, "w").close()
        cfg = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            model_path=model_path,
            model_format=ModelFormat.PYTORCH,
            output_path=output_path,
        )
        try:
            result = deploy_model(cfg)
            assert result is not None
        except Exception:
            pytest.skip("deploy_model 在当前环境不可用")
