"""
Basic tests for Config system.

These tests verify:
- Config initialization and loading
- Configuration validation
- Configuration processing results
"""

from visionframework.utils.io.config_models import Config, BaseConfig, DetectorConfig


def test_config_validate() -> None:
    """
    Config should validate configuration values.
    """
    config_dict = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    
    # Validate detector config
    is_valid = Config.validate_config(config_dict, 'detector')
    assert is_valid


def test_config_create_config_chain() -> None:
    """
    Config should create config chain from multiple sources.
    """
    base_config = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
    }
    
    override_config = {
        "device": "cuda",
        "conf_threshold": 0.6,
    }
    
    merged_config = Config.create_config_chain(base_config, override_config)
    assert isinstance(merged_config, dict)
    assert merged_config["model_path"] == "yolov8n.pt"
    assert merged_config["device"] == "cuda"
    assert merged_config["conf_threshold"] == 0.6


def test_config_get_default_config() -> None:
    """
    Config should get default configurations.
    """
    default_detector_config = Config.get_default_detector_config()
    assert isinstance(default_detector_config, dict)
    assert "model_path" in default_detector_config
    assert "device" in default_detector_config


def test_base_config_initialization() -> None:
    """
    BaseConfig should initialize successfully with a dictionary.
    """
    config_dict = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    
    config = BaseConfig(**config_dict)
    assert isinstance(config, BaseConfig)
    
    # Verify config attributes
    assert hasattr(config, "model_path")
    assert hasattr(config, "device")
    assert hasattr(config, "conf_threshold")


def test_detector_config_initialization() -> None:
    """
    DetectorConfig should initialize successfully with a dictionary.
    """
    config_dict = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    
    config = DetectorConfig(**config_dict)
    assert isinstance(config, DetectorConfig)
    
    # Verify config attributes
    assert config.model_path == "yolov8n.pt"
    assert config.device == "cpu"
    assert config.conf_threshold == 0.5


def test_base_config_to_dict() -> None:
    """
    BaseConfig should be convertible to a dictionary.
    """
    config_dict = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
    }
    
    config = BaseConfig(**config_dict)
    
    # Convert to dict
    config_as_dict = config.to_dict()
    assert isinstance(config_as_dict, dict)
    assert config_as_dict["model_path"] == "yolov8n.pt"
    assert config_as_dict["device"] == "cpu"


def test_base_config_validate_config() -> None:
    """
    BaseConfig should validate configuration values.
    """
    config_dict = {
        "model_path": "yolov8n.pt",
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    
    config = BaseConfig(**config_dict)
    
    # Validate should return True for valid config
    is_valid = config.validate_config()
    assert is_valid
