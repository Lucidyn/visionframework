"""
插件系统测试（PluginRegistry、ModelRegistry、装饰器辅助函数）。
"""

import pytest
from visionframework import (
    PluginRegistry,
    ModelRegistry,
    register_detector,
    register_tracker,
    register_segmenter,
    register_processor,
    register_model,
    register_visualizer,
    register_evaluator,
    register_custom_component,
    plugin_registry,
    model_registry,
)


# ---------------------------------------------------------------------------
# PluginRegistry — 基本注册与检索
# ---------------------------------------------------------------------------

def test_plugin_registry_creation():
    reg = PluginRegistry()
    assert isinstance(reg, PluginRegistry)


def test_register_and_get_detector():
    reg = PluginRegistry()

    class DummyDetector:
        pass

    reg.register_detector("dummy_det", DummyDetector, version="1.0")
    info = reg.get_detector("dummy_det")
    assert info is not None
    assert info["class"] is DummyDetector
    assert info["metadata"]["version"] == "1.0"


def test_register_and_get_tracker():
    reg = PluginRegistry()

    class DummyTracker:
        pass

    reg.register_tracker("dummy_trk", DummyTracker)
    info = reg.get_tracker("dummy_trk")
    assert info is not None
    assert info["class"] is DummyTracker


def test_register_and_get_segmenter():
    reg = PluginRegistry()

    class DummySeg:
        pass

    reg.register_segmenter("dummy_seg", DummySeg)
    assert reg.get_segmenter("dummy_seg")["class"] is DummySeg


def test_register_and_get_processor():
    reg = PluginRegistry()

    class DummyProc:
        pass

    reg.register_processor("dummy_proc", DummyProc)
    assert reg.get_processor("dummy_proc")["class"] is DummyProc


def test_register_and_get_model():
    reg = PluginRegistry()
    loader = lambda: "model_instance"
    reg.register_model("dummy_model", loader)
    info = reg.get_model("dummy_model")
    assert info is not None
    assert info["loader"] is loader


def test_register_and_get_visualizer():
    reg = PluginRegistry()

    class DummyVis:
        pass

    reg.register_visualizer("dummy_vis", DummyVis)
    assert reg.get_visualizer("dummy_vis")["class"] is DummyVis


def test_register_and_get_evaluator():
    reg = PluginRegistry()

    class DummyEval:
        pass

    reg.register_evaluator("dummy_eval", DummyEval)
    assert reg.get_evaluator("dummy_eval")["class"] is DummyEval


def test_register_and_get_custom_component():
    reg = PluginRegistry()
    reg.register_custom_component("my_comp", object(), tag="test")
    info = reg.get_custom_component("my_comp")
    assert info is not None
    assert info["metadata"]["tag"] == "test"


def test_list_methods_return_lists():
    reg = PluginRegistry()

    class A:
        pass

    reg.register_detector("a", A)
    reg.register_tracker("b", A)
    assert "a" in reg.list_detectors()
    assert "b" in reg.list_trackers()
    assert isinstance(reg.list_segmenters(), list)
    assert isinstance(reg.list_models(), list)
    assert isinstance(reg.list_processors(), list)
    assert isinstance(reg.list_visualizers(), list)
    assert isinstance(reg.list_evaluators(), list)
    assert isinstance(reg.list_custom_components(), list)


def test_get_nonexistent_returns_none():
    reg = PluginRegistry()
    assert reg.get_detector("nonexistent") is None
    assert reg.get_tracker("nonexistent") is None
    assert reg.get_model("nonexistent") is None


def test_add_plugin_path():
    reg = PluginRegistry()
    reg.add_plugin_path("/tmp/fake_plugins")
    assert "/tmp/fake_plugins" in reg._plugin_paths
    reg.add_plugin_path("/tmp/fake_plugins")
    assert reg._plugin_paths.count("/tmp/fake_plugins") == 1


def test_load_plugins_from_nonexistent_path_does_not_raise():
    reg = PluginRegistry()
    reg.load_plugins_from_path("/nonexistent/path/to/plugins")


# ---------------------------------------------------------------------------
# 装饰器辅助函数（作用于全局 plugin_registry）
# ---------------------------------------------------------------------------

def test_register_detector_decorator():
    @register_detector("test_det_deco")
    class MyDetector:
        pass

    info = plugin_registry.get_detector("test_det_deco")
    assert info is not None
    assert info["class"] is MyDetector


def test_register_tracker_decorator():
    @register_tracker("test_trk_deco")
    class MyTracker:
        pass

    assert plugin_registry.get_tracker("test_trk_deco") is not None


def test_register_processor_decorator():
    @register_processor("test_proc_deco")
    class MyProcessor:
        pass

    assert plugin_registry.get_processor("test_proc_deco") is not None


def test_register_model_decorator():
    @register_model("test_model_deco", task="detection")
    def my_loader():
        return "model"

    info = plugin_registry.get_model("test_model_deco")
    assert info is not None
    assert info["metadata"]["task"] == "detection"


def test_register_custom_component_decorator():
    @register_custom_component("test_custom_deco")
    class MyComponent:
        pass

    assert plugin_registry.get_custom_component("test_custom_deco") is not None


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

def test_model_registry_creation():
    reg = ModelRegistry()
    assert isinstance(reg, ModelRegistry)


def test_model_registry_register_and_get():
    reg = ModelRegistry()
    reg.register_model("my_model", {"loader": lambda: "loaded", "task": "detection"})
    info = reg.get_model("my_model")
    assert info is not None


def test_model_registry_list_models():
    reg = ModelRegistry()
    reg.register_model("m1", {"loader": lambda: None})
    reg.register_model("m2", {"loader": lambda: None})
    models = reg.list_models()
    assert "m1" in models
    assert "m2" in models


def test_model_registry_unload_model():
    reg = ModelRegistry()
    reg.register_model("m_unload", {"loader": lambda: "obj"})
    reg.unload_model("m_unload")


def test_model_registry_clear_cache():
    reg = ModelRegistry()
    reg.clear_cache()


def test_model_registry_get_nonexistent():
    reg = ModelRegistry()
    assert reg.get_model("does_not_exist") is None
