"""Tests for the Registry system."""

import pytest
from visionframework.core.registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            pass

        assert reg.get("Foo") is Foo

    def test_register_default_name(self):
        reg = Registry("test")

        @reg.register()
        class Bar:
            pass

        assert reg.get("Bar") is Bar

    def test_duplicate_raises(self):
        reg = Registry("test")

        @reg.register("Dup")
        class A:
            pass

        with pytest.raises(KeyError):
            @reg.register("Dup")
            class B:
                pass

    def test_get_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.get("NonExistent")

    def test_build(self):
        reg = Registry("test")

        @reg.register("Widget")
        class Widget:
            def __init__(self, size=10):
                self.size = size

        obj = reg.build({"type": "Widget", "size": 42})
        assert obj.size == 42

    def test_list(self):
        reg = Registry("test")

        @reg.register("A")
        class A:
            pass

        @reg.register("B")
        class B:
            pass

        assert sorted(reg.list()) == ["A", "B"]

    def test_contains(self):
        reg = Registry("test")

        @reg.register("X")
        class X:
            pass

        assert "X" in reg
        assert "Y" not in reg

    def test_register_module_imperative(self):
        reg = Registry("test")

        class Cls:
            pass

        reg.register_module(Cls, "MyCls")
        assert reg.get("MyCls") is Cls


class TestGlobalRegistries:
    def test_backbones_has_cspdarknet(self):
        from visionframework.core.registry import BACKBONES
        import visionframework.models  # noqa: F401
        assert "CSPDarknet" in BACKBONES

    def test_backbones_has_resnet(self):
        from visionframework.core.registry import BACKBONES
        import visionframework.models  # noqa: F401
        assert "ResNet" in BACKBONES

    def test_necks_has_pan(self):
        from visionframework.core.registry import NECKS
        import visionframework.models  # noqa: F401
        assert "PAN" in NECKS

    def test_necks_has_fpn(self):
        from visionframework.core.registry import NECKS
        import visionframework.models  # noqa: F401
        assert "FPN" in NECKS

    def test_heads_has_yolohead(self):
        from visionframework.core.registry import HEADS
        import visionframework.models  # noqa: F401
        assert "YOLOHead" in HEADS

    def test_heads_has_seghead(self):
        from visionframework.core.registry import HEADS
        import visionframework.models  # noqa: F401
        assert "SegHead" in HEADS

    def test_heads_has_reidhead(self):
        from visionframework.core.registry import HEADS
        import visionframework.models  # noqa: F401
        assert "ReIDHead" in HEADS
