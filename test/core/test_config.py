"""Tests for config loading and merging."""

import json
import pytest
from pathlib import Path

from visionframework.core.config import (
    load_config,
    save_config,
    merge_configs,
    resolve_config,
    require_detector_weights,
)


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("a: 1\nb: hello\n")
        cfg = load_config(cfg_path)
        assert cfg == {"a": 1, "b": "hello"}

    def test_load_json(self, tmp_path):
        cfg_path = tmp_path / "test.json"
        cfg_path.write_text(json.dumps({"x": 10}))
        cfg = load_config(cfg_path)
        assert cfg == {"x": 10}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_unsupported_format(self, tmp_path):
        cfg_path = tmp_path / "test.txt"
        cfg_path.write_text("data")
        with pytest.raises(ValueError):
            load_config(cfg_path)


class TestSaveConfig:
    def test_save_yaml(self, tmp_path):
        cfg_path = tmp_path / "out.yaml"
        save_config({"k": "v"}, cfg_path)
        loaded = load_config(cfg_path)
        assert loaded["k"] == "v"

    def test_save_json(self, tmp_path):
        cfg_path = tmp_path / "out.json"
        save_config({"k": 42}, cfg_path)
        loaded = load_config(cfg_path)
        assert loaded["k"] == 42


class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"model": {"backbone": {"type": "A"}, "neck": {"type": "B"}}}
        override = {"model": {"backbone": {"depth": 0.5}}}
        result = merge_configs(base, override)
        assert result["model"]["backbone"] == {"type": "A", "depth": 0.5}
        assert result["model"]["neck"]["type"] == "B"


class TestRequireDetectorWeights:
    def test_passes_when_weights_file_exists(self, tmp_path):
        w = tmp_path / "model.pth"
        w.write_bytes(b"0")
        run = tmp_path / "run.yaml"
        run.write_text(f"weights: {w.name}\n", encoding="utf-8")
        require_detector_weights(tmp_path, "run.yaml")

    def test_exits_when_weights_missing(self, tmp_path, monkeypatch):
        import sys

        run = tmp_path / "run.yaml"
        run.write_text("weights: nowhere.pth\n", encoding="utf-8")
        codes = []

        def fake_exit(code):
            codes.append(code)
            raise RuntimeError("exit-stub")

        monkeypatch.setattr(sys, "exit", fake_exit)
        with pytest.raises(RuntimeError, match="exit-stub"):
            require_detector_weights(tmp_path, "run.yaml", hint="get weights")
        assert codes == [1]


class TestResolveConfig:
    def test_base_inheritance(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("a: 1\nb: 2\n")
        child = tmp_path / "child.yaml"
        child.write_text(f"_base_: base.yaml\nb: 3\nc: 4\n")
        cfg = resolve_config(child)
        assert cfg == {"a": 1, "b": 3, "c": 4}
