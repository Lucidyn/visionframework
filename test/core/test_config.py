"""Tests for config loading and merging."""

import json
import pytest
from pathlib import Path

from visionframework.core.config import load_config, save_config, merge_configs, resolve_config


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


class TestResolveConfig:
    def test_base_inheritance(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("a: 1\nb: 2\n")
        child = tmp_path / "child.yaml"
        child.write_text(f"_base_: base.yaml\nb: 3\nc: 4\n")
        cfg = resolve_config(child)
        assert cfg == {"a": 1, "b": 3, "c": 4}
