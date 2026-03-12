"""
YAML / JSON configuration loading and merging utilities.
"""

from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML or JSON config file and return a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if not _YAML:
                raise ImportError("PyYAML is required. Install with: pip install pyyaml")
            return yaml.safe_load(f) or {}
        elif suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")


def save_config(cfg: Dict[str, Any], path: Union[str, Path]):
    """Persist a config dict to YAML or JSON (inferred from extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    with open(path, "w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if not _YAML:
                raise ImportError("PyYAML is required.")
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2, ensure_ascii=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base* (override wins on conflict)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def resolve_config(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load a config and recursively resolve any ``_base_`` inheritance."""
    path = Path(path)
    if base_dir:
        path = Path(base_dir) / path
    cfg = load_config(path)
    base_file = cfg.pop("_base_", None)
    if base_file:
        parent = resolve_config(base_file, base_dir=path.parent)
        cfg = merge_configs(parent, cfg)
    return cfg
