"""
Unified component registry.

Every registrable component (backbone, neck, head, algorithm, pipeline)
is stored in its own ``Registry`` instance.  Components are registered
via a decorator and looked up by string key at build time.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A name → class mapping with decorator-based registration."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    # -- registration -------------------------------------------------------

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator that registers *cls* under *name* (defaults to cls.__name__)."""
        def wrapper(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise KeyError(f"[{self.name}] '{key}' is already registered")
            self._registry[key] = cls
            return cls
        return wrapper

    def register_module(self, cls: Type, name: Optional[str] = None):
        """Imperative (non-decorator) registration."""
        key = name or cls.__name__
        self._registry[key] = cls

    # -- lookup / build -----------------------------------------------------

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(
                f"[{self.name}] '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, cfg: Dict[str, Any]) -> Any:
        """Instantiate a registered class from a config dict containing 'type'."""
        cfg = cfg.copy()
        type_name = cfg.pop("type")
        cls = self.get(type_name)
        return cls(**cfg)

    # -- introspection ------------------------------------------------------

    def list(self) -> list:
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self):
        return f"Registry(name={self.name}, items={self.list()})"


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------
BACKBONES  = Registry("backbone")
NECKS      = Registry("neck")
HEADS      = Registry("head")
ALGORITHMS = Registry("algorithm")
PIPELINES  = Registry("pipeline")
