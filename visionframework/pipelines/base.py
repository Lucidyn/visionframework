"""
Base pipeline interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePipeline(ABC):
    """All pipelines implement ``process(frame)`` → result dict."""

    @abstractmethod
    def process(self, frame) -> Dict[str, Any]:
        ...

    def reset(self):
        """Reset any stateful components (e.g. trackers)."""
        pass
