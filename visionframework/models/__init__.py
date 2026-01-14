"""
Model Manager

Provides unified model management including loading, caching, and downloading.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


class ModelManager:
    """
    Unified model manager for handling model loading, caching, and downloads.
    
    Supports multiple model sources (YOLO, DETR, Hugging Face, etc.) with
    automatic caching and version management.
    """
    
    # Default model cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "visionframework"
    
    # Known model repositories
    MODEL_SOURCES = {
        "yolo": "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
        "clip": "https://huggingface.co/openai/",
        "detr": "https://huggingface.co/facebook/",
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for caching models. Defaults to ~/.cache/visionframework
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_registry: Dict[str, Dict[str, Any]] = {}
    
    def register_model(self, name: str, source: str, config: Dict[str, Any]) -> None:
        """
        Register a model for management.
        
        Args:
            name: Model identifier (e.g., 'yolov8n', 'clip-vit-base')
            source: Model source ('yolo', 'clip', 'detr', etc.)
            config: Model configuration dictionary
        """
        self._model_registry[name] = {
            "source": source,
            "config": config,
            "cached": False,
            "path": None
        }
    
    def get_model_path(self, name: str, download: bool = True) -> Optional[Path]:
        """
        Get path to model file, downloading if necessary.
        
        Args:
            name: Model identifier
            download: Whether to download if not cached
        
        Returns:
            Path to model file, or None if not found and download=False
        """
        # Check if already cached
        cache_path = self.cache_dir / name
        if cache_path.exists():
            return cache_path
        
        # If not cached and download is False, return None
        if not download:
            return None
        
        # Would implement actual download logic here
        # For now, just return None
        return None
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self.cache_dir
    
    def list_cached_models(self) -> list:
        """
        List all cached models.
        
        Returns:
            List of cached model names
        """
        if not self.cache_dir.exists():
            return []
        return [f.name for f in self.cache_dir.iterdir() if f.is_file()]
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache for specific model or entire cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            cache_path = self.cache_dir / model_name
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear entire cache directory
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get registered model information.
        
        Args:
            name: Model identifier
        
        Returns:
            Model information dictionary, or None if not registered
        """
        return self._model_registry.get(name)
    
    def set_cache_dir(self, cache_dir: Path) -> None:
        """
        Set custom cache directory.
        
        Args:
            cache_dir: New cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
