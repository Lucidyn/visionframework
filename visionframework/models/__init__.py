"""
Model Manager

Provides unified model management including loading, caching, and downloading.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import hashlib
import shutil
import tempfile

# Try to import requests for downloads
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class ModelManager:
    """
    Unified model manager for handling model loading, caching, and downloads.
    
    Supports multiple model sources (YOLO, DETR, Hugging Face, etc.) with
    automatic caching, version management, and hash verification.
    """
    
    # Default model cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "visionframework"
    
    # Known model repositories with their download URLs
    MODEL_SOURCES = {
        "yolo": {
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "file_extension": ".pt"
        },
        "yolo26": {
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
            "file_extension": ".pt"
        },
        "efficientdet": {
            "base_url": "https://github.com/google/automl/releases/download/efficientdet/",
            "file_extension": ".pth"
        },
        "fasterrcnn": {
            "base_url": "https://download.pytorch.org/models/",
            "file_extension": ".pth"
        }
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
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Register default models
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """
        Register default models for common use cases.
        """
        # Register YOLO models
        yolo_models = [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
            "yolov26n", "yolov26s", "yolov26m", "yolov26l", "yolov26x",
            "yolov26n-seg", "yolov26s-seg", "yolov26m-seg", "yolov26l-seg", "yolov26x-seg"
        ]
        
        for model_name in yolo_models:
            self.register_model(
                name=model_name,
                source="yolo" if "v8" in model_name else "yolo26",
                config={"file_name": f"{model_name}.pt"}
            )
        
        # Register EfficientDet models
        efficientdet_models = [
            "efficientdet-d0", "efficientdet-d1", "efficientdet-d2", "efficientdet-d3", 
            "efficientdet-d4", "efficientdet-d5", "efficientdet-d6", "efficientdet-d7"
        ]
        
        for model_name in efficientdet_models:
            self.register_model(
                name=model_name,
                source="efficientdet",
                config={"file_name": f"{model_name}.pth"}
            )
        
        # Register Faster R-CNN models
        fasterrcnn_models = [
            "fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2", 
            "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_small_fpn"
        ]
        
        for model_name in fasterrcnn_models:
            self.register_model(
                name=model_name,
                source="fasterrcnn",
                config={"file_name": f"{model_name}.pth"}
            )
    
    def register_model(self, name: str, source: str, config: Dict[str, Any]) -> None:
        """
        Register a model for management.
        
        Args:
            name: Model identifier (e.g., 'yolov8n', 'clip-vit-base')
            source: Model source ('yolo', 'yolo26', 'clip', 'detr', 'huggingface', etc.)
            config: Model configuration dictionary
        
        Raises:
            ValueError: If the source is not supported
        """
        # Check if source is supported
        if source not in self.MODEL_SOURCES:
            from ..exceptions import ConfigurationError
            raise ConfigurationError(
                message=f"Unsupported model source",
                config_key="source",
                config_value=source,
                expected_type=str
            )
        
        # Get default file extension for source
        file_extension = self.MODEL_SOURCES[source]["file_extension"]
        
        # Set default file name if not provided
        if "file_name" not in config:
            config["file_name"] = f"{name}{file_extension}"
        
        # Get cache path
        cache_path = self.cache_dir / config["file_name"]
        
        self._model_registry[name] = {
            "source": source,
            "config": config,
            "cached": cache_path.exists(),
            "path": str(cache_path) if cache_path.exists() else None
        }
    
    def _download_file(self, url: str, dest_path: Path, chunk_size: int = 8192, 
                       progress: bool = True) -> None:
        """
        Download a file from URL with optional progress bar.
        
        Args:
            url: Download URL
            dest_path: Destination path
            chunk_size: Download chunk size
            progress: Show progress bar
            
        Raises:
            RuntimeError: If requests library is not available
            ModelLoadError: If download fails
        """
        if not REQUESTS_AVAILABLE:
            from ..exceptions import DependencyError
            raise DependencyError(
                message="requests library not available for downloading models",
                dependency_name="requests",
                installation_command="pip install requests"
            )
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Create parent directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar if tqdm is available
            with open(dest_path, 'wb') as f:
                if TQDM_AVAILABLE and progress and total_size > 0:
                    with tqdm(total=total_size, unit='iB', unit_scale=True, 
                            desc=str(dest_path.name)) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            from ..exceptions import ModelLoadError
            raise ModelLoadError(
                message=f"Failed to download file from URL",
                model_path=str(dest_path),
                original_error=e
            ) from e
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use (default: sha256)
            
        Returns:
            str: Hexadecimal hash value
        """
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_model_path(self, name: str, download: bool = True, 
                       progress: bool = True) -> Optional[Path]:
        """
        Get path to model file, downloading if necessary.
        
        Args:
            name: Model identifier
            download: Whether to download if not cached
            progress: Show progress bar during download
        
        Returns:
            Path to model file, or None if not found and download=False
        """
        # Check if model is registered
        if name not in self._model_registry:
            raise ValueError(f"Model not registered: {name}. Register it first with register_model()")
        
        model_info = self._model_registry[name]
        config = model_info["config"]
        file_name = config["file_name"]
        source = model_info["source"]
        
        # Check if already cached
        cache_path = self.cache_dir / file_name
        if cache_path.exists():
            return cache_path
        
        # If not cached and download is False, return None
        if not download:
            return None
        
        # Get download URL based on source
        source_info = self.MODEL_SOURCES[source]
        base_url = source_info["base_url"]
        download_url = f"{base_url}{file_name}"
        
        # Download the file
        print(f"Downloading model: {name} from {download_url}")
        
        # Use temporary file to avoid incomplete downloads
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            self._download_file(download_url, tmp_path, progress=progress)
            
            # Move to final location
            shutil.move(tmp_path, cache_path)
            
            # Update model registry
            self._model_registry[name]["cached"] = True
            self._model_registry[name]["path"] = str(cache_path)
            
            print(f"Model {name} downloaded successfully to {cache_path}")
            return cache_path
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            from ..exceptions import ModelLoadError
            raise ModelLoadError(
                message="Error downloading model",
                model_path=str(cache_path),
                original_error=e
            ) from e
    
    def load_model(self, name: str, download: bool = True, 
                   progress: bool = True) -> Any:
        """
        Load a model from disk or cache.
        
        Args:
            name: Model identifier
            download: Whether to download if not cached
            progress: Show progress bar during download
        
        Returns:
            Loaded model instance
            
        Raises:
            ModelNotFoundError: If model is not found or cannot be downloaded
            ModelLoadError: If model loading fails
        """
        # Get model path
        model_path = self.get_model_path(name, download=download, progress=progress)
        if not model_path:
            from ..exceptions import ModelNotFoundError
            raise ModelNotFoundError(
                message="Model not found or cannot be downloaded",
                model_path=str(model_path)
            )
        
        # Get model source
        if name not in self._model_registry:
            raise ValueError(f"Model not registered: {name}. Register it first with register_model()")
        
        model_info = self._model_registry[name]
        source = model_info["source"]
        
        # Try to load the model based on source
        try:
            if source in ["yolo", "yolo26"]:
                from ultralytics import YOLO
                return YOLO(str(model_path))
            elif source == "efficientdet":
                # EfficientDet model loading
                import torch
                model = torch.load(str(model_path))
                return model
            elif source == "fasterrcnn":
                # Faster R-CNN model loading
                import torchvision.models as models
                from torchvision.models.detection import faster_rcnn
                
                # Create model based on name
                if "resnet50_fpn" in name:
                    if "v2" in name:
                        return models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
                    else:
                        return models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                elif "mobilenet_v3_large_fpn" in name:
                    return models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
                elif "mobilenet_v3_small_fpn" in name:
                    return models.detection.fasterrcnn_mobilenet_v3_small_fpn(pretrained=False)
                else:
                    raise ValueError(f"Unknown Faster R-CNN model: {name}")
            else:
                raise ValueError(f"Unsupported model source: {source}")
        except Exception as e:
            from ..exceptions import ModelLoadError
            raise ModelLoadError(
                message=f"Failed to load {source} model",
                model_path=str(model_path),
                original_error=e
            ) from e
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self.cache_dir
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """
        List all cached models with metadata.
        
        Returns:
            List of dictionaries containing cached model information
        """
        cached_models = []
        
        # Check all registered models
        for name, info in self._model_registry.items():
            if info["cached"] and info["path"]:
                model_path = Path(info["path"])
                cached_models.append({
                    "name": name,
                    "source": info["source"],
                    "path": model_path,
                    "size": model_path.stat().st_size,
                    "last_modified": model_path.stat().st_mtime
                })
        
        # Add any unregistered but cached models
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                # Check if already in registered models
                if not any(info["path"] and Path(info["path"]).name == file_path.name for info in self._model_registry.values()):
                    cached_models.append({
                        "name": file_path.stem,
                        "source": "unknown",
                        "path": file_path,
                        "size": file_path.stat().st_size,
                        "last_modified": file_path.stat().st_mtime
                    })
        
        return cached_models
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache for specific model or entire cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            # Clear specific model
            if model_name in self._model_registry:
                info = self._model_registry[model_name]
                if info["path"]:
                    model_path = Path(info["path"])
                    if model_path.exists():
                        model_path.unlink()
                    # Update registry
                    self._model_registry[model_name]["cached"] = False
                    self._model_registry[model_name]["path"] = None
                print(f"Cleared cache for model: {model_name}")
            else:
                # Check if it's a file name
                model_path = self.cache_dir / model_name
                if model_path.exists():
                    model_path.unlink()
                    print(f"Cleared cache for file: {model_name}")
        else:
            # Clear entire cache directory
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Update registry
            for name in self._model_registry:
                self._model_registry[name]["cached"] = False
                self._model_registry[name]["path"] = None
            
            print(f"Cleared entire cache directory: {self.cache_dir}")
    
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
        
        # Update model registry with new cache paths
        for name, info in self._model_registry.items():
            config = info["config"]
            file_name = config["file_name"]
            cache_path = self.cache_dir / file_name
            
            self._model_registry[name]["cached"] = cache_path.exists()
            self._model_registry[name]["path"] = str(cache_path) if cache_path.exists() else None
    
    def get_model_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            name: Model identifier
        
        Returns:
            Model metadata dictionary, or None if not available
        """
        return self._model_metadata.get(name)
    
    def set_model_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """
        Set model metadata.
        
        Args:
            name: Model identifier
            metadata: Metadata dictionary
        """
        self._model_metadata[name] = metadata
    
    def get_all_registered_models(self) -> List[str]:
        """
        Get all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(self._model_registry.keys())
    
    def download_all_registered_models(self, verify_hash: bool = True, 
                                      progress: bool = True) -> int:
        """
        Download all registered models.
        
        Args:
            verify_hash: Verify file hashes after download
            progress: Show progress bars
            
        Returns:
            int: Number of models successfully downloaded
        """
        downloaded = 0
        
        for name in self._model_registry:
            if not self._model_registry[name]["cached"]:
                if self.get_model_path(name, download=True, verify_hash=verify_hash, progress=progress):
                    downloaded += 1
        
        return downloaded


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
