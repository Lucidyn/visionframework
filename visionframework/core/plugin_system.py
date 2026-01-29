"""
Plugin system and model registration for Vision Framework

This module provides a plugin system and model registration mechanism for extending
Vision Framework functionality. It allows users to register custom models, detectors,
trackers, and other components without modifying the core codebase.
"""

from typing import Dict, Any, List, Optional, Callable, Type, Union
import importlib
import sys
import os
from pathlib import Path
from ..utils.monitoring.logger import get_logger

logger = get_logger(__name__)


class PluginRegistry:
    """
    Registry for plugins and custom components
    """
    
    def __init__(self):
        """
        Initialize plugin registry
        """
        self._registries: Dict[str, Dict[str, Any]] = {
            "detectors": {},
            "trackers": {},
            "segmenters": {},
            "models": {},
            "processors": {},
            "visualizers": {},
            "evaluators": {},
            "custom_components": {}
        }
        self._plugin_paths: List[str] = []
    
    def register_detector(self, name: str, detector_class: Type, **metadata):
        """
        Register a custom detector
        
        Args:
            name: Detector name
            detector_class: Detector class
            **metadata: Additional metadata
        """
        self._registries["detectors"][name] = {
            "class": detector_class,
            "metadata": metadata
        }
        logger.info(f"Registered detector: {name}")
    
    def register_tracker(self, name: str, tracker_class: Type, **metadata):
        """
        Register a custom tracker
        
        Args:
            name: Tracker name
            tracker_class: Tracker class
            **metadata: Additional metadata
        """
        self._registries["trackers"][name] = {
            "class": tracker_class,
            "metadata": metadata
        }
        logger.info(f"Registered tracker: {name}")
    
    def register_segmenter(self, name: str, segmenter_class: Type, **metadata):
        """
        Register a custom segmenter
        
        Args:
            name: Segmenter name
            segmenter_class: Segmenter class
            **metadata: Additional metadata
        """
        self._registries["segmenters"][name] = {
            "class": segmenter_class,
            "metadata": metadata
        }
        logger.info(f"Registered segmenter: {name}")
    
    def register_model(self, name: str, model_loader: Callable, **metadata):
        """
        Register a custom model
        
        Args:
            name: Model name
            model_loader: Function to load the model
            **metadata: Additional metadata
        """
        self._registries["models"][name] = {
            "loader": model_loader,
            "metadata": metadata
        }
        logger.info(f"Registered model: {name}")
    
    def register_processor(self, name: str, processor_class: Type, **metadata):
        """
        Register a custom processor
        
        Args:
            name: Processor name
            processor_class: Processor class
            **metadata: Additional metadata
        """
        self._registries["processors"][name] = {
            "class": processor_class,
            "metadata": metadata
        }
        logger.info(f"Registered processor: {name}")
    
    def register_visualizer(self, name: str, visualizer_class: Type, **metadata):
        """
        Register a custom visualizer
        
        Args:
            name: Visualizer name
            visualizer_class: Visualizer class
            **metadata: Additional metadata
        """
        self._registries["visualizers"][name] = {
            "class": visualizer_class,
            "metadata": metadata
        }
        logger.info(f"Registered visualizer: {name}")
    
    def register_evaluator(self, name: str, evaluator_class: Type, **metadata):
        """
        Register a custom evaluator
        
        Args:
            name: Evaluator name
            evaluator_class: Evaluator class
            **metadata: Additional metadata
        """
        self._registries["evaluators"][name] = {
            "class": evaluator_class,
            "metadata": metadata
        }
        logger.info(f"Registered evaluator: {name}")
    
    def register_custom_component(self, name: str, component: Any, **metadata):
        """
        Register a custom component
        
        Args:
            name: Component name
            component: Component object or class
            **metadata: Additional metadata
        """
        self._registries["custom_components"][name] = {
            "component": component,
            "metadata": metadata
        }
        logger.info(f"Registered custom component: {name}")
    
    def get_detector(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered detector
        
        Args:
            name: Detector name
            
        Returns:
            Optional[Dict[str, Any]]: Detector info if found
        """
        return self._registries["detectors"].get(name)
    
    def get_tracker(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered tracker
        
        Args:
            name: Tracker name
            
        Returns:
            Optional[Dict[str, Any]]: Tracker info if found
        """
        return self._registries["trackers"].get(name)
    
    def get_segmenter(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered segmenter
        
        Args:
            name: Segmenter name
            
        Returns:
            Optional[Dict[str, Any]]: Segmenter info if found
        """
        return self._registries["segmenters"].get(name)
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered model
        
        Args:
            name: Model name
            
        Returns:
            Optional[Dict[str, Any]]: Model info if found
        """
        return self._registries["models"].get(name)
    
    def get_processor(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered processor
        
        Args:
            name: Processor name
            
        Returns:
            Optional[Dict[str, Any]]: Processor info if found
        """
        return self._registries["processors"].get(name)
    
    def get_visualizer(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered visualizer
        
        Args:
            name: Visualizer name
            
        Returns:
            Optional[Dict[str, Any]]: Visualizer info if found
        """
        return self._registries["visualizers"].get(name)
    
    def get_evaluator(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered evaluator
        
        Args:
            name: Evaluator name
            
        Returns:
            Optional[Dict[str, Any]]: Evaluator info if found
        """
        return self._registries["evaluators"].get(name)
    
    def get_custom_component(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered custom component
        
        Args:
            name: Component name
            
        Returns:
            Optional[Dict[str, Any]]: Component info if found
        """
        return self._registries["custom_components"].get(name)
    
    def list_detectors(self) -> List[str]:
        """
        List all registered detectors
        
        Returns:
            List[str]: List of detector names
        """
        return list(self._registries["detectors"].keys())
    
    def list_trackers(self) -> List[str]:
        """
        List all registered trackers
        
        Returns:
            List[str]: List of tracker names
        """
        return list(self._registries["trackers"].keys())
    
    def list_segmenters(self) -> List[str]:
        """
        List all registered segmenters
        
        Returns:
            List[str]: List of segmenter names
        """
        return list(self._registries["segmenters"].keys())
    
    def list_models(self) -> List[str]:
        """
        List all registered models
        
        Returns:
            List[str]: List of model names
        """
        return list(self._registries["models"].keys())
    
    def list_processors(self) -> List[str]:
        """
        List all registered processors
        
        Returns:
            List[str]: List of processor names
        """
        return list(self._registries["processors"].keys())
    
    def list_visualizers(self) -> List[str]:
        """
        List all registered visualizers
        
        Returns:
            List[str]: List of visualizer names
        """
        return list(self._registries["visualizers"].keys())
    
    def list_evaluators(self) -> List[str]:
        """
        List all registered evaluators
        
        Returns:
            List[str]: List of evaluator names
        """
        return list(self._registries["evaluators"].keys())
    
    def list_custom_components(self) -> List[str]:
        """
        List all registered custom components
        
        Returns:
            List[str]: List of component names
        """
        return list(self._registries["custom_components"].keys())
    
    def add_plugin_path(self, path: str):
        """
        Add a plugin path to search for plugins
        
        Args:
            path: Path to plugin directory
        """
        if path not in self._plugin_paths:
            self._plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")
    
    def load_plugins_from_path(self, path: str):
        """
        Load plugins from a directory
        
        Args:
            path: Path to plugin directory
        """
        try:
            plugin_path = Path(path)
            if not plugin_path.exists():
                logger.warning(f"Plugin path does not exist: {path}")
                return
            
            for item in plugin_path.iterdir():
                if item.is_dir() or (item.is_file() and item.suffix == '.py'):
                    self._load_plugin(item)
        except Exception as e:
            logger.error(f"Error loading plugins from path {path}: {e}")
    
    def load_all_plugins(self):
        """
        Load all plugins from registered paths
        """
        for path in self._plugin_paths:
            self.load_plugins_from_path(path)
    
    def _load_plugin(self, plugin_path: Path):
        """
        Load a single plugin
        
        Args:
            plugin_path: Path to plugin file or directory
        """
        try:
            if plugin_path.is_dir():
                # Load plugin from directory
                self._load_plugin_directory(plugin_path)
            elif plugin_path.is_file() and plugin_path.suffix == '.py':
                # Load plugin from file
                self._load_plugin_file(plugin_path)
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {e}")
    
    def _load_plugin_directory(self, plugin_dir: Path):
        """
        Load a plugin from a directory
        
        Args:
            plugin_dir: Path to plugin directory
        """
        init_file = plugin_dir / '__init__.py'
        if init_file.exists():
            # Add directory to Python path
            sys.path.insert(0, str(plugin_dir.parent))
            
            # Import plugin module
            plugin_name = plugin_dir.name
            try:
                plugin_module = importlib.import_module(plugin_name)
                self._register_plugin_components(plugin_module)
            except Exception as e:
                logger.error(f"Error importing plugin module {plugin_name}: {e}")
            finally:
                # Remove directory from Python path
                sys.path.pop(0)
    
    def _load_plugin_file(self, plugin_file: Path):
        """
        Load a plugin from a file
        
        Args:
            plugin_file: Path to plugin file
        """
        # Add directory to Python path
        sys.path.insert(0, str(plugin_file.parent))
        
        # Import plugin module
        plugin_name = plugin_file.stem
        try:
            plugin_module = importlib.import_module(plugin_name)
            self._register_plugin_components(plugin_module)
        except Exception as e:
            logger.error(f"Error importing plugin file {plugin_file}: {e}")
        finally:
            # Remove directory from Python path
            sys.path.pop(0)
    
    def _register_plugin_components(self, plugin_module):
        """
        Register components from a plugin module
        
        Args:
            plugin_module: Plugin module
        """
        # Check for registration functions in the plugin module
        if hasattr(plugin_module, 'register_plugin'):
            try:
                plugin_module.register_plugin(self)
            except Exception as e:
                logger.error(f"Error calling register_plugin: {e}")
        
        # Check for component registration attributes
        if hasattr(plugin_module, 'REGISTER_DETECTORS'):
            for name, detector_info in plugin_module.REGISTER_DETECTORS.items():
                if isinstance(detector_info, dict) and 'class' in detector_info:
                    self.register_detector(name, detector_info['class'], **detector_info.get('metadata', {}))
        
        if hasattr(plugin_module, 'REGISTER_TRACKERS'):
            for name, tracker_info in plugin_module.REGISTER_TRACKERS.items():
                if isinstance(tracker_info, dict) and 'class' in tracker_info:
                    self.register_tracker(name, tracker_info['class'], **tracker_info.get('metadata', {}))
        
        if hasattr(plugin_module, 'REGISTER_MODELS'):
            for name, model_info in plugin_module.REGISTER_MODELS.items():
                if isinstance(model_info, dict) and 'loader' in model_info:
                    self.register_model(name, model_info['loader'], **model_info.get('metadata', {}))


# Global plugin registry instance
plugin_registry = PluginRegistry()


# Convenience functions for plugin registration
def register_detector(name: str, **metadata):
    """
    Decorator to register a detector class
    
    Args:
        name: Detector name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(detector_class):
        plugin_registry.register_detector(name, detector_class, **metadata)
        return detector_class
    return decorator


def register_tracker(name: str, **metadata):
    """
    Decorator to register a tracker class
    
    Args:
        name: Tracker name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(tracker_class):
        plugin_registry.register_tracker(name, tracker_class, **metadata)
        return tracker_class
    return decorator


def register_segmenter(name: str, **metadata):
    """
    Decorator to register a segmenter class
    
    Args:
        name: Segmenter name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(segmenter_class):
        plugin_registry.register_segmenter(name, segmenter_class, **metadata)
        return segmenter_class
    return decorator


def register_model(name: str, **metadata):
    """
    Decorator to register a model loader
    
    Args:
        name: Model name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(model_loader):
        plugin_registry.register_model(name, model_loader, **metadata)
        return model_loader
    return decorator


def register_processor(name: str, **metadata):
    """
    Decorator to register a processor class
    
    Args:
        name: Processor name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(processor_class):
        plugin_registry.register_processor(name, processor_class, **metadata)
        return processor_class
    return decorator


def register_visualizer(name: str, **metadata):
    """
    Decorator to register a visualizer class
    
    Args:
        name: Visualizer name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(visualizer_class):
        plugin_registry.register_visualizer(name, visualizer_class, **metadata)
        return visualizer_class
    return decorator


def register_evaluator(name: str, **metadata):
    """
    Decorator to register an evaluator class
    
    Args:
        name: Evaluator name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(evaluator_class):
        plugin_registry.register_evaluator(name, evaluator_class, **metadata)
        return evaluator_class
    return decorator


def register_custom_component(name: str, **metadata):
    """
    Decorator to register a custom component
    
    Args:
        name: Component name
        **metadata: Additional metadata
        
    Returns:
        Callable: Decorator function
    """
    def decorator(component):
        plugin_registry.register_custom_component(name, component, **metadata)
        return component
    return decorator


# Model registration utilities
class ModelRegistry:
    """
    Model registry for managing model loading and caching
    """
    
    def __init__(self):
        """
        Initialize model registry
        """
        self._models = {}
        self._model_cache = {}
    
    def register_model(self, name: str, model_info: Dict[str, Any]):
        """
        Register a model
        
        Args:
            name: Model name
            model_info: Model information
        """
        self._models[name] = model_info
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get model information
        
        Args:
            name: Model name
            
        Returns:
            Optional[Dict[str, Any]]: Model information if found
        """
        return self._models.get(name)
    
    def load_model(self, name: str, **kwargs) -> Any:
        """
        Load a model
        
        Args:
            name: Model name
            **kwargs: Additional arguments for model loading
            
        Returns:
            Any: Loaded model
        """
        # Check cache first
        cache_key = f"{name}:{kwargs}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Get model info
        model_info = self.get_model(name)
        if not model_info:
            logger.error(f"Model {name} not found")
            return None
        
        # Load model
        try:
            model = model_info['loader'](**kwargs)
            # Cache the model
            self._model_cache[cache_key] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
            return None
    
    def unload_model(self, name: str):
        """
        Unload a model from cache
        
        Args:
            name: Model name
        """
        # Remove from cache
        keys_to_remove = []
        for key in self._model_cache:
            if key.startswith(f"{name}:"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._model_cache[key]
        
        logger.info(f"Unloaded model: {name}")
    
    def list_models(self) -> List[str]:
        """
        List all registered models
        
        Returns:
            List[str]: List of model names
        """
        return list(self._models.keys())
    
    def clear_cache(self):
        """
        Clear model cache
        """
        self._model_cache.clear()
        logger.info("Cleared model cache")


# Global model registry instance
model_registry = ModelRegistry()


# Utility functions for plugin system
def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry
    
    Returns:
        PluginRegistry: Global plugin registry instance
    """
    return plugin_registry


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry
    
    Returns:
        ModelRegistry: Global model registry instance
    """
    return model_registry


def discover_plugins():
    """
    Discover and load plugins from default locations
    """
    # Add default plugin paths
    default_paths = [
        Path.home() / '.visionframework' / 'plugins',
        Path.cwd() / 'plugins'
    ]
    
    for path in default_paths:
        if path.exists():
            plugin_registry.add_plugin_path(str(path))
    
    # Load all plugins
    plugin_registry.load_all_plugins()


# Initialize plugin system
discover_plugins()
