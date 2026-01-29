"""
Dependency management utilities for Vision Framework

This module provides utilities for managing optional dependencies across the Vision Framework.
It allows for lazy loading of heavy dependencies and provides a consistent way to check for
optional dependency availability.
"""

from typing import Dict, Optional, Set, List, Tuple, Union, Any, TypeVar
import importlib
import sys
from types import ModuleType
from .monitoring.logger import get_logger

# Type alias for module
tModule = TypeVar('tModule', bound=ModuleType)

logger = get_logger(__name__)


class DependencyManager:
    """
    Utility class for managing optional dependencies
    """

    # Map of optional dependencies and their package names
    OPTIONAL_DEPENDENCIES = {
        "clip": {
            "packages": ["transformers"],
            "minimum_version": "4.30.0",
            "description": "CLIP model support for zero-shot detection"
        },
        "sam": {
            "packages": ["segment_anything"],
            "minimum_version": "1.0",
            "description": "Segment Anything Model (SAM) support for segmentation"
        },
        "rfdetr": {
            "packages": ["rfdetr", "supervision"],
            "minimum_version": "0.1.0",
            "description": "RF-DETR model support for object detection"
        },
        "pyav": {
            "packages": ["av"],
            "minimum_version": "11.0.0",
            "description": "PyAV support for faster video processing"
        },
        "dev": {
            "packages": ["pytest", "pytest-cov", "black", "flake8", "mypy"],
            "minimum_version": "7.0.0",
            "description": "Development dependencies"
        }
    }

    def __init__(self):
        """
        Initialize dependency manager
        """
        self._available_dependencies: Set[str] = set()
        self._dependency_status: Dict[str, Dict[str, Union[bool, str]]] = {}
        # 延迟检查依赖项，在需要时才检查

    def _check_dependencies(self):
        """
        Check availability of optional dependencies
        """
        for dep_name, dep_info in self.OPTIONAL_DEPENDENCIES.items():
            if dep_name not in self._dependency_status:
                available, message = self._is_dependency_available(dep_name)
                self._dependency_status[dep_name] = {
                    "available": available,
                    "message": message
                }
                if available:
                    self._available_dependencies.add(dep_name)

    def _is_dependency_available(self, dependency: str) -> Tuple[bool, str]:
        """
        Check if a specific dependency is available

        Args:
            dependency: Name of the dependency

        Returns:
            Tuple[bool, str]: (is_available, message)
        """
        dep_info = self.OPTIONAL_DEPENDENCIES.get(dependency)
        if not dep_info:
            return False, f"Unknown dependency: {dependency}"

        for package in dep_info["packages"]:
            try:
                importlib.import_module(package)
            except ImportError:
                return False, f"Missing package: {package}"

        return True, f"All packages for {dependency} are available"

    def is_available(self, dependency: str) -> bool:
        """
        Check if a specific dependency is available

        Args:
            dependency: Name of the dependency

        Returns:
            bool: True if dependency is available, False otherwise
        """
        if dependency not in self._dependency_status:
            # 延迟检查该依赖项
            available, message = self._is_dependency_available(dependency)
            self._dependency_status[dependency] = {
                "available": available,
                "message": message
            }
            if available:
                self._available_dependencies.add(dependency)
        
        status = self._dependency_status.get(dependency, {"available": False})
        return status["available"]

    def get_available_dependencies(self) -> List[str]:
        """
        Get list of available optional dependencies

        Returns:
            List[str]: List of available dependency names
        """
        # 检查所有依赖项状态
        self._check_dependencies()
        return list(self._available_dependencies)

    def get_dependency_status(self, dependency: str) -> Dict[str, Union[bool, str]]:
        """
        Get status of a specific dependency

        Args:
            dependency: Name of the dependency

        Returns:
            Dict[str, Union[bool, str]]: Status information
        """
        if dependency not in self._dependency_status:
            # 延迟检查该依赖项
            available, message = self._is_dependency_available(dependency)
            self._dependency_status[dependency] = {
                "available": available,
                "message": message
            }
            if available:
                self._available_dependencies.add(dependency)
        
        return self._dependency_status.get(dependency, {
            "available": False,
            "message": f"Unknown dependency: {dependency}"
        })

    def get_all_dependency_status(self) -> Dict[str, Dict[str, Union[bool, str]]]:
        """
        Get status of all optional dependencies

        Returns:
            Dict[str, Dict[str, Union[bool, str]]]: Status information for all dependencies
        """
        # 检查所有依赖项状态
        self._check_dependencies()
        return self._dependency_status

    def get_dependency_info(self, dependency: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dependency

        Args:
            dependency: Name of the dependency

        Returns:
            Optional[Dict[str, Any]]: Dependency information if found, None otherwise
        """
        return self.OPTIONAL_DEPENDENCIES.get(dependency)

    def import_dependency(self, dependency: str, package: str) -> Optional[ModuleType]:
        """
        Import a package associated with a dependency
        
        Args:
            dependency: Name of the dependency
            package: Name of the package to import
            
        Returns:
            Optional[ModuleType]: Imported module if successful, None otherwise
        """
        if not self.is_available(dependency):
            logger.warning(f"Dependency {dependency} is not available, cannot import {package}")
            return None

        try:
            return importlib.import_module(package)
        except ImportError as e:
            logger.error(f"Failed to import {package}: {e}")
            return None

    def get_install_command(self, dependency: str) -> Optional[str]:
        """
        Get pip install command for a dependency

        Args:
            dependency: Name of the dependency

        Returns:
            Optional[str]: pip install command if dependency is known, None otherwise
        """
        dep_info = self.OPTIONAL_DEPENDENCIES.get(dependency)
        if not dep_info:
            return None

        packages = dep_info["packages"]
        if not packages:
            return None

        # Build install command
        install_packages = []
        for package in packages:
            if "minimum_version" in dep_info:
                install_packages.append(f"{package}>={dep_info['minimum_version']}")
            else:
                install_packages.append(package)

        return f"pip install {' '.join(install_packages)}"

    def get_missing_dependencies(self) -> List[str]:
        """
        Get list of missing optional dependencies

        Returns:
            List[str]: List of missing dependency names
        """
        # 检查所有依赖项状态
        self._check_dependencies()
        missing = []
        for dep_name in self.OPTIONAL_DEPENDENCIES:
            if not self.is_available(dep_name):
                missing.append(dep_name)
        return missing

    def validate_dependency(self, dependency: str) -> bool:
        """
        Validate that a dependency is available, and log a helpful error message if not

        Args:
            dependency: Name of the dependency

        Returns:
            bool: True if dependency is available, False otherwise
        """
        if self.is_available(dependency):
            return True

        status = self.get_dependency_status(dependency)
        dep_info = self.get_dependency_info(dependency)
        
        error_message = f"Dependency '{dependency}' is not available. {status.get('message')}"
        if dep_info:
            error_message += f"\nDescription: {dep_info.get('description', 'No description available')}"
            install_command = self.get_install_command(dependency)
            if install_command:
                error_message += f"\nTo install: {install_command}"

        logger.error(error_message)
        return False


# Global dependency manager instance
dependency_manager = DependencyManager()


# Helper functions for common dependency operations
def is_dependency_available(dependency: str) -> bool:
    """
    Check if a specific dependency is available

    Args:
        dependency: Name of the dependency

    Returns:
        bool: True if dependency is available, False otherwise
    """
    return dependency_manager.is_available(dependency)


def get_available_dependencies() -> List[str]:
    """
    Get list of available optional dependencies

    Returns:
        List[str]: List of available dependency names
    """
    return dependency_manager.get_available_dependencies()


def get_missing_dependencies() -> List[str]:
    """
    Get list of missing optional dependencies

    Returns:
        List[str]: List of missing dependency names
    """
    return dependency_manager.get_missing_dependencies()


def validate_dependency(dependency: str) -> bool:
    """
    Validate that a dependency is available, and log a helpful error message if not

    Args:
        dependency: Name of the dependency

    Returns:
        bool: True if dependency is available, False otherwise
    """
    return dependency_manager.validate_dependency(dependency)


def get_install_command(dependency: str) -> Optional[str]:
    """
    Get pip install command for a dependency

    Args:
        dependency: Name of the dependency

    Returns:
        Optional[str]: pip install command if dependency is known, None otherwise
    """
    return dependency_manager.get_install_command(dependency)


def import_optional_dependency(dependency: str, package: str) -> Optional[ModuleType]:
    """
    Import an optional dependency package
    
    Args:
        dependency: Name of the dependency
        package: Name of the package to import
        
    Returns:
        Optional[ModuleType]: Imported module if successful, None otherwise
    """
    return dependency_manager.import_dependency(dependency, package)


# Lazy import decorator
def lazy_import(dependency: str, package: str):
    """
    Decorator for lazy importing optional dependencies

    Args:
        dependency: Name of the dependency
        package: Name of the package to import

    Returns:
        callable: Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_dependency_available(dependency):
                logger.error(f"Required dependency '{dependency}' is not available")
                return None
            
            module = import_optional_dependency(dependency, package)
            if module is None:
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
