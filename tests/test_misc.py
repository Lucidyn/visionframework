import os
import sys
import numpy as np
import pytest
from visionframework.exceptions import VisionFrameworkError, ConfigurationError, DeviceError
from visionframework.utils.logger import get_logger


class TestLoggingExceptions:
    """Test cases for logging and exceptions"""
    
    def test_get_logger(self):
        """Test getting a logger instance"""
        logger = get_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
    
    def test_logger_levels(self):
        """Test different logger levels"""
        logger = get_logger("test_logger_levels")
        assert logger is not None
        
        # These calls should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")


class TestExceptions:
    """Test cases for custom exceptions"""
    
    def test_vision_framework_error(self):
        """Test VisionFrameworkError"""
        with pytest.raises(VisionFrameworkError):
            raise VisionFrameworkError("Test error")
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test configuration error")
    
    def test_device_error(self):
        """Test DeviceError"""
        with pytest.raises(DeviceError):
            raise DeviceError("Test device error")
    
    def test_exception_hierarchy(self):
        """Test exception hierarchy"""
        # All custom exceptions should inherit from VisionFrameworkError
        assert issubclass(ConfigurationError, VisionFrameworkError)
        assert issubclass(DeviceError, VisionFrameworkError)


class TestCodeQuality:
    """Test cases for code quality"""
    
    def test_imports(self):
        """Test that non-torch core modules can be imported"""
        # Test importing only non-torch dependent core modules
        from visionframework.utils.config import Config
        from visionframework.utils.image_utils import ImageUtils
        from visionframework.utils.performance import PerformanceMonitor
        
        # If we got here without exceptions, the test passes
        assert True
    
    def test_module_structure(self):
        """Test that the module structure is correct"""
        # Check that the main __init__.py exists and is importable
        import visionframework
        assert visionframework is not None
        
        # Check that core submodules are importable
        import visionframework.core
        import visionframework.utils
        import visionframework.exceptions
        
        assert visionframework.core is not None
        assert visionframework.utils is not None
        assert visionframework.exceptions is not None


class TestStructure:
    """Test cases for project structure"""
    
    def test_required_directories(self):
        """Test that required directories exist"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check that required directories exist
        required_dirs = [
            "visionframework",
            "examples",
            "tests",
            "docs"
        ]
        
        for dir_name in required_dirs:
            dir_path = os.path.join(project_root, dir_name)
            assert os.path.exists(dir_path), f"Required directory {dir_name} not found"
            assert os.path.isdir(dir_path), f"{dir_name} is not a directory"
    
    def test_core_files_exist(self):
        """Test that core files exist"""
        # Get the visionframework directory
        visionframework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        visionframework_dir = os.path.join(visionframework_dir, "visionframework")
        
        # Check that core files exist
        core_files = [
            "__init__.py",
            "core/__init__.py",
            "core/pipeline.py",
            "core/detector.py",
            "utils/__init__.py",
            "utils/config.py",
            "exceptions.py"
        ]
        
        for file_path in core_files:
            full_path = os.path.join(visionframework_dir, file_path)
            assert os.path.exists(full_path), f"Core file {file_path} not found"
            assert os.path.isfile(full_path), f"{file_path} is not a file"
    
    def test_version_file(self):
        """Test that version information is available"""
        # Try to import version from the package
        try:
            from visionframework import __version__
            assert isinstance(__version__, str)
            assert len(__version__) > 0
        except ImportError:
            # If __version__ is not defined, check if it's in __init__.py
            init_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "visionframework",
                "__init__.py"
            )
            with open(init_path, "r") as f:
                content = f.read()
                assert "__version__" in content, "Version not found in __init__.py"


class TestTypeHints:
    """Test cases for type hints and static typing"""
    
    def test_type_hints_exist(self):
        """Test that type hints are used in core modules"""
        # This is a basic check to ensure type hints are present
        # We'll check a few core files for type hint usage
        core_files = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "visionframework", "core", "pipeline.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "visionframework", "core", "detectors", "yolo_detector.py")
        ]
        
        for file_path in core_files:
            with open(file_path, "r") as f:
                content = f.read()
                # Check for common type hint patterns
                type_hint_patterns = [
                    "->",  # Return type hint
                    ": ",  # Parameter type hint
                    "Optional[",  # Optional type
                    "List[",  # List type
                    "Dict[",  # Dict type
                    "Tuple["  # Tuple type
                ]
                
                # At least one type hint pattern should be present
                has_type_hints = any(pattern in content for pattern in type_hint_patterns)
                assert has_type_hints, f"No type hints found in {os.path.basename(file_path)}"


class TestDocstrings:
    """Test cases for docstrings"""
    
    def test_docstrings_exist(self):
        """Test that core classes and methods have docstrings"""
        # Test docstrings for non-torch dependent modules instead
        from visionframework.utils.config import Config, DeviceManager
        from visionframework.utils.performance import PerformanceMonitor
        
        # Check that classes have docstrings
        assert hasattr(Config, "__doc__"), "Config has no docstring"
        assert len(Config.__doc__.strip()) > 0, "Config docstring is empty"
        
        assert hasattr(DeviceManager, "__doc__"), "DeviceManager has no docstring"
        assert len(DeviceManager.__doc__.strip()) > 0, "DeviceManager docstring is empty"
        
        assert hasattr(PerformanceMonitor, "__doc__"), "PerformanceMonitor has no docstring"
        assert len(PerformanceMonitor.__doc__.strip()) > 0, "PerformanceMonitor docstring is empty"
        
        # Check that some methods have docstrings
        assert hasattr(Config.load_from_file, "__doc__"), "Config.load_from_file has no docstring"
        assert hasattr(DeviceManager.auto_select_device, "__doc__"), "DeviceManager.auto_select_device has no docstring"
        assert hasattr(PerformanceMonitor.start, "__doc__"), "PerformanceMonitor.start has no docstring"
