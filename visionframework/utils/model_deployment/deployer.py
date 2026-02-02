"""
Model deployment core functionality

This module provides the core functionality for deploying models to different platforms.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .platforms import (
    DeploymentPlatform,
    get_platform_compatibility,
    get_platform_requirements
)
from ..model_conversion import (
    get_converted_model,
    ModelFormat
)


@dataclass
class DeploymentConfig:
    """
    Configuration for model deployment
    
    Attributes:
        platform: Deployment platform
        model_path: Path to model
        model_format: Model format
        output_path: Path to save deployed model
        config: Additional deployment configuration
        verbose: Whether to print verbose information
        optimize: Whether to optimize the model for deployment
        test: Whether to test the deployment
    """
    platform: DeploymentPlatform
    model_path: str
    model_format: Union[str, ModelFormat]
    output_path: str
    config: Optional[Dict[str, Any]] = None
    verbose: bool = False
    optimize: bool = True
    test: bool = True


class ModelDeployer:
    """
    Model deployer class for deploying models to different platforms
    """
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize model deployer
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        if self.config.config is None:
            self.config.config = {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate deployment configuration
        """
        # Check if input file exists
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        # Check if output directory exists
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Check platform compatibility
        compatible_formats = get_platform_compatibility(self.config.platform)
        model_format_str = self.config.model_format.value if isinstance(self.config.model_format, ModelFormat) else self.config.model_format
        if model_format_str not in compatible_formats:
            raise ValueError(
                f"Model format {model_format_str} is not compatible with {self.config.platform.value}. "
                f"Compatible formats: {compatible_formats}"
            )
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed
        """
        requirements = get_platform_requirements(self.config.platform)
        missing_deps = []
        
        for dep in requirements.get("dependencies", []):
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            raise ImportError(
                f"Missing dependencies for {self.config.platform.value}: {missing_deps}. "
                f"Install them with pip install {' '.join(missing_deps)}"
            )
    
    def deploy(self) -> str:
        """
        Deploy model to the specified platform
        
        Returns:
            Path to deployed model
        """
        if self.config.verbose:
            print(f"Deploying model to {self.config.platform.value}")
            print(f"Model: {self.config.model_path}")
            print(f"Format: {self.config.model_format}")
            print(f"Output: {self.config.output_path}")
        
        # Dispatch to specific deployment method
        deployment_method = self._get_deployment_method()
        deployed_path = deployment_method()
        
        # Test deployment if requested
        if self.config.test:
            if self._test_deployment(deployed_path):
                if self.config.verbose:
                    print("Deployment test passed")
            else:
                if self.config.verbose:
                    print("Deployment test failed")
        
        if self.config.verbose:
            print(f"Deployment completed successfully")
            print(f"Deployed model saved to: {deployed_path}")
        
        return deployed_path
    
    def _get_deployment_method(self) -> callable:
        """
        Get appropriate deployment method based on platform
        """
        deployment_map = {
            DeploymentPlatform.LOCAL: self._deploy_local,
            DeploymentPlatform.DOCKER: self._deploy_docker,
            DeploymentPlatform.AWS_SAGEMAKER: self._deploy_aws_sagemaker,
            DeploymentPlatform.AZURE_ML: self._deploy_azure_ml,
            DeploymentPlatform.GCP_AI_PLATFORM: self._deploy_gcp_ai_platform,
            DeploymentPlatform.NVIDIA_JETSON: self._deploy_nvidia_jetson,
            DeploymentPlatform.RASPBERRY_PI: self._deploy_raspberry_pi,
            DeploymentPlatform.EDGE_DEVICE: self._deploy_edge_device,
            DeploymentPlatform.WEB: self._deploy_web,
            DeploymentPlatform.WEB_ASSEMBLY: self._deploy_web_assembly,
            DeploymentPlatform.ANDROID: self._deploy_android,
            DeploymentPlatform.IOS: self._deploy_ios,
            DeploymentPlatform.EMBEDDED: self._deploy_embedded,
            DeploymentPlatform.MICROCONTROLLER: self._deploy_microcontroller
        }
        
        if self.config.platform not in deployment_map:
            raise ValueError(f"No deployment method found for {self.config.platform.value}")
        
        return deployment_map[self.config.platform]
    
    def _deploy_local(self) -> str:
        """
        Deploy model locally
        """
        if self.config.verbose:
            print("Deploying model locally")
        
        # For local deployment, just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        # Create deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "model_path": self.config.output_path,
            "model_format": self.config.model_format,
            "deployed_at": os.path.getmtime(self.config.output_path),
            "config": self.config.config
        }
        
        # Save deployment config
        config_path = self.config.output_path + ".json"
        with open(config_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return self.config.output_path
    
    def _deploy_docker(self) -> str:
        """
        Deploy model to Docker
        """
        if self.config.verbose:
            print("Deploying model to Docker")
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY {os.path.basename(self.config.model_path)} .
COPY app.py .

CMD ["python", "app.py"]
"""
        
        # Create requirements.txt
        requirements_content = """
torch
onnxruntime
flask
"""
        
        # Create app.py
        app_content = f"""
import torch
import onnxruntime
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model
model_path = "{os.path.basename(self.config.model_path)}"

@app.route('/predict', methods=['POST'])
def predict():
    # Process request
    data = request.json
    input_data = np.array(data['input'])
    
    # Run inference
    # Implement inference based on model format
    
    return jsonify({'output': []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
        
        # Save files
        dockerfile_path = os.path.join(os.path.dirname(self.config.output_path), "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        requirements_path = os.path.join(os.path.dirname(self.config.output_path), "requirements.txt")
        with open(requirements_path, "w") as f:
            f.write(requirements_content)
        
        app_path = os.path.join(os.path.dirname(self.config.output_path), "app.py")
        with open(app_path, "w") as f:
            f.write(app_content)
        
        # Copy model
        import shutil
        model_dest = os.path.join(os.path.dirname(self.config.output_path), os.path.basename(self.config.model_path))
        shutil.copy2(self.config.model_path, model_dest)
        
        # Create deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "dockerfile": dockerfile_path,
            "model_path": model_dest,
            "deployed_at": os.path.getmtime(dockerfile_path),
            "config": self.config.config
        }
        
        # Save deployment config
        config_path = self.config.output_path
        with open(config_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return config_path
    
    def _deploy_aws_sagemaker(self) -> str:
        """
        Deploy model to AWS SageMaker
        """
        if self.config.verbose:
            print("Deploying model to AWS SageMaker")
        
        # Create SageMaker deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "model_path": self.config.model_path,
            "model_format": self.config.model_format,
            "deployed_at": os.path.getmtime(self.config.model_path),
            "config": self.config.config
        }
        
        # Save deployment config
        with open(self.config.output_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return self.config.output_path
    
    def _deploy_azure_ml(self) -> str:
        """
        Deploy model to Azure ML
        """
        if self.config.verbose:
            print("Deploying model to Azure ML")
        
        # Create Azure ML deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "model_path": self.config.model_path,
            "model_format": self.config.model_format,
            "deployed_at": os.path.getmtime(self.config.model_path),
            "config": self.config.config
        }
        
        # Save deployment config
        with open(self.config.output_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return self.config.output_path
    
    def _deploy_gcp_ai_platform(self) -> str:
        """
        Deploy model to GCP AI Platform
        """
        if self.config.verbose:
            print("Deploying model to GCP AI Platform")
        
        # Create GCP AI Platform deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "model_path": self.config.model_path,
            "model_format": self.config.model_format,
            "deployed_at": os.path.getmtime(self.config.model_path),
            "config": self.config.config
        }
        
        # Save deployment config
        with open(self.config.output_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return self.config.output_path
    
    def _deploy_nvidia_jetson(self) -> str:
        """
        Deploy model to NVIDIA Jetson
        """
        if self.config.verbose:
            print("Deploying model to NVIDIA Jetson")
        
        # For Jetson, we should convert to TensorRT if possible
        if self.config.optimize:
            from ..model_conversion import get_converted_model, ModelFormat
            
            # Convert to TensorRT if not already
            if isinstance(self.config.model_format, str):
                model_format = ModelFormat(self.config.model_format)
            else:
                model_format = self.config.model_format
            
            if model_format != ModelFormat.TENSORRT:
                trt_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".engine")
                get_converted_model(
                    self.config.model_path,
                    trt_path,
                    model_format,
                    ModelFormat.TENSORRT
                )
                return trt_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_raspberry_pi(self) -> str:
        """
        Deploy model to Raspberry Pi
        """
        if self.config.verbose:
            print("Deploying model to Raspberry Pi")
        
        # For Raspberry Pi, we should convert to TFLite if possible
        if self.config.optimize:
            from ..model_conversion import get_converted_model, ModelFormat
            
            # Convert to TFLite if not already
            if isinstance(self.config.model_format, str):
                model_format = ModelFormat(self.config.model_format)
            else:
                model_format = self.config.model_format
            
            if model_format != ModelFormat.TFLITE:
                tflite_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".tflite")
                get_converted_model(
                    self.config.model_path,
                    tflite_path,
                    model_format,
                    ModelFormat.TFLITE
                )
                return tflite_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_edge_device(self) -> str:
        """
        Deploy model to edge device
        """
        if self.config.verbose:
            print("Deploying model to edge device")
        
        # For edge device, optimize the model
        if self.config.optimize:
            from ..model_conversion import get_converted_model, ModelFormat
            from ..model_optimization import quantize_model, QuantizationConfig
            import torch
            
            # Load model
            model = torch.load(self.config.model_path)
            
            # Quantize model
            quant_config = QuantizationConfig(
                quantization_type="dynamic",
                backend="qnnpack",
                verbose=self.config.verbose
            )
            quantized_model = quantize_model(model, quant_config)
            
            # Save quantized model
            torch.save(quantized_model, self.config.output_path)
            return self.config.output_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_web(self) -> str:
        """
        Deploy model to web
        """
        if self.config.verbose:
            print("Deploying model to web")
        
        # For web deployment, we need to convert to ONNX
        from ..model_conversion import get_converted_model, ModelFormat
        
        # Convert to ONNX if not already
        if isinstance(self.config.model_format, str):
            model_format = ModelFormat(self.config.model_format)
        else:
            model_format = self.config.model_format
        
        if model_format != ModelFormat.ONNX:
            onnx_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".onnx")
            get_converted_model(
                self.config.model_path,
                onnx_path,
                model_format,
                ModelFormat.ONNX
            )
            return onnx_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_web_assembly(self) -> str:
        """
        Deploy model to WebAssembly
        """
        if self.config.verbose:
            print("Deploying model to WebAssembly")
        
        # For WebAssembly, we need to convert to ONNX first
        from ..model_conversion import get_converted_model, ModelFormat
        
        # Convert to ONNX if not already
        if isinstance(self.config.model_format, str):
            model_format = ModelFormat(self.config.model_format)
        else:
            model_format = self.config.model_format
        
        if model_format != ModelFormat.ONNX:
            onnx_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".onnx")
            get_converted_model(
                self.config.model_path,
                onnx_path,
                model_format,
                ModelFormat.ONNX
            )
        else:
            onnx_path = self.config.model_path
        
        # Create WebAssembly deployment config
        deployment_config = {
            "platform": self.config.platform.value,
            "onnx_model": onnx_path,
            "deployed_at": os.path.getmtime(onnx_path),
            "config": self.config.config
        }
        
        # Save deployment config
        with open(self.config.output_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        return self.config.output_path
    
    def _deploy_android(self) -> str:
        """
        Deploy model to Android
        """
        if self.config.verbose:
            print("Deploying model to Android")
        
        # For Android, we should convert to TFLite if possible
        from ..model_conversion import get_converted_model, ModelFormat
        
        # Convert to TFLite if not already
        if isinstance(self.config.model_format, str):
            model_format = ModelFormat(self.config.model_format)
        else:
            model_format = self.config.model_format
        
        if model_format != ModelFormat.TFLITE:
            tflite_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".tflite")
            get_converted_model(
                self.config.model_path,
                tflite_path,
                model_format,
                ModelFormat.TFLITE
            )
            return tflite_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_ios(self) -> str:
        """
        Deploy model to iOS
        """
        if self.config.verbose:
            print("Deploying model to iOS")
        
        # For iOS, we should convert to CoreML if possible
        from ..model_conversion import get_converted_model, ModelFormat
        
        # Convert to CoreML if not already
        if isinstance(self.config.model_format, str):
            model_format = ModelFormat(self.config.model_format)
        else:
            model_format = self.config.model_format
        
        if model_format != ModelFormat.COREML:
            coreml_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".mlmodel")
            get_converted_model(
                self.config.model_path,
                coreml_path,
                model_format,
                ModelFormat.COREML
            )
            return coreml_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_embedded(self) -> str:
        """
        Deploy model to embedded device
        """
        if self.config.verbose:
            print("Deploying model to embedded device")
        
        # For embedded devices, we should optimize the model
        if self.config.optimize:
            from ..model_optimization import quantize_model, prune_model, QuantizationConfig, PruningConfig
            import torch
            
            # Load model
            model = torch.load(self.config.model_path)
            
            # Quantize model
            quant_config = QuantizationConfig(
                quantization_type="dynamic",
                backend="qnnpack",
                verbose=self.config.verbose
            )
            quantized_model = quantize_model(model, quant_config)
            
            # Prune model
            prune_config = PruningConfig(
                pruning_type="l1_unstructured",
                amount=0.3,
                verbose=self.config.verbose
            )
            pruned_model = prune_model(quantized_model, prune_config)
            
            # Save optimized model
            torch.save(pruned_model, self.config.output_path)
            return self.config.output_path
        
        # Otherwise just copy the model
        import shutil
        shutil.copy2(self.config.model_path, self.config.output_path)
        
        return self.config.output_path
    
    def _deploy_microcontroller(self) -> str:
        """
        Deploy model to microcontroller
        """
        if self.config.verbose:
            print("Deploying model to microcontroller")
        
        # For microcontrollers, we need to convert to TFLite Micro
        from ..model_conversion import get_converted_model, ModelFormat
        
        # Convert to TFLite first if not already
        if isinstance(self.config.model_format, str):
            model_format = ModelFormat(self.config.model_format)
        else:
            model_format = self.config.model_format
        
        if model_format != ModelFormat.TFLITE:
            tflite_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".tflite")
            get_converted_model(
                self.config.model_path,
                tflite_path,
                model_format,
                ModelFormat.TFLITE
            )
        else:
            tflite_path = self.config.model_path
        
        # Create C++ header file for TFLite Micro
        # This is a simplified version
        header_content = f"""
// Auto-generated model header for microcontroller deployment

#ifndef MODEL_HEADER
#define MODEL_HEADER

#include <cstdint>

constexpr int model_size = {os.path.getsize(tflite_path)};
constexpr uint8_t model_data[{os.path.getsize(tflite_path)}] = {{
    // Model data will be added here
}};

#endif // MODEL_HEADER
"""
        
        header_path = self.config.output_path.replace(os.path.splitext(self.config.output_path)[1], ".h")
        with open(header_path, "w") as f:
            f.write(header_content)
        
        return header_path
    
    def _test_deployment(self, deployed_path: str) -> bool:
        """
        Test deployment
        
        Args:
            deployed_path: Path to deployed model
        
        Returns:
            True if deployment test passed, False otherwise
        """
        if self.config.verbose:
            print(f"Testing deployment: {deployed_path}")
        
        # For most platforms, just check if the file exists
        if not os.path.exists(deployed_path):
            return False
        
        # For some platforms, we can do more thorough testing
        if self.config.platform == DeploymentPlatform.LOCAL:
            # Try to load the model
            try:
                if isinstance(self.config.model_format, str):
                    model_format = self.config.model_format
                else:
                    model_format = self.config.model_format.value
                
                if model_format == "pytorch":
                    import torch
                    model = torch.load(deployed_path)
                elif model_format == "onnx":
                    import onnx
                    model = onnx.load(deployed_path)
                elif model_format == "tflite":
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=deployed_path)
                    interpreter.allocate_tensors()
                return True
            except Exception as e:
                if self.config.verbose:
                    print(f"Deployment test failed: {e}")
                return False
        
        return True


def deploy_model(config: DeploymentConfig) -> str:
    """
    Deploy model using the given configuration
    
    Args:
        config: Deployment configuration
    
    Returns:
        Path to deployed model
    """
    deployer = ModelDeployer(config)
    return deployer.deploy()


def get_deployed_model(
    model_path: str,
    platform: Union[str, DeploymentPlatform],
    output_path: str,
    **kwargs
) -> str:
    """
    Deploy model to the specified platform
    
    Args:
        model_path: Path to model
        platform: Deployment platform
        output_path: Path to save deployed model
        **kwargs: Additional deployment options
    
    Returns:
        Path to deployed model
    """
    # Convert platform string to DeploymentPlatform enum
    if isinstance(platform, str):
        platform = DeploymentPlatform(platform)
    
    # Create deployment config
    config = DeploymentConfig(
        platform=platform,
        model_path=model_path,
        model_format=kwargs.pop("model_format", "pytorch"),
        output_path=output_path,
        config=kwargs.pop("config", None),
        verbose=kwargs.pop("verbose", False),
        optimize=kwargs.pop("optimize", True),
        test=kwargs.pop("test", True),
        **kwargs
    )
    
    # Deploy model
    return deploy_model(config)


def validate_deployment(
    deployed_path: str,
    platform: Union[str, DeploymentPlatform]
) -> bool:
    """
    Validate deployment
    
    Args:
        deployed_path: Path to deployed model
        platform: Deployment platform
    
    Returns:
        True if deployment is valid, False otherwise
    """
    # Convert platform string to DeploymentPlatform enum
    if isinstance(platform, str):
        platform = DeploymentPlatform(platform)
    
    # Check if file exists
    if not os.path.exists(deployed_path):
        return False
    
    # For some platforms, we can do more thorough testing
    if platform == DeploymentPlatform.LOCAL:
        # Try to load the model
        try:
            import torch
            model = torch.load(deployed_path)
            return True
        except Exception:
            try:
                import onnx
                model = onnx.load(deployed_path)
                return True
            except Exception:
                try:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=deployed_path)
                    interpreter.allocate_tensors()
                    return True
                except Exception:
                    return False
    
    return True
