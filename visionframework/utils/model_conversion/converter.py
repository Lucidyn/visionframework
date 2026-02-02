"""
Model conversion core functionality

This module provides the core functionality for converting models between different formats.
"""

import torch
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .formats import (
    ModelFormat,
    get_compatible_formats,
    get_format_extension,
    get_format_dependencies
)


@dataclass
class ConversionConfig:
    """
    Configuration for model conversion
    
    Attributes:
        input_format: Input model format
        output_format: Output model format
        input_path: Path to input model
        output_path: Path to save output model
        example_input: Example input tensor for tracing
        opset_version: ONNX opset version
        verbose: Whether to print verbose information
        optimize: Whether to optimize the converted model
        device: Device to use for conversion
    """
    input_format: ModelFormat
    output_format: ModelFormat
    input_path: str
    output_path: str
    example_input: Optional[torch.Tensor] = None
    opset_version: int = 13
    verbose: bool = False
    optimize: bool = True
    device: str = "cpu"


class ModelConverter:
    """
    Model converter class for converting between different model formats
    """
    
    def __init__(self, config: ConversionConfig):
        """
        Initialize model converter
        
        Args:
            config: Conversion configuration
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate conversion configuration
        """
        # Check if input format is compatible with output format
        compatible_formats = get_compatible_formats(self.config.input_format)
        if self.config.output_format not in compatible_formats:
            raise ValueError(
                f"Cannot convert from {self.config.input_format.value} to {self.config.output_format.value}. "
                f"Compatible formats: {[f.value for f in compatible_formats]}"
            )
        
        # Check if input file exists
        if not os.path.exists(self.config.input_path):
            raise FileNotFoundError(f"Input model file not found: {self.config.input_path}")
        
        # Check if output directory exists
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed
        """
        required_deps = get_format_dependencies(self.config.output_format)
        missing_deps = []
        
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            raise ImportError(
                f"Missing dependencies for {self.config.output_format.value}: {missing_deps}. "
                f"Install them with pip install {' '.join(missing_deps)}"
            )
    
    def convert(self) -> str:
        """
        Convert model from input format to output format
        
        Returns:
            Path to converted model
        """
        if self.config.verbose:
            print(f"Converting model from {self.config.input_format.value} to {self.config.output_format.value}")
            print(f"Input: {self.config.input_path}")
            print(f"Output: {self.config.output_path}")
        
        # Dispatch to specific conversion method
        conversion_method = self._get_conversion_method()
        converted_path = conversion_method()
        
        if self.config.verbose:
            print(f"Conversion completed successfully")
            print(f"Converted model saved to: {converted_path}")
        
        return converted_path
    
    def _get_conversion_method(self) -> callable:
        """
        Get appropriate conversion method based on input and output formats
        """
        conversion_map = {
            (ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT): self._convert_pytorch_to_torchscript,
            (ModelFormat.PYTORCH, ModelFormat.ONNX): self._convert_pytorch_to_onnx,
            (ModelFormat.PYTORCH, ModelFormat.TFLITE): self._convert_pytorch_to_tflite,
            (ModelFormat.PYTORCH, ModelFormat.OPENVINO): self._convert_pytorch_to_openvino,
            (ModelFormat.PYTORCH, ModelFormat.COREML): self._convert_pytorch_to_coreml,
            (ModelFormat.TORCHSCRIPT, ModelFormat.ONNX): self._convert_torchscript_to_onnx,
            (ModelFormat.ONNX, ModelFormat.TENSORRT): self._convert_onnx_to_tensorrt,
            (ModelFormat.ONNX, ModelFormat.TENSORRT_ENGINE): self._convert_onnx_to_tensorrt_engine,
            (ModelFormat.ONNX, ModelFormat.OPENVINO): self._convert_onnx_to_openvino,
            (ModelFormat.ONNX, ModelFormat.ONNX_RUNTIME): self._convert_onnx_to_onnx_runtime,
            (ModelFormat.TENSORFLOW, ModelFormat.TFLITE): self._convert_tensorflow_to_tflite,
            (ModelFormat.TENSORFLOW, ModelFormat.SAVED_MODEL): self._convert_tensorflow_to_saved_model,
            (ModelFormat.TENSORFLOW, ModelFormat.ONNX): self._convert_tensorflow_to_onnx,
            (ModelFormat.SAVED_MODEL, ModelFormat.ONNX): self._convert_saved_model_to_onnx,
            (ModelFormat.SAVED_MODEL, ModelFormat.TFLITE): self._convert_saved_model_to_tflite
        }
        
        key = (self.config.input_format, self.config.output_format)
        if key not in conversion_map:
            raise ValueError(
                f"No conversion method found for {self.config.input_format.value} to {self.config.output_format.value}"
            )
        
        return conversion_map[key]
    
    def _convert_pytorch_to_torchscript(self) -> str:
        """
        Convert PyTorch model to TorchScript
        """
        # Load PyTorch model
        model = torch.load(self.config.input_path, map_location=self.config.device)
        
        # Create example input if not provided
        if self.config.example_input is None:
            self.config.example_input = torch.randn(1, 3, 224, 224, device=self.config.device)
        
        # Convert to TorchScript
        scripted_model = torch.jit.trace(model, self.config.example_input)
        
        # Save TorchScript model
        torch.jit.save(scripted_model, self.config.output_path)
        
        return self.config.output_path
    
    def _convert_pytorch_to_onnx(self) -> str:
        """
        Convert PyTorch model to ONNX
        """
        # Load PyTorch model
        model = torch.load(self.config.input_path, map_location=self.config.device)
        model.eval()
        
        # Create example input if not provided
        if self.config.example_input is None:
            self.config.example_input = torch.randn(1, 3, 224, 224, device=self.config.device)
        
        # Convert to ONNX
        torch.onnx.export(
            model,
            self.config.example_input,
            self.config.output_path,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return self.config.output_path
    
    def _convert_pytorch_to_tflite(self) -> str:
        """
        Convert PyTorch model to TFLite
        """
        # First convert to ONNX
        onnx_path = self.config.output_path.replace('.tflite', '.onnx')
        temp_config = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=self.config.input_path,
            output_path=onnx_path,
            example_input=self.config.example_input,
            opset_version=self.config.opset_version,
            verbose=self.config.verbose,
            optimize=self.config.optimize,
            device=self.config.device
        )
        onnx_converter = ModelConverter(temp_config)
        onnx_converter.convert()
        
        # Then convert ONNX to TFLite
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel
        saved_model_path = self.config.output_path.replace('.tflite', '.savedmodel')
        tf_rep.export_graph(saved_model_path)
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        if self.config.optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(self.config.output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Clean up temporary files
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(saved_model_path):
            import shutil
            shutil.rmtree(saved_model_path)
        
        return self.config.output_path
    
    def _convert_pytorch_to_openvino(self) -> str:
        """
        Convert PyTorch model to OpenVINO
        """
        # First convert to ONNX
        onnx_path = self.config.output_path.replace('.xml', '.onnx')
        temp_config = ConversionConfig(
            input_format=ModelFormat.PYTORCH,
            output_format=ModelFormat.ONNX,
            input_path=self.config.input_path,
            output_path=onnx_path,
            example_input=self.config.example_input,
            opset_version=self.config.opset_version,
            verbose=self.config.verbose,
            optimize=self.config.optimize,
            device=self.config.device
        )
        onnx_converter = ModelConverter(temp_config)
        onnx_converter.convert()
        
        # Then convert ONNX to OpenVINO
        from openvino.tools import mo
        
        # Convert ONNX model to OpenVINO IR
        mo.convert(
            model=onnx_path,
            input_model=onnx_path,
            output_dir=os.path.dirname(self.config.output_path),
            input_shape=[1, 3, 224, 224],
            data_type="FP32"
        )
        
        # Clean up temporary files
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        return self.config.output_path
    
    def _convert_pytorch_to_coreml(self) -> str:
        """
        Convert PyTorch model to CoreML
        """
        # Load PyTorch model
        model = torch.load(self.config.input_path, map_location=self.config.device)
        model.eval()
        
        # Create example input if not provided
        if self.config.example_input is None:
            self.config.example_input = torch.randn(1, 3, 224, 224, device=self.config.device)
        
        # Convert to CoreML
        import coremltools as ct
        
        # Trace model
        traced_model = torch.jit.trace(model, self.config.example_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=self.config.example_input.shape)],
            convert_to="mlprogram"
        )
        
        # Save CoreML model
        coreml_model.save(self.config.output_path)
        
        return self.config.output_path
    
    def _convert_torchscript_to_onnx(self) -> str:
        """
        Convert TorchScript model to ONNX
        """
        # Load TorchScript model
        model = torch.jit.load(self.config.input_path, map_location=self.config.device)
        
        # Create example input if not provided
        if self.config.example_input is None:
            self.config.example_input = torch.randn(1, 3, 224, 224, device=self.config.device)
        
        # Convert to ONNX
        torch.onnx.export(
            model,
            self.config.example_input,
            self.config.output_path,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return self.config.output_path
    
    def _convert_onnx_to_tensorrt(self) -> str:
        """
        Convert ONNX model to TensorRT
        """
        import tensorrt as trt
        
        # Create TensorRT builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(self.config.input_path, 'rb') as f:
            parser.parse(f.read())
        
        # Build engine
        config = builder.create_builder_config()
        if self.config.optimize:
            config.max_workspace_size = 1 << 30  # 1GB
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        # Save TensorRT engine
        with open(self.config.output_path, 'wb') as f:
            f.write(serialized_engine)
        
        return self.config.output_path
    
    def _convert_onnx_to_tensorrt_engine(self) -> str:
        """
        Convert ONNX model to TensorRT engine
        """
        # Same as TensorRT conversion
        return self._convert_onnx_to_tensorrt()
    
    def _convert_onnx_to_openvino(self) -> str:
        """
        Convert ONNX model to OpenVINO
        """
        from openvino.tools import mo
        
        # Convert ONNX model to OpenVINO IR
        mo.convert(
            model=self.config.input_path,
            input_model=self.config.input_path,
            output_dir=os.path.dirname(self.config.output_path),
            input_shape=[1, 3, 224, 224],
            data_type="FP32"
        )
        
        return self.config.output_path
    
    def _convert_onnx_to_onnx_runtime(self) -> str:
        """
        Convert ONNX model to ONNX Runtime format
        """
        # ONNX Runtime uses ONNX format directly, so just copy the file
        import shutil
        shutil.copy2(self.config.input_path, self.config.output_path)
        
        # Optimize if requested
        if self.config.optimize:
            import onnxruntime
            from onnxruntime.quantization import quantize_dynamic
            quantize_dynamic(
                self.config.output_path,
                self.config.output_path.replace('.onnx', '_optimized.onnx'),
                per_channel=False,
                reduce_range=False
            )
        
        return self.config.output_path
    
    def _convert_tensorflow_to_tflite(self) -> str:
        """
        Convert TensorFlow model to TFLite
        """
        import tensorflow as tf
        
        # Load TensorFlow model
        model = tf.saved_model.load(self.config.input_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(self.config.input_path)
        if self.config.optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(self.config.output_path, 'wb') as f:
            f.write(tflite_model)
        
        return self.config.output_path
    
    def _convert_tensorflow_to_saved_model(self) -> str:
        """
        Convert TensorFlow model to SavedModel
        """
        import tensorflow as tf
        
        # Load TensorFlow model
        model = tf.keras.models.load_model(self.config.input_path)
        
        # Save as SavedModel
        tf.saved_model.save(model, self.config.output_path)
        
        return self.config.output_path
    
    def _convert_tensorflow_to_onnx(self) -> str:
        """
        Convert TensorFlow model to ONNX
        """
        import tensorflow as tf
        from tf2onnx.convert import convert
        
        # Load TensorFlow model
        model = tf.keras.models.load_model(self.config.input_path)
        
        # Convert to ONNX
        convert(
            input_signature=[tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input")],
            output_path=self.config.output_path,
            saved_model=self.config.input_path
        )
        
        return self.config.output_path
    
    def _convert_saved_model_to_onnx(self) -> str:
        """
        Convert SavedModel to ONNX
        """
        from tf2onnx.convert import convert
        
        # Convert SavedModel to ONNX
        convert(
            input_signature=[tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input")],
            output_path=self.config.output_path,
            saved_model=self.config.input_path
        )
        
        return self.config.output_path
    
    def _convert_saved_model_to_tflite(self) -> str:
        """
        Convert SavedModel to TFLite
        """
        import tensorflow as tf
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(self.config.input_path)
        if self.config.optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(self.config.output_path, 'wb') as f:
            f.write(tflite_model)
        
        return self.config.output_path


def convert_model(config: ConversionConfig) -> str:
    """
    Convert model using the given configuration
    
    Args:
        config: Conversion configuration
    
    Returns:
        Path to converted model
    """
    converter = ModelConverter(config)
    return converter.convert()


def get_converted_model(
    input_path: str,
    output_path: str,
    input_format: Union[str, ModelFormat],
    output_format: Union[str, ModelFormat],
    **kwargs
) -> str:
    """
    Convert model from input path to output path
    
    Args:
        input_path: Path to input model
        output_path: Path to save output model
        input_format: Input model format
        output_format: Output model format
        **kwargs: Additional conversion options
    
    Returns:
        Path to converted model
    """
    # Convert format strings to ModelFormat enum
    if isinstance(input_format, str):
        input_format = ModelFormat(input_format)
    if isinstance(output_format, str):
        output_format = ModelFormat(output_format)
    
    # Create conversion config
    config = ConversionConfig(
        input_format=input_format,
        output_format=output_format,
        input_path=input_path,
        output_path=output_path,
        **kwargs
    )
    
    # Convert model
    return convert_model(config)


def validate_converted_model(
    model_path: str,
    model_format: Union[str, ModelFormat],
    example_input: Optional[torch.Tensor] = None
) -> bool:
    """
    Validate converted model
    
    Args:
        model_path: Path to converted model
        model_format: Model format
        example_input: Example input tensor
    
    Returns:
        True if model is valid, False otherwise
    """
    if isinstance(model_format, str):
        model_format = ModelFormat(model_format)
    
    try:
        if model_format == ModelFormat.ONNX:
            import onnx
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            return True
        
        elif model_format == ModelFormat.TORCHSCRIPT:
            model = torch.jit.load(model_path)
            if example_input is None:
                example_input = torch.randn(1, 3, 224, 224)
            model(example_input)
            return True
        
        elif model_format == ModelFormat.TFLITE:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return True
        
        elif model_format == ModelFormat.OPENVINO:
            from openvino.runtime import Core
            ie = Core()
            ie.read_model(model_path)
            return True
        
        elif model_format == ModelFormat.TENSORRT:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(model_path, 'rb') as f:
                runtime.deserialize_cuda_engine(f.read())
            return True
        
        else:
            # For other formats, just check if file exists
            return os.path.exists(model_path)
    
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
