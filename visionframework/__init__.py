"""
Vision Framework — 综合计算机视觉框架

用法：
    from visionframework import Vision

    v = Vision(model="yolov8n.pt", track=True)
    for frame, meta, result in v.run("video.mp4"):
        print(result["detections"])
"""

__version__ = "0.4.0"

# ── 主要公开 API ──────────────────────────────────────────────────────
from .api import Vision

# ── 数据结构（类型标注 & 测试常用）────────────────────────────────────
from .data import Detection, Track, STrack, Pose, KeyPoint, ROI

# ── 可视化 ─────────────────────────────────────────────────────────────
from .utils.visualization.unified_visualizer import Visualizer

# ── 结果导出 ────────────────────────────────────────────────────────────
from .utils.data.export import ResultExporter

# ── 异常类 ──────────────────────────────────────────────────────────────
from .exceptions import (
    VisionFrameworkError,
    DetectorInitializationError, DetectorInferenceError,
    TrackerInitializationError, TrackerUpdateError,
    ConfigurationError, ModelNotFoundError,
    ModelLoadError, DeviceError, DependencyError,
    DataFormatError, ProcessingError,
)


def __getattr__(name: str):
    """按需懒加载高级/内部符号，避免不必要的启动开销。"""
    _lazy_map = {
        # ── 基类 ──────────────────────────────────────────────────────
        "BaseDetector":  (".core.components.detectors.base_detector",  "BaseDetector"),
        "BaseTracker":   (".core.components.trackers.base_tracker",    "BaseTracker"),
        "BaseProcessor": (".core.components.processors.feature_extractor", "BaseProcessor"),

        # ── 检测器 ────────────────────────────────────────────────────
        "YOLODetector":  (".core.components.detectors.yolo_detector",  "YOLODetector"),
        "DETRDetector":  (".core.components.detectors.detr_detector",  "DETRDetector"),
        "RFDETRDetector":(".core.components.detectors.rfdetr_detector","RFDETRDetector"),

        # ── 跟踪器 ────────────────────────────────────────────────────
        "IOUTracker":    (".core.components.trackers.iou_tracker",     "IOUTracker"),
        "ByteTracker":   (".core.components.trackers.byte_tracker",    "ByteTracker"),
        "ReIDTracker":   (".core.components.trackers.reid_tracker",    "ReIDTracker"),

        # ── 跟踪工具函数 ──────────────────────────────────────────────
        "calculate_iou":     (".core.components.trackers.utils", "calculate_iou"),
        "iou_cost_matrix":   (".core.components.trackers.utils", "iou_cost_matrix"),
        "linear_assignment": (".core.components.trackers.utils", "linear_assignment"),
        "SCIPY_AVAILABLE":   (".core.components.trackers.utils", "SCIPY_AVAILABLE"),

        # ── 处理器 ────────────────────────────────────────────────────
        "PoseEstimator": (".core.components.processors.pose_estimator","PoseEstimator"),
        "CLIPExtractor": (".core.components.processors.clip_extractor","CLIPExtractor"),
        "ReIDExtractor": (".core.components.processors.reid_extractor","ReIDExtractor"),

        # ── 分割器 ────────────────────────────────────────────────────
        "SAMSegmenter":  (".core.components.segmenters.sam_segmenter", "SAMSegmenter"),

        # ── 管道 ──────────────────────────────────────────────────────
        "VisionPipeline":(".core.pipelines.pipeline", "VisionPipeline"),
        "BatchPipeline": (".core.pipelines.batch",    "BatchPipeline"),
        "VideoPipeline": (".core.pipelines.video",    "VideoPipeline"),

        # ── 插件系统 ──────────────────────────────────────────────────
        "PluginRegistry":          (".core.plugin_system", "PluginRegistry"),
        "ModelRegistry":           (".core.plugin_system", "ModelRegistry"),
        "register_detector":       (".core.plugin_system", "register_detector"),
        "register_tracker":        (".core.plugin_system", "register_tracker"),
        "register_segmenter":      (".core.plugin_system", "register_segmenter"),
        "register_processor":      (".core.plugin_system", "register_processor"),
        "register_model":          (".core.plugin_system", "register_model"),
        "register_visualizer":     (".core.plugin_system", "register_visualizer"),
        "register_evaluator":      (".core.plugin_system", "register_evaluator"),
        "register_custom_component":(".core.plugin_system","register_custom_component"),
        "plugin_registry":         (".core.plugin_system", "plugin_registry"),
        "model_registry":          (".core.plugin_system", "model_registry"),

        # ── ROI / 计数 ────────────────────────────────────────────────
        "ROIDetector": (".core.roi_detector", "ROIDetector"),
        "Counter":     (".core.counter",      "Counter"),

        # ── 配置 ──────────────────────────────────────────────────────
        "Config":        (".utils.io.config_models", "Config"),
        "BaseConfig":    (".utils.io.config_models", "BaseConfig"),
        "DetectorConfig":(".utils.io.config_models", "DetectorConfig"),

        # ── 监控 ──────────────────────────────────────────────────────
        "PerformanceMonitor":  (".utils.monitoring.performance", "PerformanceMonitor"),
        "PerformanceMetrics":  (".utils.monitoring.performance", "PerformanceMetrics"),
        "Timer":               (".utils.monitoring.timer",       "Timer"),

        # ── 媒体源 ────────────────────────────────────────────────────
        "iter_frames": (".utils.io.media_source", "iter_frames"),

        # ── 内存管理 ──────────────────────────────────────────────────
        "MemoryPool":           (".utils.memory.memory_manager", "MemoryPool"),
        "MultiMemoryPool":      (".utils.memory.memory_manager", "MultiMemoryPool"),
        "create_memory_pool":   (".utils.memory.memory_manager", "create_memory_pool"),
        "acquire_memory":       (".utils.memory.memory_manager", "acquire_memory"),
        "release_memory":       (".utils.memory.memory_manager", "release_memory"),
        "get_memory_pool_status":(".utils.memory.memory_manager","get_memory_pool_status"),
        "optimize_memory_usage":(".utils.memory.memory_manager", "optimize_memory_usage"),
        "clear_memory_pool":    (".utils.memory.memory_manager", "clear_memory_pool"),
        "clear_all_memory_pools":(".utils.memory.memory_manager","clear_all_memory_pools"),

        # ── 并发处理 ──────────────────────────────────────────────────
        "Task":                (".utils.concurrent.concurrent_processor", "Task"),
        "ThreadPoolProcessor": (".utils.concurrent.concurrent_processor", "ThreadPoolProcessor"),
        "parallel_map":        (".utils.concurrent.concurrent_processor", "parallel_map"),

        # ── 数据增强 ──────────────────────────────────────────────────
        "ImageAugmenter":    (".utils.data_augmentation.augmenter", "ImageAugmenter"),
        "AugmentationConfig":(".utils.data_augmentation.augmenter", "AugmentationConfig"),
        "AugmentationType":  (".utils.data_augmentation.augmenter", "AugmentationType"),
        "InterpolationType": (".utils.data_augmentation.augmenter", "InterpolationType"),

        # ── 模型优化 ──────────────────────────────────────────────────
        "QuantizationConfig":      (".utils.model_optimization.quantization", "QuantizationConfig"),
        "quantize_model":          (".utils.model_optimization.quantization", "quantize_model"),
        "PruningConfig":           (".utils.model_optimization.pruning",      "PruningConfig"),
        "prune_model":             (".utils.model_optimization.pruning",       "prune_model"),
        "DistillationConfig":      (".utils.model_optimization.distillation",  "DistillationConfig"),
        "distill_model":           (".utils.model_optimization.distillation",  "distill_model"),
        "compare_model_performance":(".utils.model_optimization.quantization", "compare_model_performance"),

        # ── 模型训练 ──────────────────────────────────────────────────
        "FineTuningConfig":   (".utils.model_training.fine_tuner", "FineTuningConfig"),
        "FineTuningStrategy": (".utils.model_training.fine_tuner", "FineTuningStrategy"),
        "ModelFineTuner":     (".utils.model_training.fine_tuner", "ModelFineTuner"),

        # ── 模型转换 ──────────────────────────────────────────────────
        "ModelFormat":           (".utils.model_conversion.formats",   "ModelFormat"),
        "get_supported_formats": (".utils.model_conversion.formats",   "get_supported_formats"),
        "is_format_supported":   (".utils.model_conversion.formats",   "is_format_supported"),
        "get_compatible_formats":(".utils.model_conversion.formats",   "get_compatible_formats"),
        "get_format_extension":  (".utils.model_conversion.formats",   "get_format_extension"),
        "get_format_dependencies":(".utils.model_conversion.formats",  "get_format_dependencies"),
        "get_format_from_extension":(".utils.model_conversion.formats","get_format_from_extension"),
        "ConversionConfig":      (".utils.model_conversion.converter", "ConversionConfig"),
        "ModelConverter":        (".utils.model_conversion.converter", "ModelConverter"),
        "convert_model":         (".utils.model_conversion.converter", "convert_model"),
        "validate_converted_model":(".utils.model_conversion.converter","validate_converted_model"),

        # ── 模型部署 ──────────────────────────────────────────────────
        "DeploymentPlatform":       (".utils.model_deployment.platforms", "DeploymentPlatform"),
        "get_supported_platforms":  (".utils.model_deployment.platforms", "get_supported_platforms"),
        "is_platform_supported":    (".utils.model_deployment.platforms", "is_platform_supported"),
        "get_platform_compatibility":(".utils.model_deployment.platforms","get_platform_compatibility"),
        "get_platform_requirements":(".utils.model_deployment.platforms", "get_platform_requirements"),
        "get_platform_from_string": (".utils.model_deployment.platforms", "get_platform_from_string"),
        "DeploymentConfig":         (".utils.model_deployment.deployer",  "DeploymentConfig"),
        "ModelDeployer":            (".utils.model_deployment.deployer",  "ModelDeployer"),
        "deploy_model":             (".utils.model_deployment.deployer",  "deploy_model"),
        "validate_deployment":      (".utils.model_deployment.deployer",  "validate_deployment"),

        # ── 模型管理 ──────────────────────────────────────────────────
        "select_model":    (".utils.model_management.auto_selector", "select_model"),
        "ModelSelector":   (".utils.model_management.auto_selector", "ModelSelector"),
        "ModelType":       (".utils.model_management.auto_selector", "ModelType"),
        "ModelRequirement":(".utils.model_management.auto_selector", "ModelRequirement"),
        "HardwareInfo":    (".utils.model_management.auto_selector", "HardwareInfo"),
        "HardwareTier":    (".utils.model_management.auto_selector", "HardwareTier"),

        # ── 多模态融合 ────────────────────────────────────────────────
        "FusionType":      (".utils.multimodal.fusion", "FusionType"),
        "MultimodalFusion":(".utils.multimodal.fusion", "MultimodalFusion"),
        "fuse_features":   (".utils.multimodal.fusion", "fuse_features"),
        "get_fusion_model":(".utils.multimodal.fusion", "get_fusion_model"),

        # ── 轨迹分析 ──────────────────────────────────────────────────
        "TrajectoryAnalyzer": (".utils.data.trajectory_analyzer", "TrajectoryAnalyzer"),

        # ── 评估工具 ──────────────────────────────────────────────────
        "DetectionEvaluator": (".utils.evaluation.detection_evaluator",  "DetectionEvaluator"),
        "TrackingEvaluator":  (".utils.evaluation.tracking_evaluator",   "TrackingEvaluator"),

        # ── 错误处理 & 依赖管理 ───────────────────────────────────────
        "ErrorHandler":              (".utils.error_handling",    "ErrorHandler"),
        "DependencyManager":         (".utils.dependency_manager","DependencyManager"),
        "dependency_manager":        (".utils.dependency_manager","dependency_manager"),
        "is_dependency_available":   (".utils.dependency_manager","is_dependency_available"),
        "get_available_dependencies":(".utils.dependency_manager","get_available_dependencies"),
        "get_missing_dependencies":  (".utils.dependency_manager","get_missing_dependencies"),
        "validate_dependency":       (".utils.dependency_manager","validate_dependency"),
        "get_install_command":       (".utils.dependency_manager","get_install_command"),
        "import_optional_dependency":(".utils.dependency_manager","import_optional_dependency"),
    }

    if name in _lazy_map:
        module_path, attr_name = _lazy_map[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"模块 {__name__!r} 中没有属性 {name!r}")


__all__ = [
    # 主要 API
    "Vision",
    "__version__",
    # 数据结构
    "Detection", "Track", "STrack", "Pose", "KeyPoint", "ROI",
    # 可视化
    "Visualizer",
    # 导出
    "ResultExporter",
    # 异常
    "VisionFrameworkError",
    "DetectorInitializationError", "DetectorInferenceError",
    "TrackerInitializationError", "TrackerUpdateError",
    "ConfigurationError", "ModelNotFoundError",
    "ModelLoadError", "DeviceError", "DependencyError",
    "DataFormatError", "ProcessingError",
]
