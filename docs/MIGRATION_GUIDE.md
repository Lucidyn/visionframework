"""
导入迁移指南 - v0.2.8

本文档说明如何更新代码以使用 v0.2.8 的新导入结构。
"""

# ============================================================================
# 特征提取器导入变化
# ============================================================================

# OLD: v0.2.7 及更早版本
from visionframework.core.clip import CLIPExtractor
from visionframework.core.pose_estimator import PoseEstimator
from visionframework.core.reid import ReIDExtractor

# NEW: v0.2.8+（推荐方式）
from visionframework import CLIPExtractor, PoseEstimator, ReIDExtractor

# 也可以从 processors 子模块导入
from visionframework.core.processors import CLIPExtractor, PoseEstimator, ReIDExtractor

# ============================================================================
# 异常导入（新增）
# ============================================================================

# OLD: 没有统一的异常系统（v0.2.7 及更早）

# NEW: v0.2.8+（推荐方式）
from visionframework import (
    VisionFrameworkError,
    DetectorInitializationError,
    DetectorInferenceError,
    TrackerInitializationError,
    TrackerUpdateError,
    ConfigurationError,
    ModelNotFoundError,
    ModelLoadError,
    DeviceError,
    DependencyError,
    DataFormatError,
    ProcessingError
)

# 使用异常
try:
    detector = YOLODetector(config)
except DetectorInitializationError as e:
    print(f"检测器初始化失败: {e}")

# ============================================================================
# 模型管理（新增）
# ============================================================================

# OLD: 没有统一的模型管理（v0.2.7 及更早）

# NEW: v0.2.8+（推荐方式）
from visionframework import ModelManager, get_model_manager

# 使用全局实例
model_manager = get_model_manager()
model_path = model_manager.get_model_path("yolov8n.pt")

# 或创建自己的实例
custom_manager = ModelManager(cache_dir="/custom/path")

# ============================================================================
# 特征提取器基类（新增）
# ============================================================================

# NEW: v0.2.8+
from visionframework.core.processors import FeatureExtractor

# 创建自定义特征提取器
class CustomFeatureExtractor(FeatureExtractor):
    def initialize(self) -> None:
        # 初始化你的模型
        pass
    
    def extract(self, input_data):
        # 实现特征提取
        pass
    
    def _move_to_device(self, device: str) -> None:
        # 实现设备移动
        pass

# ============================================================================
# 向后兼容性
# ============================================================================

# 为了向后兼容，旧的导入路径仍然可用：
from visionframework.core.clip import CLIPExtractor  # 仍然有效（deprecated）
from visionframework.core.pose_estimator import PoseEstimator  # 仍然有效（deprecated）

# 但强烈建议更新到新的导入路径

# ============================================================================
# 迁移检查表
# ============================================================================

# [ ] 更新所有 from visionframework.core.clip 导入
# [ ] 更新所有 from visionframework.core.pose_estimator 导入
# [ ] 更新所有 from visionframework.core.reid 导入
# [ ] 添加适当的异常处理
# [ ] 考虑使用 ModelManager 来管理模型
# [ ] 更新任何自定义特征提取器以继承 FeatureExtractor
