"""
VisionFramework — 模块化、组件式计算机视觉框架。

通过 backbone / neck / head 可组合组件构建视觉模型，
全程 YAML 配置驱动，支持检测、分割、跟踪、ReID 等任务。

唯一公共入口: ``TaskRunner(yaml_path)``（``yaml_path`` 为 ``str`` 或 ``pathlib.Path``）
"""

__version__ = "2.0.0"

# 数据结构
from visionframework.data import Detection, Track, STrack, Pose, KeyPoint, ROI

# 触发所有内置组件的注册
import visionframework.models  # noqa: F401
import visionframework.pipelines  # noqa: F401

# 公共接口
from visionframework.task_api import TaskRunner
from visionframework.utils.visualization import Visualizer

__all__ = [
    "__version__",
    "Detection", "Track", "STrack", "Pose", "KeyPoint", "ROI",
    "TaskRunner", "Visualizer",
]
