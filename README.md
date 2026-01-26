# Vision Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

轻量、模块化的计算机视觉框架，支持目标检测、跟踪、实例分割、姿态估计与结果导出。该仓库提供统一的高层 API，便于在工程中快速集成多种视觉能力。**新增：配置系统优化、模型管理增强、设备自动选择！**

主要目标：易用、模块化、可扩展。核心接口示例与快速上手指南见下文与 `docs/`。

## ⚡ 最短快速开始

**单张处理**：

```bash
pip install -e .
```

```python
from visionframework.core.detectors.yolo_detector import YOLODetector
import cv2

det = YOLODetector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
det.initialize()
img = cv2.imread("your_image.jpg")
print(len(det.detect(img)))  # 50 FPS
```

**批量处理（推荐）**：

```python
from visionframework.core.pipeline import VisionPipeline

pipeline = VisionPipeline({
    "detector_config": {"model_path": "yolov8n.pt", "batch_inference": True},
    "enable_tracking": True
})
pipeline.initialize()

frames = [cv2.imread(f"frame_{i}.jpg") for i in range(4)]
results = pipeline.process_batch(frames)  # 200 FPS！
```

## 📊 性能对比

| 方式 | 吞吐量 | 场景 |
|------|--------|------|
| 单张处理 | 50 FPS | 实时 |
| 批处理 (size=4) | 150 FPS | 视频 |
| 批处理 (size=8) | 200 FPS | 批量 |

## 文档

仓库的完整文档位于 `docs/`（已重建）。主要入口：

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 安装与最小示例 |
| [快速参考](docs/QUICK_REFERENCE.md) | 常用 API 与配置速查 |
| [功能特性](docs/FEATURES.md) | 功能一览与场景说明 |
| [项目结构](docs/PROJECT_STRUCTURE.md) | 代码组织与模块说明 |
| [架构概览](docs/ARCHITECTURE_V0.2.8.md) | 高层架构与组件交互 |
| [迁移指南](docs/MIGRATION_GUIDE.md) | 从旧版本迁移要点 |
| [变更日志](docs/CHANGELOG.md) | 版本历史 |
| **[批处理指南](BATCH_PROCESSING_GUIDE.md)** | **批处理详细使用指南** |
| **[批处理总结](BATCH_PROCESSING_SUMMARY.md)** | **批处理实现总结与API参考** |

示例脚本在 `examples/` 下，推荐先查看 `examples/README.md` 获取运行命令。

## 关键更新

**v0.2.12 - 示例与文档完善**:
- ✨ **CLIP示例代码**：新增 `09_clip_features.py` 示例，演示CLIP特征功能（图像-文本相似度、零样本分类）
- ✨ **姿态估计示例代码**：新增 `10_pose_estimation.py` 示例，演示姿态估计功能（YOLO Pose、MediaPipe Pose）
- ✨ **文档更新**：更新了 `examples/README.md`，添加了新示例的说明
- ✨ **示例优化**：优化了现有示例代码，修复了姿态估计示例中的方法调用错误

**v0.2.11 - 模型功能增强**:
- ✨ **SAM分割器集成**：添加 `SAMSegmenter` 类，支持自动分割、交互式分割（点/框提示），支持多种SAM模型变体（vit_h, vit_l, vit_b）
- ✨ **检测器+分割器集成**：增强 `Detector` 类，支持集成SAM分割器，实现检测+分割联合推理
- ✨ **CLIP模型扩展**：扩展 `CLIPExtractor` 类，支持多种CLIP模型（OpenAI CLIP、OpenCLIP、中文CLIP）
- ✨ **姿态估计增强**：增强 `PoseEstimator` 类，支持YOLO Pose和MediaPipe Pose模型
- ✨ **SAM示例代码**：新增 `08_segmentation_sam.py` 示例，演示SAM分割功能

**v0.2.10 - 配置与模型管理优化**:
- ✨ **配置系统优化**：整合 `Config` 类与 Pydantic 模型，消除重复默认值定义，添加 `load_as_model` 和 `save_model` 方法
- ✨ **模型管理增强**：增强 `ModelCache` 类，添加 `load_model` 方法，支持直接加载模型实例，改进模型下载和加载流程
- ✨ **设备管理改进**：添加设备自动选择功能，提供更详细的设备信息，支持 `auto_select_device`、`get_available_devices` 等方法
- ✨ **YOLODetector 简化**：简化模型加载逻辑，利用 `ModelManager` 和 `ModelCache` 来加载模型，改进设备选择和初始化流程
- ✨ **统一异常处理**：在所有模块中使用一致的异常类型，提供更详细的异常上下文信息

**v0.2.9 - 批处理优化**:
- ✨ **所有检测器支持批处理**：`detect_batch()` 方法，性能提升 4 倍
- ✨ **追踪器支持多帧处理**：`process_batch()` 方法，保持轨迹状态一致性
- ✨ **处理器支持批处理**：ReID、Pose 等处理器支持批量处理
- ✨ **端到端 Pipeline 批处理**：`VisionPipeline.process_batch()` 用于视频处理
- 🔍 **懒加载保护**：防止模块导入崩溃

**v0.2.8 - 类别过滤**:
- 新增 `categories` 参数：可在 `Detector` 配置或调用 `detect(image, categories=[...])` 时使用，用于在框架层面过滤返回结果（按类别名或 id）。

## 贡献与支持

欢迎通过 Issue/PR 贡献。有关开发依赖、测试和本地运行，请参阅 `pyproject.toml` 与 `requirements.txt`。

---

## API 示例

### 配置管理

```python
from visionframework.utils.config import Config
from visionframework.utils.config_models import DetectorConfig

# 获取各模块默认配置
detector_config = Config.get_default_detector_config()
tracker_config = Config.get_default_tracker_config()
pipeline_config = Config.get_default_pipeline_config()

# 直接从文件加载为 Pydantic 模型
model_config = Config.load_as_model("config.yaml", DetectorConfig)
print(model_config.model_path)  # yolov8n.pt
```

### 模型管理

```python
from visionframework.models import get_model_manager

# 获取模型管理器实例
model_manager = get_model_manager()

# 注册自定义模型
model_manager.register_model(
    name="custom_yolo",
    source="yolo",
    config={"file_path": "path/to/your/custom_model.pt"}
)

# 直接加载模型实例
model = model_manager.load_model("custom_yolo")
```

### 设备管理

```python
from visionframework.utils.device import DeviceManager

# 自动选择最佳可用设备
device = DeviceManager.auto_select_device()
print(f"Selected device: {device}")

# 获取所有可用设备
available_devices = DeviceManager.get_available_devices()
print(f"Available devices: {available_devices}")

# 获取设备详细信息
device_info = DeviceManager.get_device_info(device)
print(f"Device info: {device_info}")
```

### 工具类使用

#### 可视化工具

```python
from visionframework.utils.visualization import Visualizer
from visionframework.data.detection import Detection
import cv2
import numpy as np

# 创建可视化器
visualizer = Visualizer()

# 创建测试图像和检测结果
image = np.zeros((480, 640, 3), dtype=np.uint8)
detections = [
    Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
    Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car")
]

# 绘制检测结果
result = visualizer.draw_detections(image, detections)
cv2.imshow("Detections", result)
```

#### 评估工具

```python
from visionframework.utils.evaluation.detection_evaluator import DetectionEvaluator
from visionframework.data.detection import Detection

# 创建评估器
evaluator = DetectionEvaluator(iou_threshold=0.5)

# 创建预测和真实检测结果
pred_detections = [
    Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
    Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car")
]

gt_detections = [
    Detection(bbox=(105, 105, 205, 205), confidence=1.0, class_id=0, class_name="person"),
    Detection(bbox=(310, 160, 410, 260), confidence=1.0, class_id=1, class_name="car")
]

# 计算评估指标
metrics = evaluator.calculate_metrics(pred_detections, gt_detections)
print(f"准确率: {metrics['precision']:.2f}, 召回率: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}")
```

#### 性能监控

```python
from visionframework.utils.monitoring.performance import PerformanceMonitor, Timer
import time

# 创建性能监控器
monitor = PerformanceMonitor(window_size=30)
monitor.start()

# 模拟处理过程
with Timer("测试处理") as timer:
    for i in range(5):
        # 模拟检测过程
        with Timer() as det_timer:
            time.sleep(0.1)  # 模拟检测耗时
        monitor.record_detection_time(det_timer.get_elapsed())
        
        # 记录帧处理
        monitor.tick()

# 获取性能指标
print(f"当前FPS: {monitor.get_current_fps():.2f}")
print(f"平均FPS: {monitor.get_average_fps():.2f}")

# 打印性能摘要
monitor.print_summary()
```

#### 结果导出

```python
from visionframework.utils.data.export import ResultExporter
from visionframework.data.detection import Detection

# 创建结果导出器
exporter = ResultExporter()

# 创建检测结果
detections = [
    Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
    Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car")
]

# 导出为不同格式
exporter.export_detections_to_json(detections, "output/detections.json")
exporter.export_detections_to_csv(detections, "output/detections.csv")

# 导出为COCO格式
image_info = {"width": 640, "height": 480, "file_name": "test.jpg"}
exporter.export_to_coco_format(detections, 1, image_info, "output/coco_annotations.json")
```

## 示例代码

查看 `examples/` 目录获取完整示例代码：

| 示例 | 说明 |
|------|------|
| `00_basic_detection.py` | 基础目标检测示例 |
| `01_detection_with_tracking.py` | 带跟踪的目标检测示例 |
| `02_simplified_api.py` | 简化API使用示例 |
| `03_video_processing.py` | 视频文件处理示例 |
| `04_stream_processing.py` | 视频流处理示例 |
| `05_advanced_features.py` | 高级功能示例（模型管理、批量处理、配置文件、结果导出） |
| `06_tools_usage.py` | 工具类使用示例 |
| `07_enhanced_features.py` | 增强功能示例（ReID跟踪、轨迹分析、性能监控） |
| `08_segmentation_sam.py` | SAM分割示例（自动分割、交互式分割、检测+分割联合推理） |
| `09_clip_features.py` | CLIP特征示例（图像-文本相似度、零样本分类、图像特征提取） |
| `10_pose_estimation.py` | 姿态估计示例（YOLO Pose、MediaPipe Pose、关键点检测与可视化） |

## 依赖项

### 必需依赖
- opencv-python >= 4.8.0
- numpy >= 1.24.0, < 2.0.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics >= 8.0.0
- scipy >= 1.10.0
- Pillow >= 10.0.0

### 可选依赖
- transformers >= 4.30.0 (用于 DETR/CLIP 模型)
- segment-anything >= 1.0 (用于 SAM 分割模型)
- rfdetr >= 0.1.0 (用于 RF-DETR 模型)
- supervision >= 0.18.0 (用于 RF-DETR 模型)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 常见问题

**Q: 如何选择检测器？**  
A: YOLO 最快，DETR 精度最高，RF-DETR 平衡两者。根据需求选择。

**Q: 支持 GPU 加速吗？**  
A: 是的，所有模块都支持 CUDA。设置 `device: "cuda"` 即可。

**Q: 如何使用自定义模型？**  
A: 通过 `model_path` 参数指定模型文件路径即可。

**Q: 能扩展新功能吗？**  
A: 可以，所有模块都是可扩展的，支持继承和定制。

## 支持

- 阅读 [文档](docs/)
- 查看 [示例代码](examples/)
- 运行 [测试](tests/)
- 提出 [问题/建议](https://github.com/visionframework/visionframework/issues)

---

**Vision Framework v0.2.12** | 示例与文档完善版本 | 生产就绪

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
