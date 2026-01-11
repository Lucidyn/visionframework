# Vision Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.5-orange.svg)](https://github.com/visionframework/visionframework)

一个功能完整的计算机视觉框架，包含目标检测、目标跟踪等功能，可以轻松集成到您的项目中。

## 功能特性

- **目标检测**: 支持 YOLO、DETR 和 RF-DETR 三种检测模型
- **实例分割**: 支持 YOLO 实例分割，可获取对象掩码
- **目标跟踪**: 支持基础跟踪、ByteTrack 高级跟踪和 ReID 特征跟踪
- **姿态估计**: 人体/物体姿态检测，支持关键点检测和骨架绘制
- **轨迹分析**: 速度、方向、距离等轨迹分析工具
- **评估工具**: 检测评估（mAP）和跟踪评估（MOTA）工具
- **可视化工具**: 丰富的可视化功能，支持检测框、跟踪轨迹、姿态骨架等
- **区域检测**: ROI/Zone 检测，支持矩形、多边形、圆形区域
- **计数功能**: 对象进入/离开/停留计数
- **结果导出**: 支持 JSON、CSV、COCO 格式导出
- **性能分析**: FPS 统计、处理时间分析
- **视频处理**: 便捷的视频读取、写入和处理工具
- **模块化设计**: 各模块独立，可灵活组合使用

## 安装

### 安装依赖

```bash
pip install -r requirements.txt
```

或使用 setup.py 安装：

```bash
pip install -e .
```

## 快速开始

### 基本使用 - 仅检测

```python
from visionframework import Detector, Visualizer
import cv2

# 初始化检测器
detector = Detector({
    "model_path": "yolov8n.pt",  # 会自动下载
    "conf_threshold": 0.25
})
detector.initialize()

# 加载图像
image = cv2.imread("your_image.jpg")

# 运行检测
detections = detector.detect(image)
print(f"检测到 {len(detections)} 个对象")

# 可视化结果
visualizer = Visualizer()
result_image = visualizer.draw_detections(image, detections)
cv2.imwrite("output.jpg", result_image)
```

### 检测 + 跟踪

```python
from visionframework import VisionPipeline, Visualizer
import cv2

# 初始化完整管道
pipeline = VisionPipeline({
    "enable_tracking": True,
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.25
    }
})
pipeline.initialize()

# 处理视频帧
cap = cv2.VideoCapture("your_video.mp4")
visualizer = Visualizer()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理帧
    results = pipeline.process(frame)
    tracks = results["tracks"]
    
    # 可视化
    result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
    cv2.imshow("Tracking", result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 检测 + 跟踪 + 实例分割 (ReID)

```python
from visionframework import VisionPipeline, Visualizer
import cv2

# 初始化完整管道
pipeline = VisionPipeline({
    "enable_tracking": True,
    "detector_config": {
        "model_type": "yolo",
        "model_path": "yolov8n-seg.pt",  # 使用分割模型
        "enable_segmentation": True,     # 开启实例分割
        "conf_threshold": 0.25
    },
    "tracker_config": {
        "tracker_type": "reid",          # 使用 ReID 跟踪器
        "reid_config": {
            "device": "cuda"             # ReID 模型设备
        },
        "max_age": 30,
        "min_hits": 3
    }
})
pipeline.initialize()

# ... (后续处理代码相同)
```

### 使用不同的检测模型

#### YOLO（默认）

```python
detector = Detector({
    "model_type": "yolo",
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.25
})
```

#### DETR

```python
detector = Detector({
    "model_type": "detr",
    "detr_model_name": "facebook/detr-resnet-50",
    "conf_threshold": 0.5
})
```

#### RF-DETR

```python
detector = Detector({
    "model_type": "rfdetr",
    "conf_threshold": 0.5,
    "device": "cuda"  # 推荐使用 GPU
})
```

## API 文档

### Detector (检测器)

检测器用于在图像中检测目标对象。

**配置参数:**
- `model_type`: 模型类型，可选: "yolo", "detr", "rfdetr" (默认: "yolo")
- `model_path`: 模型文件路径 (默认: "yolov8n.pt")
- `conf_threshold`: 置信度阈值 (默认: 0.25)
- `iou_threshold`: IoU 阈值 (默认: 0.45)
- `enable_segmentation`: 是否开启实例分割 (默认: False)
- `device`: 设备 ("cpu", "cuda", "mps") (默认: "cpu")

### Tracker (跟踪器)

跟踪器用于跟踪检测到的目标。

**配置参数:**
- `tracker_type`: 跟踪器类型，可选 "sort", "byte", "reid" (默认: "byte")
- `max_age`: 目标丢失的最大帧数 (默认: 30)
- `min_hits`: 确认跟踪的最小命中次数 (默认: 3)
- `iou_threshold`: 匹配的 IoU 阈值 (默认: 0.3)
- `reid_config`: ReID 跟踪器配置 (仅当 tracker_type="reid" 时有效)

### VisionPipeline (视觉管道)

完整的检测和跟踪管道，返回检测结果和跟踪结果。

### Visualizer (可视化器)

用于可视化检测和跟踪结果，支持绘制检测框、跟踪轨迹、姿态骨架等。

## 配置示例

框架支持多种配置方式，包括 Python 字典和配置文件（YAML/JSON）。

### 1. 使用字典配置

```python
from visionframework import VisionPipeline

config = {
    "enable_tracking": True,
    "detector_config": {
        "model_path": "yolov8s.pt",
        "conf_threshold": 0.3,
        "device": "cuda"
    },
    "tracker_config": {
        "max_age": 50,
        "min_hits": 5
    }
}

pipeline = VisionPipeline(config)
```

### 2. 使用配置文件 (推荐)

支持 YAML 和 JSON 格式的配置文件。

`config.yaml`:
```yaml
enable_tracking: true
detector_config:
  model_type: "yolo"
  model_path: "yolov8s.pt"
  conf_threshold: 0.3
  device: "cuda"
tracker_config:
  max_age: 50
  min_hits: 5
```

Python 代码:
```python
from visionframework import Config, VisionPipeline

# 加载配置
config = Config.load("config.yaml")
pipeline = VisionPipeline(config)
```

### 3. 获取默认配置

```python
from visionframework import Config

# 获取各模块默认配置
detector_config = Config.get_default_detector_config()
tracker_config = Config.get_default_tracker_config()
pipeline_config = Config.get_default_pipeline_config()
```

## 示例代码

查看 `examples/` 目录获取更多使用示例：

### 基础示例
- `basic_usage.py`: 基本使用示例（详细注释）
- `video_tracking.py`: 视频跟踪示例（详细注释）
- **`config_example.py`**: 使用配置文件示例（推荐，支持 YAML/JSON）

### 检测器示例
- `rfdetr_example.py`: RF-DETR 检测器示例
- `rfdetr_tracking.py`: RF-DETR 检测 + 跟踪示例
- `yolo_pose_example.py`: YOLO Pose 姿态估计示例

### 高级功能示例
- `advanced_features.py`: 高级功能示例（ROI、计数、性能监控等，详细注释）
- `batch_processing.py`: 批量图像处理示例

## 文档

- [快速开始指南](docs/QUICKSTART.md)
- [高级功能文档](docs/FEATURES.md)
- [项目结构说明](docs/PROJECT_STRUCTURE.md)
- [更新日志](docs/CHANGELOG.md)

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
- transformers >= 4.30.0 (用于 DETR 模型)
- rfdetr >= 0.1.0 (用于 RF-DETR 模型)
- supervision >= 0.18.0 (用于 RF-DETR 模型)
- pyyaml >= 6.0 (用于 YAML 配置文件支持)

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
