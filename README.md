# Vision Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.8-orange.svg)](https://github.com/visionframework/visionframework)

一个功能完整、易于使用的计算机视觉框架，提供检测、跟踪、分割、姿态估计等多种 CV 任务支持。

## 主要功能

- 目标检测: YOLO、DETR、RF-DETR 多种模型
- 实例分割: YOLO 分割，支持掩码输出
- 目标跟踪: IOU、ByteTrack、ReID 三种跟踪方案
- 姿态估计: YOLO Pose，COCO 17 关键点检测
- 轨迹分析: 速度、方向、距离计算
- 评估工具: 检测评估（mAP）、跟踪评估（MOTA/MOTP/IDF1）
- 可视化: 检测框、轨迹、骨架、热力图等
- 区域检测: ROI/Zone 检测，矩形/多边形/圆形支持
- 计数功能: 进入/离开/停留计数
- 结果导出: JSON、CSV、COCO 格式
- 性能工具: FPS 统计、处理时间分析
- 视频处理: 视频读取、写入、处理工具
- 模块化: 独立模块，灵活组合

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 新手入门指南 |
| [功能特性](docs/FEATURES.md) | 完整功能列表 |
| [项目结构](docs/PROJECT_STRUCTURE.md) | 代码组织说明 |
| [迁移指南](docs/MIGRATION_GUIDE.md) | v0.2.7 升级指南 |
| [架构设计](docs/ARCHITECTURE_V0.2.8.md) | v0.2.8 架构说明 |
| [版本历史](docs/CHANGELOG.md) | 更新日志 |
| [快速参考](QUICK_REFERENCE.md) | API 快速查询 |

## 安装

### 基础安装

```bash
# 最小依赖安装
pip install -e .
```

### 可选功能安装

框架支持可选的功能组，可以按需安装：

```bash
# 安装 CLIP 零样本分类支持
pip install -e ".[clip]"

# 安装 DETR 检测器支持
pip install -e ".[detr]"

# 安装 RF-DETR 检测器支持
pip install -e ".[rfdetr]"

# 安装开发工具（测试、代码检查）
pip install -e ".[dev]"

# 安装所有功能和开发工具
pip install -e ".[all]"
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

### CLIP 零样本分类

```python
from visionframework import CLIPExtractor
from PIL import Image

# 初始化 CLIP 提取器
clip = CLIPExtractor({
    "device": "cuda",
    "use_fp16": True  # 在 GPU 上使用 FP16 加速
})
clip.initialize()

# 加载图像
image = Image.open("image.jpg")

# 零样本分类
labels = ["a cat", "a dog", "a person"]
scores = clip.zero_shot_classify(image, labels)

# 输出概率分数
for label, score in zip(labels, scores):
    print(f"{label}: {score:.4f}")

# 也可以获取单独的图像或文本嵌入
image_embedding = clip.encode_image(image)          # shape: (1, 512)
text_embeddings = clip.encode_text(labels)          # shape: (3, 512)

# 计算图像-文本相似度矩阵
similarity = clip.image_text_similarity(image, labels)  # shape: (1, 3)
```

### 跟踪评估工具

```python
from visionframework import TrackingEvaluator

# 初始化评估器（IoU 阈值 0.5）
evaluator = TrackingEvaluator(iou_threshold=0.5)

# 准备预测和真值轨迹数据
# 格式: List[List[{track_id, bbox}]] 其中每个内层列表代表一帧
pred_tracks = [
    [{"track_id": 1, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
    [{"track_id": 1, "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}}]
]
gt_tracks = [
    [{"track_id": 1, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
    [{"track_id": 1, "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}}]
]

# 一次性计算所有指标
results = evaluator.evaluate(pred_tracks, gt_tracks)
print(f"MOTA: {results['MOTA']:.4f}")  # Multiple Object Tracking Accuracy
print(f"MOTP: {results['MOTP']:.4f}")  # Multiple Object Tracking Precision (pixels)
print(f"IDF1: {results['IDF1']:.4f}")  # ID F1 Score
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")

# 或分别计算各项指标
mota_result = evaluator.calculate_mota(pred_tracks, gt_tracks)
motp_result = evaluator.calculate_motp(pred_tracks, gt_tracks)
idf1_result = evaluator.calculate_idf1(pred_tracks, gt_tracks)
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

性能选项（可通过 `performance` 子配置控制）:

- `batch_inference` (bool): 是否启用批量推理接口（对于支持的后端）；当为 True 时，`Detector.detect()` 可接受图像列表并返回按图像分组的检测结果。默认: False。
- `use_fp16` (bool): 在 GPU 上启用半精度推理（FP16），需在 `device` 设置为 `cuda` 时使用以获得加速。默认: False。

示例（启用 FP16 和批量推理）:

```python
detector = Detector({
    "model_type": "detr",
    "device": "cuda",
    "performance": {"batch_inference": True, "use_fp16": True}
})
detector.initialize()

# 单张图像
detections = detector.detect(image)

# 批量图像（返回按图像分组的检测结果）
batch_results = detector.detect([img1, img2])
```

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

## 示例代码

查看 `examples/` 目录获取完整示例代码：

| 示例 | 说明 |
|------|------|
| `basic_usage.py` | 基本使用示例 |
| `config_example.py` | 配置文件用法（推荐） |
| `video_tracking.py` | 视频跟踪示例 |
| `advanced_features.py` | 高级功能（ROI、计数等） |
| `batch_processing.py` | 批量处理示例 |
| `yolo_pose_example.py` | 姿态估计示例 |
| `rfdetr_example.py` | RF-DETR 检测器示例 |
| `rfdetr_tracking.py` | RF-DETR 跟踪示例 |
| `clip_example.py` | CLIP 零样本分类示例 |
| `tracking_evaluation_example.py` | 跟踪评估示例 |

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

**Vision Framework v0.2.8** | 架构优化版本 | 生产就绪
- pyyaml >= 6.0 (用于 YAML 配置文件支持)

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
