# 快速开始指南

## 安装

1. **安装依赖**
```bash
pip install -r requirements.txt
```

注意：如果遇到 NumPy 兼容性问题，请确保使用 NumPy < 2.0.0

2. **验证安装**
```bash
python tests/quick_test.py
```

## 基本使用

### 1. 仅使用检测功能

```python
from visionframework import Detector, Visualizer
import cv2

# 创建检测器
detector = Detector({
    "model_path": "yolov8n.pt",  # 首次使用会自动下载
    "conf_threshold": 0.25
})
detector.initialize()

# 加载图像
image = cv2.imread("your_image.jpg")

# 检测
detections = detector.detect(image)
print(f"检测到 {len(detections)} 个对象")

# 可视化
visualizer = Visualizer()
result = visualizer.draw_detections(image, detections)
cv2.imwrite("output.jpg", result)
```

### 2. 检测 + 跟踪

```python
from visionframework import VisionPipeline, Visualizer
import cv2

# 创建完整管道
pipeline = VisionPipeline({
    "enable_tracking": True,
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.25
    }
})
pipeline.initialize()

# 处理视频
cap = cv2.VideoCapture("your_video.mp4")
visualizer = Visualizer()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理
    results = pipeline.process(frame)
    tracks = results["tracks"]
    
    # 可视化（包含跟踪轨迹）
    result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
    cv2.imshow("Tracking", result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. 分别使用各个组件

```python
from visionframework import Detector, Tracker, Visualizer
import cv2

# 分别初始化
detector = Detector({"model_path": "yolov8n.pt"})
detector.initialize()

tracker = Tracker({"max_age": 30, "min_hits": 3})
tracker.initialize()

# 处理
image = cv2.imread("image.jpg")
detections = detector.detect(image)
tracks = tracker.update(detections)

# 可视化
visualizer = Visualizer()
result = visualizer.draw_tracks(image, tracks)
cv2.imwrite("output.jpg", result)
```

### 4. 实例分割 + ReID 跟踪

```python
from visionframework import VisionPipeline, Visualizer
import cv2

# 初始化管道
pipeline = VisionPipeline({
    "enable_tracking": True,
    "detector_config": {
        "model_type": "yolo",
        "model_path": "yolov8n-seg.pt",  # 分割模型
        "enable_segmentation": True      # 开启分割
    },
    "tracker_config": {
        "tracker_type": "reid",          # ReID 跟踪
        "reid_config": {
            "device": "cpu"              # 或 "cuda"
        }
    }
})
pipeline.initialize()

# 处理
image = cv2.imread("image.jpg")
results = pipeline.process(image)

# 可视化（自动绘制掩码）
visualizer = Visualizer()
result = visualizer.draw_results(image, results["detections"], results["tracks"])
cv2.imwrite("output_seg.jpg", result)
```

## 配置选项

### 检测器配置
- `model_type`: 模型类型 "yolo"/"detr"/"rfdetr"（默认: "yolo"）
- `model_path`: 模型路径（默认: "yolov8n.pt"，仅用于 YOLO）
- `conf_threshold`: 置信度阈值（默认: 0.25）
- `iou_threshold`: IoU阈值（默认: 0.45，仅用于 YOLO）
- `enable_segmentation`: 是否开启实例分割（默认: False）
- `device`: 设备类型 "cpu"/"cuda"/"mps"（默认: "cpu"）
- `detr_model_name`: DETR 模型名称（默认: "facebook/detr-resnet-50"）
- `rfdetr_model_name`: RF-DETR 模型名称（默认: None，使用默认模型）

性能选项（可通过 `performance` 子配置控制）:

- `batch_inference` (bool): 是否启用批量推理接口（对于支持的后端）；当为 True 时，`Detector.detect()` 可接受图像列表并返回按图像分组的检测结果。默认: False。
- `use_fp16` (bool): 在 GPU 上启用半精度推理（FP16），需在 `device` 设置为 `cuda` 时使用以获得加速。默认: False。

示例（在配置中启用性能选项）:

```python
detector_cfg = {
    "model_type": "detr",
    "device": "cuda",
    "performance": {"batch_inference": True, "use_fp16": True}
}
detector = Detector(detector_cfg)
detector.initialize()

# 单张图像
detections = detector.detect(image)

# 批量图像（返回按图像分组的检测结果）
batch_results = detector.detect([img1, img2])
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

### 跟踪器配置
- `tracker_type`: 跟踪器类型 "sort"/"byte"/"reid"（默认: "byte"）
- `max_age`: 目标丢失最大帧数（默认: 30）
- `min_hits`: 确认跟踪的最小命中次数（默认: 3）
- `iou_threshold`: 匹配IoU阈值（默认: 0.3）
- `reid_config`: ReID配置（仅 reid 类型有效）

### 可视化器配置
- `show_labels`: 显示标签（默认: True）
- `show_confidences`: 显示置信度（默认: True）
- `show_track_ids`: 显示跟踪ID（默认: True）
- `line_thickness`: 线条粗细（默认: 2）
- `font_scale`: 字体大小（默认: 0.5）

## 更多示例

查看 `examples/` 目录获取更多使用示例：

### 基础示例（详细注释）
- `basic_usage.py`: 基本使用示例（详细中文注释）
- `video_tracking.py`: 视频跟踪示例（详细中文注释）
- **`config_example.py`**: 使用配置文件示例（推荐，支持 YAML/JSON）

### 检测器示例
- `rfdetr_example.py`: RF-DETR 检测器使用示例
- `rfdetr_tracking.py`: RF-DETR 检测 + 跟踪示例
- `yolo_pose_example.py`: YOLO Pose 姿态估计示例

### 高级功能示例
- `advanced_features.py`: 高级功能示例（ROI、计数、性能监控等，详细注释）
- `batch_processing.py`: 批量图像处理示例

## 常见问题

**Q: 模型文件在哪里？**
A: 首次使用时，YOLO模型会自动下载到当前目录。DETR 和 RF-DETR 模型会自动从 HuggingFace/Roboflow 下载。

**Q: 如何提高检测精度？**
A: 使用更大的模型（如 yolov8s.pt, yolov8m.pt）或调整 conf_threshold。

**Q: 如何加速处理？**
A: 使用 GPU（设置 device="cuda"）或使用更小的模型。

**Q: 如何自定义跟踪参数？**
A: 在 tracker_config 中调整 max_age、min_hits 等参数。

## 下一步

- 阅读完整的 [README.md](README.md) 了解详细文档
- 查看 [examples/](examples/) 目录的示例代码
- 根据需要扩展框架功能

