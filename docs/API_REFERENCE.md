# API 参考文档

## VisionPipeline

用于创建计算机视觉管道的主类。

### 构造函数

```python
VisionPipeline(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `detector_config`：检测器配置
    - `model_path`：模型文件路径或模型名称（例如，"yolov8n.pt"、"yolov26n.pt"）
    - `device`：运行推理的设备（"auto"、"cuda"、"cpu"、"cuda:0"）
    - `conf_threshold`：检测结果的默认置信度阈值
    - `iou_threshold`：NMS（非最大抑制）的IoU阈值
    - `category_thresholds`：特定类别的置信度阈值字典（例如，{"person": 0.5, "car": 0.3}）
    - `use_fp16`：是否使用FP16精度推理
    - `batch_inference`：是否启用批量推理
    - `dynamic_batch_size`：是否启用动态批量大小
  - `tracker_config`：跟踪器配置（如果启用跟踪）
    - `tracker_type`：使用的跟踪器类型（"bytetrack"、"ioutracker"、"reidtracker"）
    - `max_age`：轨迹在被移除前的最大年龄
    - `iou_threshold`：轨迹匹配的IoU阈值
  - `enable_tracking`：是否启用目标跟踪
  - `enable_performance_monitoring`：是否启用性能监控
  - `performance_metrics`：要监控的性能指标列表

### 方法

#### `process(image: np.ndarray) -> Dict[str, Any]`
处理单张图像并返回检测/跟踪结果。

#### `process_frame(image: np.ndarray) -> Dict[str, Any]`
`process`方法的别名，用于视频帧处理。

#### `process_batch(images: List[np.ndarray]) -> List[Dict[str, Any]]`
批量处理多张图像。

#### `process_video(input_source: str or int, output_path: Optional[str] = None, start_frame: int = 0, end_frame: Optional[int] = None, skip_frames: int = 0, frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None, progress_callback: Optional[Callable[[float, int, int], None]] = None) -> bool`
处理视频文件或流。

#### `process_video_batch(input_source: str or int, output_path: Optional[str] = None, start_frame: int = 0, end_frame: Optional[int] = None, batch_size: int = 8, skip_frames: int = 0, frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None, progress_callback: Optional[Callable[[float, int, int], None]] = None) -> bool`
使用批量处理来处理视频，以提高性能。

### 静态方法

#### `VisionPipeline.process_image(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
无需初始化管道即可快速处理图像。

#### `VisionPipeline.run_video(input_source: str or int, output_path: Optional[str] = None, model_path: str = "yolov8n.pt", enable_tracking: bool = False, conf_threshold: float = 0.25, batch_size: int = 0, **kwargs) -> bool`
使用简单API运行视频处理。

### 类方法

#### `VisionPipeline.with_tracking(config: Optional[Dict[str, Any]] = None) -> VisionPipeline`
创建一个启用了跟踪功能的管道。

#### `VisionPipeline.from_model(model_path: str, enable_tracking: bool = False, conf_threshold: float = 0.25) -> VisionPipeline`
从特定模型创建管道。

## 检测器

### YOLODetector

```python
YOLODetector(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_path`：模型文件路径或模型名称（例如，"yolov8n.pt"、"yolov26n.pt"）
  - `device`：运行推理的设备（"auto"、"cuda"、"cpu"、"cuda:0"）
  - `conf_threshold`：检测结果的默认置信度阈值
  - `iou_threshold`：NMS（非最大抑制）的IoU阈值
  - `category_thresholds`：特定类别的置信度阈值字典（例如，{"person": 0.5, "car": 0.3}）
  - `enable_segmentation`：是否启用实例分割
  - `batch_inference`：是否启用批量推理
  - `dynamic_batch_size`：是否启用动态批量大小
  - `use_fp16`：是否使用FP16精度推理

**方法：**
- `detect(image: np.ndarray) -> List[Dict[str, Any]]`：检测图像中的目标
- `detect_batch(images: List[np.ndarray]) -> List[List[Dict[str, Any]]]`：批量检测多张图像中的目标
- `initialize() -> None`：初始化检测器（加载模型等）
- `cleanup() -> None`：清理资源

## 跟踪器

### BaseTracker

```python
BaseTracker(
    max_age: int = 30,
    iou_threshold: float = 0.3,
    min_hits: int = 3
)
```

**方法：**
- `update(detections: List[Dict[str, Any]], frame: np.ndarray = None) -> List[Dict[str, Any]]`：使用新的检测结果更新跟踪
- `reset() -> None`：重置跟踪器

### ByteTrack

```python
ByteTrack(
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    frame_rate: int = 30
)
```

### IoUTracker

```python
IoUTracker(
    max_age: int = 30,
    iou_threshold: float = 0.3,
    min_hits: int = 3
)
```

### ReidTracker

```python
ReidTracker(
    max_age: int = 30,
    iou_threshold: float = 0.3,
    min_hits: int = 3,
    reid_model: str = "osnet_x0_25_market1501.pt"
)
```

## 数据结构

### 检测结果

```python
{
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],  # 边界框坐标
            "confidence": 0.95,        # 置信度分数
            "class_id": 0,             # 类别ID
            "class_name": "person",   # 类别名称
            "track_id": 123,           # 跟踪ID（如果启用跟踪）
            "color": [0, 255, 0]       # 可视化颜色
        }
    ],
    "timestamp": 1620000000.123,  # 时间戳（秒）
    "frame_id": 1,                # 帧ID（用于视频）
    "metadata": {
        "model": "yolov8n.pt",
        "input_shape": (640, 640),
        "processing_time": 0.01     # 处理时间（秒）
    }
}
```

## 工具类

### VideoUtils

#### `VideoProcessor`
```python
VideoProcessor(
    input_path: str,
    output_path: str = None,
    show: bool = False,
    fps: int = None,
    resolution: Tuple[int, int] = None
)
```

**方法：**
- `get_frame() -> Tuple[bool, np.ndarray]`：获取下一帧
- `write_frame(frame: np.ndarray) -> None`：将帧写入输出
- `release() -> None`：释放资源

#### `process_video(input_path: str, process_func: Callable, output_path: str = None, show: bool = False) -> None`
使用自定义处理函数处理视频。

### 配置工具

#### `load_config(config_path: str) -> Dict[str, Any]`
从YAML文件加载配置。

#### `save_config(config: Dict[str, Any], config_path: str) -> None`
将配置保存到YAML文件。

#### `DeviceManager`
```python
DeviceManager(
    preferred_device: str = "auto"
)
```

**方法：**
- `get_best_device() -> str`：获取最佳可用设备
- `normalize_device(device: str) -> str`：规范化设备字符串

### 模型工具

#### `ModelCache`
```python
ModelCache(
    cache_dir: str = None
)
```

**方法：**
- `get_model_path(model_name: str) -> str`：获取缓存模型的路径
- `cache_model(model_path: str, model_name: str) -> str`：缓存模型
- `clear_cache() -> None`：清除缓存

## 异常类

### VisionFrameworkError
所有Vision Framework错误的基类。

### ModelError
当模型加载或推理出现错误时引发。

### ConfigurationError
当配置出现错误时引发。

### DeviceError
当设备选择或使用出现错误时引发。

### VideoError
当视频处理出现错误时引发。

## 配置示例

### YAML配置

```yaml
model:
  path: "yolov8n.pt"
  device: "auto"
  conf_threshold: 0.25
  iou_threshold: 0.45
  fp16: false
  verbose: false

detection:
  classes: [0, 1, 2]  # 人、自行车、汽车
  class_conf_thresholds:
    person: 0.5
    car: 0.3

tracking:
  enable: true
  type: "bytetrack"
  max_age: 30
  iou_threshold: 0.3

visualization:
  show_bbox: true
  show_labels: true
  show_scores: true
  show_track_id: true
```

### Python配置

```python
config = {
    "model": {
        "path": "yolov26n.pt",
        "device": "auto",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    },
    "detection": {
        "classes": [0, 2, 5, 7],  # 人、汽车、公交车、卡车
        "class_conf_thresholds": {
            "person": 0.5,
            "car": 0.3
        }
    },
    "tracking": {
        "enable": True,
        "type": "bytetrack"
    }
}
```

## 使用示例

### 基本检测

```python
from visionframework.core.pipeline import VisionPipeline
import cv2

# 初始化管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.5
    }
})

# 处理图像
image = cv2.imread("test.jpg")
results = pipeline.process(image)

# 手动绘制边界框
for detection in results["detections"]:
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, detection["class_name"], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow("结果", image)
cv2.waitKey(0)
```

### 带跟踪的视频处理

```python
from visionframework.core.pipeline import VisionPipeline

# 初始化带跟踪的管道
pipeline = VisionPipeline.with_tracking({
    "detector_config": {
        "model_path": "yolov8n.pt"
    },
    "tracker_config": {
        "tracker_type": "bytetrack"
    },
    "conf_threshold": 0.3
})

# 处理视频
pipeline.process_video(
    input_path="input.mp4",
    output_path="output.mp4"
)
```

### 简化API

```python
from visionframework.core.pipeline import VisionPipeline
import cv2

# 使用静态方法快速处理
image = cv2.imread("test.jpg")
results = VisionPipeline.process_image(
    image,
    {
        "detector_config": {
            "model_path": "yolov26n.pt",
            "conf_threshold": 0.4
        },
        "enable_tracking": True
    }
)

# 绘制结果
for det in results["detections"]:
    x1, y1, x2, y2 = det["bbox"]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), det["color"], 2)
    cv2.putText(image, f"{det['class_name']} {det['confidence']:.2f}",
                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det["color"], 2)

cv2.imshow("结果", image)
cv2.waitKey(0)
```
