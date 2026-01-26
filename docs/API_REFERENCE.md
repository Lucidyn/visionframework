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

### Detector

统一检测器接口，支持多种检测模型和SAM分割器集成。

```python
Detector(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_path`：模型文件路径或模型名称（例如，"yolov8n.pt"、"yolov26n.pt"）
  - `model_type`：模型类型（"yolo"、"detr"、"rfdetr"）
  - `device`：运行推理的设备（"auto"、"cuda"、"cpu"、"cuda:0"）
  - `conf_threshold`：检测结果的默认置信度阈值
  - `iou_threshold`：NMS（非最大抑制）的IoU阈值
  - `category_thresholds`：特定类别的置信度阈值字典（例如，{"person": 0.5, "car": 0.3}）
  - `enable_segmentation`：是否启用实例分割
  - `batch_inference`：是否启用批量推理
  - `dynamic_batch_size`：是否启用动态批量大小
  - `use_fp16`：是否使用FP16精度推理
  - `segmenter_type`：分割器类型（None、"sam"）
  - `sam_model_path`：SAM模型文件路径
  - `sam_model_type`：SAM模型类型（"vit_h"、"vit_l"、"vit_b"）
  - `sam_use_fp16`：是否对SAM使用FP16精度

**方法：**
- `detect(image: np.ndarray, categories: Optional[list] = None) -> List[Detection]`：检测图像中的目标
- `detect_batch(images: List[np.ndarray], categories: Optional[list] = None) -> List[List[Detection]]`：批量检测多张图像中的目标
- `initialize() -> bool`：初始化检测器（加载模型等）
- `cleanup() -> None`：清理资源

### YOLODetector

YOLO检测器实现，支持YOLOv8和YOLO26系列模型。

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

## 分割器

### SAMSegmenter

Segment Anything Model (SAM) 分割器，支持自动分割和交互式分割。

```python
SAMSegmenter(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_type`：SAM模型类型（"vit_h"、"vit_l"、"vit_b"）
  - `model_path`：SAM模型文件路径
  - `device`：运行推理的设备（"auto"、"cuda"、"cpu"、"cuda:0"）
  - `use_fp16`：是否使用FP16精度推理
  - `automatic_threshold`：自动分割的质量阈值
  - `max_masks`：自动分割返回的最大掩码数量

**方法：**
- `automatic_segment(image: np.ndarray) -> List[Dict[str, Any]]`：自动分割图像
- `segment_with_points(image: np.ndarray, points: List[Tuple[int, int]], labels: List[int]) -> List[Dict[str, Any]]`：使用点提示分割
- `segment_with_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]`：使用框提示分割
- `segment_detections(image: np.ndarray, detections: List[Detection]) -> List[Detection]`：根据检测结果分割
- `process(image: np.ndarray, detections: Optional[List[Detection]] = None, **kwargs) -> Any`：通用处理方法
- `initialize() -> bool`：初始化分割器
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

### 可视化工具

#### Visualizer 类

统一可视化器，支持检测、跟踪和姿态估计结果的可视化。

```python
Visualizer(config: Optional[Dict[str, Any]] = None)
```

**参数：**
- `config`：配置字典，包含可视化相关的配置选项

**方法：**
- `draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray`：绘制检测结果
- `draw_tracks(image: np.ndarray, tracks: List[Track], draw_history: bool = True) -> np.ndarray`：绘制跟踪结果
- `draw_poses(image: np.ndarray, poses: List[Pose], draw_skeleton: bool = True, draw_keypoints: bool = True, draw_bbox: bool = True) -> np.ndarray`：绘制姿态估计结果
- `draw_results(image: np.ndarray, detections: Optional[List[Detection]] = None, tracks: Optional[List[Track]] = None, poses: Optional[List[Pose]] = None, draw_history: bool = True) -> np.ndarray`：绘制所有结果

#### DetectionVisualizer 类

检测结果可视化器，继承自 BaseVisualizer。

```python
DetectionVisualizer(config: Optional[Dict[str, Any]] = None)
```

**方法：**
- `draw_detection(image: np.ndarray, detection: Detection, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray`：绘制单个检测结果
- `draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray`：绘制多个检测结果

#### TrackVisualizer 类

跟踪结果可视化器，继承自 BaseVisualizer。

```python
TrackVisualizer(config: Optional[Dict[str, Any]] = None)
```

**方法：**
- `draw_track(image: np.ndarray, track: Track, color: Optional[Tuple[int, int, int]] = None, draw_history: bool = True) -> np.ndarray`：绘制单个跟踪结果
- `draw_tracks(image: np.ndarray, tracks: List[Track], draw_history: bool = True) -> np.ndarray`：绘制多个跟踪结果

### 评估工具

#### DetectionEvaluator 类

检测结果评估器，用于计算检测性能指标。

```python
DetectionEvaluator(iou_threshold: float = 0.5)
```

**参数：**
- `iou_threshold`：IoU 阈值，用于匹配预测和真实检测结果，默认为 0.5

**方法：**
- `calculate_metrics(pred_detections: List[Detection], gt_detections: List[Detection]) -> Dict[str, float]`：计算检测性能指标
- `calculate_map(all_pred_detections: List[List[Detection]], all_gt_detections: List[List[Detection]], num_classes: Optional[int] = None) -> Dict[str, Any]`：计算 mAP（mean Average Precision）

#### TrackingEvaluator 类

跟踪结果评估器，用于计算跟踪性能指标。

```python
TrackingEvaluator(iou_threshold: float = 0.5)
```

**参数：**
- `iou_threshold`：IoU 阈值，用于匹配预测和真实跟踪结果，默认为 0.5

**方法：**
- `calculate_mota(pred_tracks: List[Dict[str, Any]], gt_tracks: List[Dict[str, Any]]) -> Dict[str, float]`：计算 MOTA（Multiple Object Tracking Accuracy）
- `calculate_motp(pred_tracks: List[Dict[str, Any]], gt_tracks: List[Dict[str, Any]]) -> Dict[str, float]`：计算 MOTP（Multiple Object Tracking Precision）
- `calculate_idf1(pred_tracks: List[Dict[str, Any]], gt_tracks: List[Dict[str, Any]]) -> Dict[str, float]`：计算 IDF1（ID F1 Score）
- `evaluate(pred_tracks: List[Dict[str, Any]], gt_tracks: List[Dict[str, Any]]) -> Dict[str, Any]`：综合评估，计算所有跟踪指标

### 性能监控

#### PerformanceMonitor 类

性能监控器，用于监控和分析性能指标。

```python
PerformanceMonitor(window_size: int = 30)
```

**参数：**
- `window_size`：滑动窗口大小，用于计算 FPS 和其他指标，默认为 30

**方法：**
- `start()`：开始性能监控
- `tick()`：记录一帧处理完成
- `record_component_time(component: str, elapsed: float)`：记录组件处理时间
  - `component`：组件名称（"detection"、"tracking"、"visualization"）
  - `elapsed`：处理时间（秒）
- `get_metrics() -> PerformanceMetrics`：获取综合性能指标
- `reset()`：重置性能监控器

#### Timer 类

简单的计时器，用于测量代码块的执行时间。

```python
Timer(name: str = "Operation")
```

**参数：**
- `name`：计时器名称，默认为 "Operation"

**方法：**
- `__enter__() -> Timer`：进入上下文管理器，开始计时
- `__exit__(exc_type, exc_val, exc_tb) -> bool`：退出上下文管理器，结束计时
- `get_elapsed() -> float`：获取经过的时间

### 结果导出

#### ResultExporter 类

结果导出器，用于将检测和跟踪结果导出为各种格式。

```python
ResultExporter()
```

**方法：**
- `export_detections_to_json(detections: List[Detection], output_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool`：将检测结果导出为 JSON 格式
- `export_tracks_to_json(tracks: List[Track], output_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool`：将跟踪结果导出为 JSON 格式
- `export_detections_to_csv(detections: List[Detection], output_path: str) -> bool`：将检测结果导出为 CSV 格式
- `export_tracks_to_csv(tracks: List[Track], output_path: str) -> bool`：将跟踪结果导出为 CSV 格式
- `export_video_results_to_json(video_results: List[Dict[str, Any]], output_path: str, video_info: Optional[Dict[str, Any]] = None) -> bool`：将视频处理结果导出为 JSON 格式
- `export_to_coco_format(detections: List[Detection], image_id: int, image_info: Dict[str, Any], output_path: str) -> bool`：将检测结果导出为 COCO 格式

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

## CLIP提取器

### CLIPExtractor

CLIP模型封装，支持多种CLIP模型变体，用于图像-文本交互。

```python
CLIPExtractor(
    config: Optional[dict] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_name`：CLIP模型名称（例如，"openai/clip-vit-base-patch32"、"OFA-Sys/chinese-clip-vit-base-patch16"）
  - `device`：运行推理的设备（"auto"、"cuda"、"cpu"）
  - `use_fp16`：是否使用FP16精度推理
  - `cache_enabled`：是否启用嵌入缓存
  - `max_cache_size`：嵌入缓存的最大大小
  - `preprocess_options`：预处理选项

**方法：**
- `initialize() -> bool`：初始化CLIP模型和处理器
- `encode_image(image: Any) -> np.ndarray`：编码单张图像或图像列表
- `encode_text(texts: List[str]) -> np.ndarray`：编码文本列表
- `image_text_similarity(image: Any, texts: List[str]) -> np.ndarray`：计算图像与文本的相似度
- `zero_shot_classify(image: Any, candidate_labels: List[str]) -> List[float]`：零样本分类
- `filter_detections_by_text(image: np.ndarray, detections: List[Any], text_description: str, threshold: float = 0.5) -> List[Any]`：根据文本过滤检测结果
- `clear_cache(cache_type: Optional[str] = None) -> None`：清除缓存
- `get_cache_status() -> Dict[str, int]`：获取缓存状态
- `cleanup() -> None`：清理资源

## 姿态估计器

### PoseEstimator

姿态估计器，支持YOLO Pose和MediaPipe Pose模型。

```python
PoseEstimator(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_path`：模型文件路径或模型名称（例如，"yolov8n-pose.pt"）
  - `model_type`：模型类型（"yolo_pose"、"mediapipe"）
  - `conf_threshold`：姿态检测的置信度阈值
  - `keypoint_threshold`：关键点的置信度阈值
  - `device`：运行推理的设备（"auto"、"cuda"、"cpu"）
  - `min_detection_confidence`：MediaPipe特定，初始检测的最小置信度
  - `min_tracking_confidence`：MediaPipe特定，跟踪的最小置信度

**方法：**
- `estimate(image: np.ndarray) -> List[Pose]`：估计图像中的姿态
- `process(image: np.ndarray) -> List[Pose]`：处理图像，估计姿态
- `initialize() -> bool`：初始化姿态估计器

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

### SAM分割示例

```python
from visionframework.core.detector import Detector
import cv2

# 初始化带SAM分割器的检测器
detector = Detector({
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.3,
    "segmenter_type": "sam",
    "sam_model_type": "vit_b",
    "device": "cuda"
})
detector.initialize()

# 加载图像
image = cv2.imread("test.jpg")

# 检测+分割联合推理
detections = detector.detect(image)
print(f"检测到 {len(detections)} 个目标，其中 {sum(1 for d in detections if hasattr(d, 'mask') and d.mask is not None)} 个带有分割掩码")

# 绘制带有掩码的检测结果
result = image.copy()
for det in detections:
    if hasattr(det, 'mask') and det.mask is not None:
        # 绘制掩码
        mask = det.mask.astype(np.uint8) * 255
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 128] = (0, 255, 0)
        result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
    
    # 绘制边界框和标签
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(result, f"{det.class_name}: {det.confidence:.2f}", 
                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存结果
cv2.imwrite("sam_segmentation_result.jpg", result)
```

### CLIP零样本分类示例

```python
from visionframework.core.clip import CLIPExtractor
import cv2

# 初始化CLIP提取器
clip = CLIPExtractor({
    "model_name": "OFA-Sys/chinese-clip-vit-base-patch16",
    "device": "cuda"
})
clip.initialize()

# 加载图像
image = cv2.imread("test.jpg")

# 零样本分类
candidate_labels = ["猫", "狗", "汽车", "人", "自行车"]
scores = clip.zero_shot_classify(image, candidate_labels)

# 打印结果
for label, score in zip(candidate_labels, scores):
    print(f"{label}: {score:.4f}")
```

### 姿态估计示例

```python
from visionframework.core.pose_estimator import PoseEstimator
import cv2

# 初始化姿态估计器（使用MediaPipe）
pose_estimator = PoseEstimator({
    "model_type": "mediapipe",
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
})
pose_estimator.initialize()

# 加载图像
image = cv2.imread("person.jpg")

# 估计姿态
poses = pose_estimator.estimate(image)
print(f"检测到 {len(poses)} 个姿态")

# 绘制姿态关键点和骨架
result = image.copy()
for pose in poses:
    # 绘制关键点
    for kp in pose.keypoints:
        cv2.circle(result, (int(kp.x), int(kp.y)), 5, (0, 255, 0), -1)
        cv2.putText(result, f"{kp.keypoint_name[:2]}", 
                    (int(kp.x)+10, int(kp.y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# 保存结果
cv2.imwrite("pose_estimation_result.jpg", result)
```
