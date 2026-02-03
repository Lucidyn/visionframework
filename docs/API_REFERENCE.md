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

#### `process_batch(images: List[np.ndarray], max_batch_size: Optional[int] = None, use_parallel: bool = False, max_workers: Optional[int] = None, enable_memory_optimization: bool = True) -> List[Dict[str, Any]]`
批量处理多张图像，支持自动分块以优化内存使用。

**参数：**
- `images`：要处理的图像列表（BGR格式，OpenCV标准）
- `max_batch_size`：最大批处理大小，超过此大小会自动分块处理，默认为None（自动根据图像数量调整）
- `use_parallel`：是否使用并行处理，适用于独立图像处理场景
- `max_workers`：并行处理的最大工作线程数，默认为None（自动选择）
- `enable_memory_optimization`：是否启用内存优化技术，包括内存池和垃圾回收优化

**返回值：**
- `List[Dict[str, Any]]`：处理结果列表，每个元素为包含以下键的字典：
  - `detections`：检测结果列表（`List[Detection]`）
  - `tracks`：跟踪结果列表（`List[Track]`，如果启用跟踪）
  - `poses`：姿态估计结果列表（`List[Pose]`，如果启用姿态估计）
  - `frame_idx`：帧索引（int）
  - `processing_time`：处理时间（秒，float）

**示例：**
```python
# 批量处理图像
results = pipeline.process_batch(
    images,
    max_batch_size=4,  # 每批处理4张图像
    use_parallel=True,  # 启用并行处理
    max_workers=4,  # 使用4个工作线程
    enable_memory_optimization=True  # 启用内存优化
)
```

#### `process_video(input_source: Union[str, int], output_path: Optional[str] = None, use_pyav: bool = False) -> bool`
处理视频文件、摄像头或视频流。

**参数：**
- `input_source`：视频源，可以是：
  - 视频文件路径（字符串）
  - RTSP/HTTP流URL（字符串）
  - 摄像头索引（整数，0为默认摄像头）
- `output_path`：输出视频文件路径（可选），如果为None则不保存视频
- `use_pyav`：是否使用PyAV进行视频处理（默认False，使用OpenCV）
  - 注意：PyAV支持视频文件和RTSP/HTTP流，但不支持摄像头
  - PyAV提供比OpenCV更高的视频处理性能

**返回值：**
- `bool`：处理是否成功完成

**示例：**
```python
# 处理视频文件
pipeline.process_video("input.mp4", "output.mp4")

# 处理RTSP流（使用PyAV）
pipeline.process_video("rtsp://example.com/stream", "output.mp4", use_pyav=True)

# 处理摄像头
pipeline.process_video(0, "output.mp4")
```

#### `process_video_batch(input_source: Union[str, int], output_path: Optional[str] = None, start_frame: int = 0, end_frame: Optional[int] = None, batch_size: int = 8, skip_frames: int = 0, frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None, progress_callback: Optional[Callable[[float, int, int], None]] = None, use_pyav: bool = False) -> bool`
使用批量处理来处理视频，以提高性能。该方法将视频帧分批处理，充分利用GPU的批处理能力。

**参数：**
- `input_source`：视频源，可以是：
  - 视频文件路径（字符串）
  - RTSP/HTTP流URL（字符串）
  - 摄像头索引（整数，0为默认摄像头）
- `output_path`：输出视频文件路径（可选），如果为None则不保存视频
- `start_frame`：开始处理的帧号（默认0）
- `end_frame`：结束处理的帧号（None表示处理到结尾，仅对视频文件有效）
- `batch_size`：每批处理的帧数（默认8），较大的批次可以提高GPU利用率但需要更多内存
- `skip_frames`：跳过的帧数（默认0），用于降低处理帧率
- `frame_callback`：可选的帧回调函数 `(frame, frame_number, results) -> processed_frame`，用于自定义帧处理
- `progress_callback`：可选的进度回调函数 `(progress, current_frame, total_frames) -> None`，用于显示处理进度
- `use_pyav`：是否使用PyAV进行视频处理（默认False，使用OpenCV）
  - 注意：PyAV支持视频文件和RTSP/HTTP流，但不支持摄像头
  - PyAV提供比OpenCV更高的视频处理性能

**返回值：**
- `bool`：处理是否成功完成

**示例：**
```python
# 批量处理视频文件
pipeline.process_video_batch(
    "input.mp4", 
    "output.mp4",
    batch_size=16,  # 每批处理16帧
    use_pyav=True  # 使用PyAV后端
)

# 处理RTSP流并显示进度
def progress_cb(progress, current, total):
    print(f"进度: {progress:.1%} ({current}/{total})")

pipeline.process_video_batch(
    "rtsp://example.com/stream",
    "output.mp4",
    batch_size=8,
    progress_callback=progress_cb,
    use_pyav=True
)
```

### 静态方法

#### `VisionPipeline.process_image(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
无需初始化管道即可快速处理图像。这是一个静态方法，适合一次性处理。

**参数：**
- `image`：输入图像（numpy数组，BGR格式）
- `config`：配置字典（可选），包含检测器、跟踪器等配置

**返回值：**
- `Dict[str, Any]`：处理结果，包含：
  - `detections`：检测结果列表
  - `tracks`：跟踪结果列表（如果启用跟踪）
  - `poses`：姿态估计结果列表（如果启用姿态估计）

**示例：**
```python
import cv2
from visionframework import VisionPipeline

image = cv2.imread("test.jpg")
results = VisionPipeline.process_image(
    image,
    {
        "detector_config": {"model_path": "yolov8n.pt"},
        "enable_tracking": True
    }
)
```

#### `VisionPipeline.run_video(input_source: Union[str, int], output_path: Optional[str] = None, model_path: str = "yolov8n.pt", enable_tracking: bool = False, enable_segmentation: bool = False, enable_pose_estimation: bool = False, conf_threshold: float = 0.25, batch_size: int = 0, use_pyav: bool = False, **kwargs) -> bool`
使用简单API运行视频处理。这是一个静态方法，无需创建管道实例即可使用。

**参数：**
- `input_source`：视频源（视频文件路径、RTSP/HTTP流URL或摄像头索引）
- `output_path`：输出视频文件路径（可选）
- `model_path`：检测模型路径（默认"yolov8n.pt"）
- `enable_tracking`：是否启用跟踪（默认False）
- `enable_segmentation`：是否启用分割（默认False）
- `enable_pose_estimation`：是否启用姿态估计（默认False）
- `conf_threshold`：置信度阈值（默认0.25）
- `batch_size`：批处理大小（0表示不使用批处理，默认0）
- `use_pyav`：是否使用PyAV进行视频处理（默认False，使用OpenCV）
  - 注意：PyAV支持视频文件和RTSP/HTTP流，但不支持摄像头
- `**kwargs`：其他配置参数

**返回值：**
- `bool`：处理是否成功完成

**示例：**
```python
# 简单视频处理
VisionPipeline.run_video("input.mp4", "output.mp4", model_path="yolov8n.pt")

# 带跟踪和批处理
VisionPipeline.run_video(
    "input.mp4", 
    "output.mp4",
    model_path="yolov8n.pt",
    enable_tracking=True,
    batch_size=8,
    use_pyav=True
)
```

### 类方法

#### `VisionPipeline.with_tracking(config: Optional[Dict[str, Any]] = None) -> VisionPipeline`
创建一个启用了跟踪功能的管道。

#### `VisionPipeline.from_model(model_path: str, enable_tracking: bool = False, conf_threshold: float = 0.25) -> VisionPipeline`
从特定模型创建管道。

## 检测器

### BaseDetector

统一检测器接口，支持多种检测模型和SAM分割器集成。

```python
BaseDetector(
    config: Optional[Dict[str, Any]] = None
)
```

**参数：**
- `config`：配置字典，包含以下可选键：
  - `model_path`：模型文件路径或模型名称（例如，"yolov8n.pt"、"yolov26n.pt"）
  - `model_type`：模型类型（"yolo"、"detr"、"rfdetr"、"efficientdet"、"faster_rcnn"）
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

基于ReID（Person Re-Identification）的跟踪器，使用外观特征来提高跨帧目标跟踪的准确性。

```python
ReidTracker(
    max_age: int = 30,
    iou_threshold: float = 0.3,
    min_hits: int = 3,
    reid_model: str = "osnet_x0_25_market1501.pt",
    reid_threshold: float = 0.5
)
```

**参数：**
- `max_age`：目标消失后保持跟踪ID的最大帧数
- `iou_threshold`：IoU阈值，用于匹配检测结果和跟踪目标
- `min_hits`：开始跟踪前需要连续检测到目标的最小次数
- `reid_model`：ReID模型路径或名称
- `reid_threshold`：ReID特征匹配的阈值，超过此阈值认为是同一目标

**方法：**
- `update(detections: List[Dict[str, Any]], frame: np.ndarray = None) -> List[Dict[str, Any]]`：使用新的检测结果和ReID特征更新跟踪
- `reset() -> None`：重置跟踪器
- `extract_features(bbox: Tuple[int, int, int, int], frame: np.ndarray) -> np.ndarray`：从目标区域提取ReID特征

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

性能监控器，用于监控和分析性能指标。支持多种监控模式，包括GPU、磁盘I/O和网络I/O监控。

```python
PerformanceMonitor(
    window_size: int = 30,
    enabled_metrics: Optional[List[str]] = None,
    enable_gpu_monitoring: bool = False,
    enable_disk_monitoring: bool = False,
    enable_network_monitoring: bool = False
)
```

**参数：**
- `window_size`：滑动窗口大小，用于计算 FPS 和其他指标，默认为 30
- `enabled_metrics`：要启用的指标列表（None表示启用所有）
- `enable_gpu_monitoring`：是否启用GPU监控（默认False）
- `enable_disk_monitoring`：是否启用磁盘I/O监控（默认False）
- `enable_network_monitoring`：是否启用网络I/O监控（默认False）

**方法：**
- `start()`：开始性能监控
- `tick()`：记录一帧处理完成
- `record_component_time(component: str, elapsed: float)`：记录组件处理时间
  - `component`：组件名称（"detection"、"tracking"、"visualization"、"pose_estimation"、"clip_extraction"、"reid"、"batch_processing"、"memory_management"、"io_operations"、"network_operations"）
  - `elapsed`：处理时间（秒）
- `get_metrics() -> PerformanceMetrics`：获取综合性能指标
- `get_detailed_report() -> Dict[str, Any]`：获取详细的性能报告
- `reset()`：重置性能监控器

**示例：**
```python
from visionframework import PerformanceMonitor

# 创建性能监控器
monitor = PerformanceMonitor(
    window_size=60,
    enable_gpu_monitoring=True,
    enable_disk_monitoring=True
)

monitor.start()
# ... 处理代码 ...
monitor.tick()
monitor.record_component_time("detection", 0.05)

# 获取指标
metrics = monitor.get_metrics()
print(f"FPS: {metrics.fps:.2f}")
print(f"平均帧时间: {metrics.avg_time_per_frame*1000:.2f}ms")

# 获取详细报告
report = monitor.get_detailed_report()
print(f"GPU内存使用: {report['gpu']['memory_usage_mb']:.2f}MB")
```

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
    video_path: str or int
)
```

**参数：**
- `video_path`：视频文件路径、视频流URL或摄像头索引

**方法：**
- `open() -> bool`：打开视频文件或摄像头
- `read_frame() -> Tuple[bool, Optional[np.ndarray]]`：读取下一帧
- `get_frame(frame_number: int) -> Optional[np.ndarray]`：获取特定帧
- `get_info() -> Dict[str, Any]`：获取视频信息
- `close() -> None`：关闭视频文件或摄像头

#### `PyAVVideoProcessor`
基于FFmpeg的高性能视频处理器，提供比OpenCV更高的视频处理性能。

```python
PyAVVideoProcessor(
    video_path: str
)
```

**参数：**
- `video_path`：视频文件路径或RTSP/HTTP流URL

**方法：**
- `open() -> bool`：打开视频文件或流
- `read_frame() -> Tuple[bool, Optional[np.ndarray]]`：读取下一帧
- `get_frame(frame_number: int) -> Optional[np.ndarray]`：获取特定帧
- `get_info() -> Dict[str, Any]`：获取视频信息
- `close() -> None`：关闭视频文件或流

#### `VideoWriter`
```python
VideoWriter(
    output_path: str,
    fps: float = 30.0,
    frame_size: Optional[Tuple[int, int]] = None,
    fourcc: str = "mp4v"
)
```

**参数：**
- `output_path`：输出视频文件路径
- `fps`：帧率
- `frame_size`：帧大小（宽度，高度）
- `fourcc`：视频编码

**方法：**
- `open(frame_size: Optional[Tuple[int, int]] = None) -> bool`：打开视频写入器
- `write(frame: np.ndarray) -> bool`：写入帧
- `close() -> None`：关闭视频写入器

#### `PyAVVideoWriter`
基于FFmpeg的高性能视频写入器，提供比OpenCV更多的编码选项。

```python
PyAVVideoWriter(
    output_path: str,
    fps: float = 30.0,
    frame_size: Optional[Tuple[int, int]] = None,
    codec: str = "h264"
)
```

**参数：**
- `output_path`：输出视频文件路径
- `fps`：帧率
- `frame_size`：帧大小（宽度，高度）
- `codec`：视频编码

**方法：**
- `open(frame_size: Optional[Tuple[int, int]] = None) -> bool`：打开视频写入器
- `write(frame: np.ndarray) -> bool`：写入帧
- `close() -> None`：关闭视频写入器

#### `process_video(input_path: str or int, output_path: Optional[str] = None, frame_callback: Optional[Callable[[np.ndarray, int], np.ndarray]] = None, start_frame: int = 0, end_frame: Optional[int] = None, skip_frames: int = 0, use_pyav: bool = False) -> bool`
使用自定义处理函数处理视频。

**参数：**
- `input_path`：视频文件路径、视频流URL或摄像头索引
- `output_path`：输出视频文件路径
- `frame_callback`：帧处理函数
- `start_frame`：开始处理的帧号
- `end_frame`：结束处理的帧号
- `skip_frames`：跳过的帧数
- `use_pyav`：是否使用PyAV进行视频处理（默认False，使用OpenCV）
  - 注意：PyAV支持视频文件和RTSP/HTTP流，但不支持摄像头

### 内存管理工具

#### `MemoryManager` 和 `MemoryPool`
内存管理器，用于高效内存分配和重用。支持全局内存池和多内存池管理。

```python
from visionframework.utils.memory import MemoryManager

# 获取全局内存池
memory_pool = MemoryManager.get_global_memory_pool()
memory_pool.initialize(
    min_blocks=4,  # 最小内存块数量
    block_size=(480, 640, 3),  # 每个内存块的形状
    max_blocks=10  # 最大内存块数量
)
```

**MemoryPool 方法：**
- `initialize(min_blocks: int, block_size: Tuple[int, ...], max_blocks: int) -> None`：初始化内存池
- `acquire() -> np.ndarray`：从池中获取内存块
- `release(block: np.ndarray) -> None`：将内存块返回池
- `clear() -> None`：清空池中的内存块（保留至少min_blocks）
- `resize(new_max_blocks: int) -> None`：调整池大小
- `get_status() -> Dict[str, Any]`：获取池状态
- `get_stats() -> Dict[str, Any]`：获取池统计信息
- `optimize() -> None`：优化内存池

#### 内存管理函数

- `create_memory_pool(pool_name: str, block_shape: Tuple[int, ...], dtype: np.dtype = np.uint8, max_blocks: int = 10, min_blocks: int = 0) -> MemoryPool`：创建内存池
- `acquire_memory(pool_name: str) -> Optional[np.ndarray]`：从池获取内存
- `release_memory(pool_name: str, block: np.ndarray) -> None`：释放内存到池
- `resize_memory_pool(pool_name: str, new_max_blocks: int) -> None`：调整内存池大小
- `optimize_memory_usage() -> Dict[str, Any]`：优化内存使用

**示例：**
```python
from visionframework.utils.memory import (
    MemoryManager, create_memory_pool, 
    acquire_memory, release_memory
)
import numpy as np

# 方法1：使用全局内存池
pool = MemoryManager.get_global_memory_pool()
pool.initialize(min_blocks=4, block_size=(480, 640, 3), max_blocks=10)
memory = pool.acquire()
# 使用内存...
pool.release(memory)

# 方法2：使用命名内存池
create_memory_pool("detection_pool", (640, 640, 3), max_blocks=10)
memory = acquire_memory("detection_pool")
# 使用内存...
release_memory("detection_pool", memory)
```

### 配置工具

#### `Config` 类
配置管理器，支持配置加载、保存和继承。

**方法：**
- `load_from_file(file_path: str, return_default_if_not_found: bool = False, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`：从文件加载配置
- `save_to_file(config: Dict[str, Any], file_path: str, format: Optional[str] = None)`：将配置保存到文件
- `load_with_inheritance(base_file: str, *override_files: str) -> Dict[str, Any]`：加载配置并支持继承
- `load_model_with_inheritance(base_file: str, model_type, *override_files: str) -> BaseConfig`：加载配置并解析为模型
- `compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]`：比较两个配置

#### `BaseConfig` 类
基础配置模型，支持配置继承和验证。

**方法：**
- `from_parent(parent_config: Optional[BaseConfig] = None, **overrides) -> BaseConfig`：从父配置创建新配置
- `merge(other_config: BaseConfig) -> BaseConfig`：合并另一个配置
- `validate_config() -> bool`：验证配置
- `get_nested(key_path: str, default: Any = None) -> Any`：获取嵌套配置值
- `set_nested(key_path: str, value: Any) -> BaseConfig`：设置嵌套配置值

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

**参数：**
- `cache_dir`：模型缓存目录，默认为None（使用默认缓存目录）

**方法：**
- `get_model_path(model_name: str) -> str`：获取缓存模型的路径
- `cache_model(model_path: str, model_name: str) -> str`：缓存模型
- `clear_cache() -> None`：清除缓存
- `get_cache_info() -> Dict[str, Any]`：获取缓存信息
- `delete_model(model_name: str) -> bool`：删除指定模型

**支持的模型类型：**
- YOLO系列模型（yolov8n.pt, yolov26n.pt等）
- CLIP模型（openai/clip-vit-base-patch32等）
- DETR模型
- 姿态估计模型（yolov8n-pose.pt等）
- ReID模型（osnet_x0_25_market1501.pt等）

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

## 插件系统

### PluginRegistry
插件注册表，用于管理和发现插件组件。

```python
PluginRegistry()
```

**方法：**
- `register_detector(name: str, detector_class: Type, **metadata)`：注册自定义检测器
- `register_tracker(name: str, tracker_class: Type, **metadata)`：注册自定义跟踪器
- `register_segmenter(name: str, segmenter_class: Type, **metadata)`：注册自定义分割器
- `register_model(name: str, model_loader: Callable, **metadata)`：注册自定义模型
- `register_processor(name: str, processor_class: Type, **metadata)`：注册自定义处理器
- `register_visualizer(name: str, visualizer_class: Type, **metadata)`：注册自定义可视化器
- `register_evaluator(name: str, evaluator_class: Type, **metadata)`：注册自定义评估器
- `register_custom_component(name: str, component: Any, **metadata)`：注册自定义组件
- `get_detector(name: str) -> Optional[Dict[str, Any]]`：获取注册的检测器
- `get_tracker(name: str) -> Optional[Dict[str, Any]]`：获取注册的跟踪器
- `get_segmenter(name: str) -> Optional[Dict[str, Any]]`：获取注册的分割器
- `get_model(name: str) -> Optional[Dict[str, Any]]`：获取注册的模型
- `get_processor(name: str) -> Optional[Dict[str, Any]]`：获取注册的处理器
- `get_visualizer(name: str) -> Optional[Dict[str, Any]]`：获取注册的可视化器
- `get_evaluator(name: str) -> Optional[Dict[str, Any]]`：获取注册的评估器
- `get_custom_component(name: str) -> Optional[Dict[str, Any]]`：获取注册的自定义组件
- `list_detectors() -> List[str]`：列出所有注册的检测器
- `list_trackers() -> List[str]`：列出所有注册的跟踪器
- `list_segmenters() -> List[str]`：列出所有注册的分割器
- `list_models() -> List[str]`：列出所有注册的模型
- `list_processors() -> List[str]`：列出所有注册的处理器
- `list_visualizers() -> List[str]`：列出所有注册的可视化器
- `list_evaluators() -> List[str]`：列出所有注册的评估器
- `list_custom_components() -> List[str]`：列出所有注册的自定义组件
- `add_plugin_path(path: str)`：添加插件路径
- `load_plugins_from_path(path: str)`：从路径加载插件
- `load_all_plugins()`：加载所有插件

### ModelRegistry
模型注册表，用于管理模型加载和缓存。

```python
ModelRegistry()
```

**方法：**
- `register_model(name: str, model_info: Dict[str, Any])`：注册模型
- `get_model(name: str) -> Optional[Dict[str, Any]]`：获取模型信息
- `load_model(name: str, **kwargs) -> Any`：加载模型
- `unload_model(name: str)`：卸载模型
- `list_models() -> List[str]`：列出所有注册的模型
- `clear_cache()`：清除模型缓存

### 插件注册装饰器

**register_detector(name: str, **metadata)**
注册检测器类的装饰器。

**register_tracker(name: str, **metadata)**
注册跟踪器类的装饰器。

**register_segmenter(name: str, **metadata)**
注册分割器类的装饰器。

**register_model(name: str, **metadata)**
注册模型加载函数的装饰器。

**register_processor(name: str, **metadata)**
注册处理器类的装饰器。

**register_visualizer(name: str, **metadata)**
注册可视化器类的装饰器。

**register_evaluator(name: str, **metadata)**
注册评估器类的装饰器。

**register_custom_component(name: str, **metadata)**
注册自定义组件的装饰器。

## 依赖管理

### DependencyManager
依赖管理器，用于管理可选依赖。

```python
DependencyManager()
```

**方法：**
- `is_available(dependency: str) -> bool`：检查依赖是否可用
- `import_dependency(dependency: str, package: str) -> Optional[ModuleType]`：导入依赖
- `get_dependency_info(dependency: str) -> Optional[Dict[str, Any]]`：获取依赖信息
- `get_install_command(dependency: str) -> Optional[str]`：获取安装命令
- `get_dependency_status(dependency: str) -> Dict[str, Union[bool, str]]`：获取依赖状态
- `get_all_dependency_status() -> Dict[str, Dict[str, Union[bool, str]]]`：获取所有依赖状态
- `get_available_dependencies() -> List[str]`：获取可用的依赖
- `get_missing_dependencies() -> List[str]`：获取缺失的依赖
- `validate_dependency(dependency: str) -> bool`：验证依赖是否可用

### 依赖管理函数

**is_dependency_available(dependency: str) -> bool**
检查依赖是否可用。

**import_optional_dependency(dependency: str, package: str) -> Optional[ModuleType]**
导入可选依赖。

**get_available_dependencies() -> List[str]**
获取可用的依赖。

**get_missing_dependencies() -> List[str]**
获取缺失的依赖。

**validate_dependency(dependency: str) -> bool**
验证依赖是否可用。

**get_install_command(dependency: str) -> Optional[str]**
获取依赖的安装命令。

## 错误处理

### ErrorHandler
错误处理器，用于统一错误处理。

```python
ErrorHandler()
```

**方法：**
- `handle_error(error: Exception, error_type: type, message: str, context: Optional[Dict[str, Any]] = None, raise_error: bool = False, log_traceback: bool = True) -> Optional[Exception]`：处理错误
- `wrap_error(func, error_type: type, message: str, context: Optional[Dict[str, Any]] = None, default_return=None, raise_error: bool = False)`：包装函数，捕获和处理错误
- `validate_input(input_value: Any, expected_type: type, param_name: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]`：验证输入参数
- `format_error_message(message: str, error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> str`：格式化错误消息

## 模型优化工具

### 量化 (Quantization)

#### `quantize_model(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module`
量化PyTorch模型以减少模型大小和提高推理速度。

**参数：**
- `model`: 要量化的PyTorch模型
- `config`: 量化配置（`QuantizationConfig`）
  - `quantization_type`: 量化类型（"dynamic"、"static"、"aware"）
  - `backend`: 量化后端（"qnnpack"、"fbgemm"、"onednn"）
  - `dtype`: 量化数据类型（`torch.quint8`、`torch.qint8`）

**示例：**
```python
from visionframework.utils.model_optimization import quantize_model, QuantizationConfig
import torch

config = QuantizationConfig(
    quantization_type="dynamic",
    backend="fbgemm"
)
quantized_model = quantize_model(model, config)
```

### 剪枝 (Pruning)

#### `prune_model(model: nn.Module, config: PruningConfig) -> nn.Module`
剪枝PyTorch模型以减少模型大小。

**参数：**
- `model`: 要剪枝的PyTorch模型
- `config`: 剪枝配置（`PruningConfig`）
  - `pruning_type`: 剪枝类型（"l1_unstructured"、"l2_unstructured"、"random_unstructured"、"ln_structured"）
  - `amount`: 剪枝比例（0.0到1.0）
  - `target_modules`: 要剪枝的模块类型列表

**示例：**
```python
from visionframework.utils.model_optimization import prune_model, PruningConfig
import torch.nn as nn

config = PruningConfig(
    pruning_type="l1_unstructured",
    amount=0.2,
    target_modules=[nn.Linear, nn.Conv2d]
)
pruned_model = prune_model(model, config)
```

### 知识蒸馏 (Distillation)

#### `distill_model(teacher_model: nn.Module, student_model: nn.Module, train_data: DataLoader, config: DistillationConfig) -> nn.Module`
从教师模型向学生模型进行知识蒸馏。

**参数：**
- `teacher_model`: 教师模型
- `student_model`: 学生模型
- `train_data`: 训练数据加载器
- `config`: 蒸馏配置（`DistillationConfig`）
  - `temperature`: 软化logits的温度
  - `alpha`: 蒸馏损失的权重
  - `epochs`: 蒸馏轮数

**示例：**
```python
from visionframework.utils.model_optimization import distill_model, DistillationConfig
from torch.utils.data import DataLoader

config = DistillationConfig(
    temperature=3.0,
    alpha=0.7,
    epochs=10
)
distilled_model = distill_model(teacher_model, student_model, train_loader, config)
```

## 模型训练工具

### ModelFineTuner

模型微调器，用于在自定义数据集上微调模型。

```python
from visionframework.utils.model_training import ModelFineTuner, FineTuningConfig

config = FineTuningConfig(
    strategy="freeze",  # 或 "full", "lora", "qlora"
    epochs=10,
    batch_size=32,
    learning_rate=1e-4
)

fine_tuner = ModelFineTuner(config)
fine_tuned_model = fine_tuner.fine_tune(
    model=base_model,
    train_data=train_loader,
    val_data=val_loader
)
```

## 模型转换工具

### ModelConverter

模型转换器，用于在不同模型格式之间转换。

```python
from visionframework.utils.model_conversion import ModelConverter, ConversionConfig, ModelFormat

config = ConversionConfig(
    input_format=ModelFormat.PYTORCH,
    output_format=ModelFormat.ONNX,
    input_path="model.pth",
    output_path="model.onnx"
)

converter = ModelConverter()
converter.convert(config)
```

## 模型部署工具

### ModelDeployer

模型部署器，用于将模型部署到不同平台。

```python
from visionframework.utils.model_deployment import ModelDeployer, DeploymentConfig, DeploymentPlatform

config = DeploymentConfig(
    platform=DeploymentPlatform.TENSORRT,
    model_path="model.onnx",
    model_format="onnx",
    output_path="deployed_model"
)

deployer = ModelDeployer()
deployer.deploy(config)
```

## 模型管理工具

### AutoSelector

自动模型选择器，根据硬件配置和任务需求自动选择最合适的模型。

```python
from visionframework.utils.model_management import AutoSelector, ModelRequirement, HardwareTier

requirement = ModelRequirement(
    model_type="detection",
    accuracy=80,
    speed=70,
    memory=1000
)

selector = AutoSelector()
recommended_model = selector.select_model(requirement, HardwareTier.MID_RANGE)
```

## 多模态融合工具

### MultimodalFusion

多模态融合器，用于融合不同模态的信息。

```python
from visionframework.utils.multimodal import MultimodalFusion, FusionConfig, FusionType

config = FusionConfig(
    fusion_type=FusionType.ATTENTION,
    input_dims=[512, 768],  # 视觉和文本特征维度
    hidden_dim=256,
    output_dim=512
)

fusion = MultimodalFusion(config)
fused_features = fusion.forward([vision_features, text_features])
```

## 数据增强工具

### ImageAugmenter

图像增强器，用于训练数据的增强。

```python
from visionframework.utils.data_augmentation import ImageAugmenter, AugmentationConfig, AugmentationType

config = AugmentationConfig(
    augmentations=[
        AugmentationType.FLIP,
        AugmentationType.ROTATE,
        AugmentationType.BRIGHTNESS,
        AugmentationType.CONTRAST
    ],
    probability=0.5
)

augmenter = ImageAugmenter(config)
augmented_image = augmenter.augment(
    image,
    angle=15,  # 旋转角度
    brightness_factor=1.0,  # 亮度因子
    contrast_factor=1.0  # 对比度因子
)
```

## 轨迹分析工具

### TrajectoryAnalyzer

轨迹分析器，用于分析目标轨迹。

```python
from visionframework.utils import TrajectoryAnalyzer
from visionframework import Track

analyzer = TrajectoryAnalyzer(fps=30.0, pixel_to_meter=0.1)

# 计算速度
speed_x, speed_y = analyzer.calculate_speed(track, use_real_world=True)

# 计算方向
direction = analyzer.calculate_direction(track)

# 预测未来位置
future_pos = analyzer.predict_future_position(track, frames_ahead=10)

# 计算轨迹统计
stats = analyzer.analyze_track(track)
```

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

### VideoProcessingError
视频处理错误的基类。

### VideoReaderError
视频读取错误。

### VideoWriterError
视频写入错误。

### SegmenterInitializationError
分割器初始化错误。

### SegmentationError
分割错误。

### ConcurrentProcessingError
并发处理错误。

### MemoryAllocationError
内存分配错误。

### TimeoutError
超时错误。

### ROIProcessingError
ROI处理错误。

### BatchProcessingError
批量处理错误。

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
from visionframework import VisionPipeline
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
from visionframework import VisionPipeline

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
from visionframework import VisionPipeline
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
from visionframework import YOLODetector
import cv2

# 初始化带SAM分割器的检测器
detector = YOLODetector({
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.3,
    "device": "cuda"
})
detector.initialize()

# 加载图像
image = cv2.imread("test.jpg")

# 检测+分割联合推理
detections = detector.detect(image)
print(f"检测到 {len(detections)} 个目标")

# 绘制检测结果
result = image.copy()
for det in detections:
    # 绘制边界框和标签
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(result, f"{det.class_name}: {det.confidence:.2f}", 
                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存结果
cv2.imwrite("detection_result.jpg", result)
```

### CLIP零样本分类示例

```python
from visionframework import CLIPExtractor
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
from visionframework import PoseEstimator
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

### 批量处理优化示例

```python
from visionframework import VisionPipeline
import cv2
import numpy as np

# 初始化管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.5,
        "batch_inference": True  # 启用批量推理
    }
})

# 加载多张图像
images = []
for i in range(10):
    # 假设我们有10张测试图像
    image = cv2.imread(f"test_{i}.jpg")
    if image is not None:
        images.append(image)

# 批量处理，启用并行处理和内存优化
results = pipeline.process_batch(
    images,
    max_batch_size=4,  # 最大批处理大小
    use_parallel=True,  # 启用并行处理
    max_workers=4,      # 4个工作线程
    enable_memory_optimization=True  # 启用内存优化
)

# 处理结果
for i, result in enumerate(results):
    print(f"图像 {i+1} 检测到 {len(result['detections'])} 个目标，处理时间: {result['processing_time']:.4f}秒")

# 内存池使用示例
from visionframework.utils.memory import create_memory_pool, acquire_memory, release_memory

# 创建内存池
pool = create_memory_pool(
    "detection_pool",
    block_shape=(640, 640, 3),
    dtype=np.uint8,
    max_blocks=10,
    min_blocks=2
)

# 使用内存池
for image in images:
    # 从池获取内存
    memory_block = acquire_memory("detection_pool")
    
    # 处理图像
    memory_block[:] = image[:640, :640, :]  # 假设图像大小合适
    
    # 释放内存回池
    release_memory("detection_pool", memory_block)
```

### ReID跟踪示例

```python
from visionframework import VisionPipeline
import cv2

# 初始化带ReID跟踪的管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.5
    },
    "tracker_config": {
        "tracker_type": "reidtracker",
        "reid_model": "osnet_x0_25_market1501.pt",
        "reid_threshold": 0.5
    },
    "enable_tracking": True
})

# 处理视频
pipeline.process_video(
    input_path="input.mp4",
    output_path="output_reid.mp4"
)
```

### PyAV视频处理示例

```python
from visionframework import VisionPipeline

# 初始化管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.3
    },
    "enable_tracking": True
})

# 使用PyAV处理视频（高性能）
pipeline.process_video(
    input_path="input.mp4",
    output_path="output_pyav.mp4",
    use_pyav=True  # 启用PyAV后端
)

# 使用简化API
from visionframework import VisionPipeline

# 使用静态方法快速处理视频（带PyAV）
VisionPipeline.run_video(
    input_source="input.mp4",
    output_path="output_simple.mp4",
    model_path="yolov8n.pt",
    enable_tracking=True,
    use_pyav=True  # 启用PyAV后端
)

# 直接使用PyAVVideoProcessor
from visionframework.utils.io import PyAVVideoProcessor, PyAVVideoWriter
import cv2

# 读取视频
with PyAVVideoProcessor("input.mp4") as processor:
    info = processor.get_info()
    print(f"视频信息: {info}")
    
    # 创建写入器
    with PyAVVideoWriter("output_direct.mp4", fps=info["fps"], frame_size=(info["width"], info["height"])) as writer:
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            
            # 简单处理
            cv2.putText(frame, "Processed with PyAV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 写入帧
            writer.write(frame)
```
