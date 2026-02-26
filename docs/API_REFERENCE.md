# API 参考文档

> **v0.4.0** — 所有组件均可直接从 `visionframework` 导入，无需记忆内部模块路径。

## Vision 类

`Vision` 是整个框架的唯一入口。所有功能通过这一个类访问。

```python
from visionframework import Vision
```

### 构造函数

```python
Vision(
    model: str = "yolov8n.pt",
    model_type: str = "yolo",
    device: str = "auto",
    conf: float = 0.25,
    iou: float = 0.45,
    track: bool = False,
    tracker: str = "bytetrack",
    segment: bool = False,
    pose: bool = False,
    fp16: bool = False,
    batch_inference: bool = False,
    dynamic_batch: bool = False,
    max_batch_size: int = 8,
    category_thresholds: dict = None,
    **extra
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | `"yolov8n.pt"` | 模型路径或名称 |
| `model_type` | str | `"yolo"` | 检测器后端: `"yolo"` / `"detr"` / `"rfdetr"` |
| `device` | str | `"auto"` | 推理设备: `"auto"` / `"cpu"` / `"cuda"` / `"cuda:0"` |
| `conf` | float | `0.25` | 全局置信度阈值 |
| `iou` | float | `0.45` | NMS IoU 阈值 |
| `track` | bool | `False` | 是否开启多目标跟踪 |
| `tracker` | str | `"bytetrack"` | 跟踪器类型: `"bytetrack"` / `"ioutracker"` / `"reidtracker"` |
| `segment` | bool | `False` | 是否开启实例分割 |
| `pose` | bool | `False` | 是否开启姿态估计 |
| `fp16` | bool | `False` | FP16 半精度推理 (仅 CUDA，提升速度) |
| `batch_inference` | bool | `False` | 启用批量推理 |
| `dynamic_batch` | bool | `False` | 动态调整批量大小 |
| `max_batch_size` | int | `8` | 批量推理时的最大 batch 大小 |
| `category_thresholds` | dict/None | `None` | 按类别设置不同阈值, 如 `{"person": 0.5, "car": 0.3}` |
| `**extra` | - | - | 额外参数透传给检测器配置 |

**示例：**
```python
# 最简用法
v = Vision()

# 常见配置
v = Vision(
    model="yolov8s.pt",
    device="cuda",
    conf=0.3,
    track=True,
    fp16=True,
    category_thresholds={"person": 0.5, "car": 0.3}
)
```

### `Vision.from_config(path)`

从配置文件或字典创建 Vision 实例。

```python
# 从 JSON 文件
v = Vision.from_config("config.json")

# 从 YAML 文件
v = Vision.from_config("config.yaml")

# 从字典
v = Vision.from_config({"model": "yolov8n.pt", "track": True, "fp16": True})
```

**配置文件示例 (JSON)：**
```json
{
    "model": "yolov8s.pt",
    "device": "auto",
    "conf": 0.25,
    "iou": 0.45,
    "track": true,
    "tracker": "bytetrack",
    "segment": false,
    "pose": false,
    "fp16": true,
    "batch_inference": false,
    "category_thresholds": {
        "person": 0.5,
        "car": 0.3
    }
}
```

### `Vision.run(source, *, recursive, skip_frames, start_frame, end_frame)`

处理任意媒体源，返回迭代器。

```python
for frame, meta, result in v.run(source):
    # frame: np.ndarray (BGR)
    # meta:  dict (source_path, frame_index, is_video, ...)
    # result: dict (detections, tracks, poses)
    ...
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | str/int/list/ndarray/Path | - | 媒体源 |
| `recursive` | bool | `False` | 文件夹是否递归 |
| `skip_frames` | int | `0` | 视频跳帧数 |
| `start_frame` | int | `0` | 视频起始帧 |
| `end_frame` | int/None | `None` | 视频结束帧 |

**`source` 支持的类型：**

| 值 | 类型 |
|----|------|
| `"test.jpg"` | 图片文件 |
| `"video.mp4"` | 视频文件 |
| `0` | 摄像头 |
| `"rtsp://..."` | RTSP / HTTP 流 |
| `"folder/"` | 文件夹 |
| `["a.jpg", "b.mp4"]` | 混合列表 |
| `np.ndarray` | BGR 图像数组 |

**返回值 `result` 结构：**

| 键 | 类型 | 说明 |
|----|------|------|
| `"detections"` | `List[Detection]` | 检测结果列表 |
| `"tracks"` | `List[Track]` | 跟踪结果列表 (需 `track=True`) |
| `"poses"` | `List[Pose]` | 姿态结果列表 (需 `pose=True`) |
| `"counts"` | `dict` | ROI 计数结果 (需先调用 `add_roi()`) |

### `Vision.add_roi(name, points, roi_type="polygon")`

注册感兴趣区域，开启区域计数功能。返回 `self` 以支持链式调用。

```python
v = Vision(model="yolov8n.pt", track=True)
v.add_roi("entrance", [(100,100),(400,100),(400,400),(100,400)])
v.add_roi("exit",     [(500,100),(800,100),(800,400),(500,400)], roi_type="rectangle")

for frame, meta, result in v.run("video.mp4"):
    counts = result["counts"]
    # {"entrance": {"inside": 3, "entering": 1, "exiting": 0,
    #               "total_entered": 12, "total_exited": 9}}
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 区域名称 |
| `points` | list[(x,y)] | 多边形顶点（矩形用两个角点） |
| `roi_type` | str | `"polygon"` / `"rectangle"` / `"circle"` |

### `Vision.process_batch(images)`

批量处理 numpy 图像列表，返回结果列表。

```python
results = v.process_batch([img1, img2, img3])
for r in results:
    print(len(r["detections"]))
```

### `Vision.info()`

返回当前实例的配置摘要字典。

```python
print(v.info())
# {"model": "yolov8n.pt", "device": "cpu", "track": True,
#  "rois": ["entrance", "exit"], ...}
```

### `Vision.draw(frame, result, **kwargs)`

在帧上绘制检测/跟踪/姿态结果（`Visualizer.draw()` 的快捷方式）。

```python
annotated = v.draw(frame, result)
cv2.imshow("Result", annotated)
```

### `Vision.pipeline`

访问底层 `VisionPipeline` 实例，用于高级操作。

### `Vision.cleanup()`

释放模型资源和 GPU 显存。

---

## 数据结构

### Detection

检测结果对象。

| 属性 | 类型 | 说明 |
|------|------|------|
| `bbox` | `Tuple[float, float, float, float]` | 边界框 `(x1, y1, x2, y2)` |
| `confidence` | `float` | 置信度分数 |
| `class_id` | `int` | 类别 ID |
| `class_name` | `str` | 类别名称 |
| `color` | `Tuple[int, int, int]` | 可视化颜色 |

### Track

跟踪结果对象，继承 Detection 的全部属性。

| 属性 | 类型 | 说明 |
|------|------|------|
| `track_id` | `int` | 跟踪 ID |
| `history` | `List` | 轨迹历史 |

### Pose

姿态估计结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `keypoints` | `List[KeyPoint]` | 关键点列表 |
| `bbox` | `Tuple` | 人体边界框 |
| `confidence` | `float` | 置信度 |

### KeyPoint

单个关键点。

| 属性 | 类型 | 说明 |
|------|------|------|
| `x` | `float` | x 坐标 |
| `y` | `float` | y 坐标 |
| `confidence` | `float` | 关键点置信度 |
| `keypoint_name` | `str` | 关键点名称 |

---

## CLIPExtractor

CLIP 模型封装，支持图像-文本交互。

```python
from visionframework import CLIPExtractor

clip = CLIPExtractor(
    model_name="openai/clip-vit-base-patch32",
    device="cpu",
    use_fp16=False
)
clip.initialize()
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | `"openai/clip-vit-base-patch32"` | HuggingFace 模型名 |
| `device` | str | `"cpu"` | 推理设备 |
| `use_fp16` | bool | `False` | 是否使用 FP16 精度 |

**方法：**

| 方法 | 说明 |
|------|------|
| `initialize()` | 初始化模型和处理器 |
| `encode_image(image)` | 编码图像为归一化嵌入, 返回 `np.ndarray (N, D)` |
| `encode_text(texts)` | 编码文本列表为归一化嵌入, 返回 `np.ndarray (N, D)` |
| `image_text_similarity(image, texts)` | 计算图文相似度矩阵, 返回 `np.ndarray (1, T)` |
| `zero_shot_classify(image, labels)` | 零样本分类, 返回 `List[float]` |
| `extract(input_data)` | 通用接口, 自动判断图像/文本 |
| `cleanup()` | 释放资源 |

**示例：**
```python
from visionframework import CLIPExtractor

clip = CLIPExtractor(device="cuda", use_fp16=True)
clip.initialize()

# 零样本分类
scores = clip.zero_shot_classify(image, ["cat", "dog", "car"])
for label, score in zip(["cat", "dog", "car"], scores):
    print(f"{label}: {score:.4f}")

# 图文相似度
sim = clip.image_text_similarity(image, ["a photo of a cat", "a photo of a dog"])

# 批量图像编码
embeddings = clip.encode_image([img1, img2, img3])

clip.cleanup()
```

---

## Visualizer

统一可视化器，支持检测、跟踪和姿态估计结果的可视化。

```python
from visionframework import Visualizer

vis = Visualizer()
```

**方法：**

| 方法 | 说明 |
|------|------|
| `draw_detections(image, detections)` | 绘制检测结果 |
| `draw_tracks(image, tracks, draw_history=True)` | 绘制跟踪结果 |
| `draw_poses(image, poses)` | 绘制姿态结果 |
| `draw_results(image, detections, tracks, poses)` | 绘制所有结果 |
| `draw(image, result)` | 传入 `v.run()` 返回的 result dict，自动绘制 |
| `draw_heatmap(frame, tracks, *, alpha, radius, colormap, accumulate, _heat_state)` | 轨迹热力图叠加 |

**`draw_heatmap` 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `frame` | ndarray | - | BGR 图像 |
| `tracks` | list[Track] | - | 当前帧的跟踪结果 |
| `alpha` | float | `0.5` | 热力图混合权重 (0–1) |
| `radius` | int | `20` | 高斯热点半径（像素） |
| `colormap` | int | `cv2.COLORMAP_JET` | OpenCV 颜色映射 |
| `accumulate` | bool | `False` | 是否跨帧累积热力图 |
| `_heat_state` | dict\|None | `None` | 跨帧累积状态字典（传入同一个 dict 即可） |

```python
from visionframework import Visualizer
vis = Visualizer()

# 单帧热力图
heatmap = vis.draw_heatmap(frame, tracks)

# 累积热力图（跨帧）
state = {}
for frame, meta, result in v.run("video.mp4"):
    heatmap = vis.draw_heatmap(frame, result["tracks"],
                               accumulate=True, _heat_state=state)
```

---

## ResultExporter

结果导出器。

```python
from visionframework import ResultExporter

exporter = ResultExporter()
```

**方法：**

| 方法 | 说明 |
|------|------|
| `export_detections_to_json(detections, path)` | 导出检测结果为 JSON |
| `export_tracks_to_json(tracks, path)` | 导出跟踪结果为 JSON |
| `export_detections_to_csv(detections, path)` | 导出检测结果为 CSV |
| `export_tracks_to_csv(tracks, path)` | 导出跟踪结果为 CSV |
| `export_to_coco_format(detections, image_id, image_info, path)` | 导出为 COCO 格式 |

---

## 异常类

所有异常均可从 `visionframework` 直接导入。

| 异常 | 说明 |
|------|------|
| `VisionFrameworkError` | 所有错误的基类 |
| `DetectorInitializationError` | 检测器初始化失败 |
| `DetectorInferenceError` | 检测推理失败 |
| `TrackerInitializationError` | 跟踪器初始化失败 |
| `TrackerUpdateError` | 跟踪更新失败 |
| `ConfigurationError` | 配置错误 |
| `ModelNotFoundError` | 模型文件未找到 |
| `ModelLoadError` | 模型加载失败 |
| `DeviceError` | 设备错误 |
| `DependencyError` | 依赖缺失 |
| `DataFormatError` | 数据格式错误 |
| `ProcessingError` | 处理错误 |

---

## 使用示例

### 基本检测

```python
from visionframework import Vision
import cv2

v = Vision(model="yolov8n.pt", conf=0.5)
for frame, meta, result in v.run("test.jpg"):
    annotated = v.draw(frame, result)
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
v.cleanup()
```

### 检测 + 跟踪 + 姿态

```python
from visionframework import Vision

v = Vision(
    model="yolov8s.pt",
    track=True,
    pose=True,
    fp16=True,
    device="cuda"
)

for frame, meta, result in v.run("video.mp4"):
    for det in result["detections"]:
        print(f"{det.class_name}: {det.confidence:.2f}")
    for track in result["tracks"]:
        print(f"ID {track.track_id}: {track.bbox}")
v.cleanup()
```

### 高性能批量推理

```python
from visionframework import Vision

v = Vision(
    model="yolov8n.pt",
    fp16=True,
    batch_inference=True,
    dynamic_batch=True,
    max_batch_size=16,
    device="cuda"
)

for frame, meta, result in v.run("image_folder/", recursive=True):
    print(f"{meta['source_path']}: {len(result['detections'])} objects")
v.cleanup()
```

### 分类别阈值

```python
from visionframework import Vision

v = Vision(
    model="yolov8n.pt",
    conf=0.25,
    category_thresholds={
        "person": 0.6,
        "car": 0.4,
        "dog": 0.3
    }
)

for frame, meta, result in v.run("street.jpg"):
    for det in result["detections"]:
        print(f"{det.class_name}: {det.confidence:.2f}")
```

### 从配置文件运行

```python
from visionframework import Vision

v = Vision.from_config("config.yaml")
for frame, meta, result in v.run(0):  # 摄像头
    annotated = v.draw(frame, result)
    cv2.imshow("Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
v.cleanup()
```

### CLIP 零样本分类

```python
from visionframework import CLIPExtractor
import cv2

clip = CLIPExtractor(device="cuda", use_fp16=True)
clip.initialize()

image = cv2.imread("test.jpg")
labels = ["猫", "狗", "汽车", "人", "自行车"]
scores = clip.zero_shot_classify(image, labels)

for label, score in sorted(zip(labels, scores), key=lambda x: x[1], reverse=True):
    print(f"{label}: {score:.4f}")

clip.cleanup()
```

### ReID 跟踪

```python
from visionframework import Vision

v = Vision(
    model="yolov8n.pt",
    track=True,
    tracker="reidtracker",
    conf=0.5
)

for frame, meta, result in v.run("input.mp4"):
    for track in result["tracks"]:
        print(f"ID {track.track_id}")
v.cleanup()
```

---

## 可导入符号速查表

所有符号均可通过 `from visionframework import <名称>` 直接导入：

| 类别 | 符号 |
|------|------|
| **主入口** | `Vision` |
| **数据结构** | `Detection`, `Track`, `STrack`, `Pose`, `KeyPoint`, `ROI` |
| **可视化** | `Visualizer` |
| **导出** | `ResultExporter` |
| **基类** | `BaseDetector`, `BaseTracker`, `BaseProcessor` |
| **检测器** | `YOLODetector`, `DETRDetector`, `RFDETRDetector` |
| **跟踪器** | `IOUTracker`, `ByteTracker`, `ReIDTracker` |
| **跟踪工具** | `calculate_iou`, `iou_cost_matrix`, `linear_assignment`, `SCIPY_AVAILABLE` |
| **处理器** | `PoseEstimator`, `CLIPExtractor`, `ReIDExtractor` |
| **分割器** | `SAMSegmenter` |
| **管道** | `VisionPipeline`, `BatchPipeline`, `VideoPipeline` |
| **插件系统** | `PluginRegistry`, `ModelRegistry`, `plugin_registry`, `model_registry`, `register_detector`, `register_tracker`, `register_segmenter`, `register_processor`, `register_model`, `register_visualizer`, `register_evaluator`, `register_custom_component` |
| **ROI / 计数** | `ROIDetector`, `Counter` |
| **配置** | `Config`, `BaseConfig`, `DetectorConfig` |
| **监控** | `PerformanceMonitor`, `PerformanceMetrics`, `Timer` |
| **媒体源** | `iter_frames` |
| **内存管理** | `MemoryPool`, `MultiMemoryPool`, `create_memory_pool`, `acquire_memory`, `release_memory`, `get_memory_pool_status`, `optimize_memory_usage`, `clear_memory_pool`, `clear_all_memory_pools` |
| **并发处理** | `Task`, `ThreadPoolProcessor`, `parallel_map` |
| **数据增强** | `ImageAugmenter`, `AugmentationConfig`, `AugmentationType`, `InterpolationType` |
| **模型优化** | `QuantizationConfig`, `quantize_model`, `PruningConfig`, `prune_model`, `DistillationConfig`, `distill_model`, `compare_model_performance` |
| **模型训练** | `FineTuningConfig`, `FineTuningStrategy`, `ModelFineTuner` |
| **模型转换** | `ModelFormat`, `ConversionConfig`, `ModelConverter`, `convert_model`, `validate_converted_model`, `get_supported_formats`, `get_compatible_formats`, `get_format_extension`, `get_format_dependencies`, `get_format_from_extension` |
| **模型部署** | `DeploymentPlatform`, `DeploymentConfig`, `ModelDeployer`, `deploy_model`, `validate_deployment`, `get_supported_platforms`, `get_platform_compatibility`, `get_platform_requirements`, `get_platform_from_string` |
| **模型管理** | `select_model`, `ModelSelector`, `ModelType`, `ModelRequirement`, `HardwareInfo`, `HardwareTier` |
| **多模态融合** | `FusionType`, `MultimodalFusion`, `fuse_features`, `get_fusion_model` |
| **轨迹分析** | `TrajectoryAnalyzer` |
| **评估工具** | `DetectionEvaluator`, `TrackingEvaluator` |
| **错误处理** | `ErrorHandler`, `DependencyManager`, `dependency_manager`, `is_dependency_available`, `get_available_dependencies`, `get_missing_dependencies`, `validate_dependency`, `get_install_command`, `import_optional_dependency` |
| **异常** | `VisionFrameworkError`, `DetectorInitializationError`, `DetectorInferenceError`, `TrackerInitializationError`, `TrackerUpdateError`, `ConfigurationError`, `ModelNotFoundError`, `ModelLoadError`, `DeviceError`, `DependencyError`, `DataFormatError`, `ProcessingError` |

---

## 配置文件格式

### JSON

```json
{
    "model": "yolov8s.pt",
    "model_type": "yolo",
    "device": "auto",
    "conf": 0.25,
    "iou": 0.45,
    "track": true,
    "tracker": "bytetrack",
    "segment": false,
    "pose": false,
    "fp16": true,
    "batch_inference": false,
    "dynamic_batch": false,
    "max_batch_size": 8,
    "category_thresholds": {
        "person": 0.5,
        "car": 0.3
    }
}
```

### YAML

```yaml
model: yolov8s.pt
model_type: yolo
device: auto
conf: 0.25
iou: 0.45
track: true
tracker: bytetrack
segment: false
pose: false
fp16: true
batch_inference: false
dynamic_batch: false
max_batch_size: 8
category_thresholds:
  person: 0.5
  car: 0.3
```
