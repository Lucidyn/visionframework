# API 参考文档

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

### `Vision.draw(frame, result)`

在帧上绘制检测/跟踪/姿态结果。

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
