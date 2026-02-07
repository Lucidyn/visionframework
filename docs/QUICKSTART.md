# 快速开始指南

## 安装

```bash
git clone https://github.com/yourusername/visionframework.git
cd visionframework
pip install -e .
```

## 核心概念

整个框架只有一个入口类 `Vision`，有两种创建方式：

```python
from visionframework import Vision

# 方式一：关键字参数
v = Vision(model="yolov8n.pt", track=True)

# 方式二：配置文件 (JSON / YAML / dict)
v = Vision.from_config("config.json")
```

创建后，使用 `v.run(source)` 处理任意媒体源。`source` 支持：图片路径、视频路径、摄像头索引 (0)、RTSP 流、文件夹、路径列表、numpy 数组。

## 示例 1: 基本目标检测

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt")

for frame, meta, result in v.run("test.jpg"):
    detections = result["detections"]
    print(f"检测到 {len(detections)} 个物体")
    for det in detections:
        print(f"  {det.class_name}: {det.confidence:.2f}")
```

## 示例 2: 检测 + 跟踪

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)

for frame, meta, result in v.run("video.mp4"):
    tracks = result["tracks"]
    print(f"帧 {meta.get('frame_index')}: 跟踪 {len(tracks)} 个目标")
    for t in tracks:
        print(f"  ID={t.track_id}, 类别={t.class_name}, 位置={t.bbox}")
```

## 示例 3: 从配置文件

```json
{
    "model": "yolov8n.pt",
    "model_type": "yolo",
    "device": "auto",
    "conf": 0.3,
    "track": true,
    "tracker": "bytetrack",
    "pose": false,
    "segment": false
}
```

```python
from visionframework import Vision

v = Vision.from_config("config.json")

for frame, meta, result in v.run(0):  # 摄像头
    print(result["detections"])
```

## 示例 4: 姿态估计

```python
from visionframework import Vision

v = Vision(model="yolov8n-pose.pt", pose=True)

for frame, meta, result in v.run("test.jpg"):
    poses = result["poses"]
    print(f"检测到 {len(poses)} 个人体姿态")
```

## 示例 5: 视频 + 可视化

```python
import cv2
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)

for frame, meta, result in v.run("video.mp4", skip_frames=2):
    annotated = v.draw(frame, result)
    cv2.imshow("Vision Framework", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
v.cleanup()
```

## 示例 6: 处理文件夹

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt")

# 递归处理文件夹中所有图片和视频
for frame, meta, result in v.run("images_folder/", recursive=True):
    print(f"[{meta.get('source_path')}] {len(result['detections'])} 个检测")
```

## Vision 类参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | `"yolov8n.pt"` | 模型路径或名称 |
| `model_type` | str | `"yolo"` | 检测器后端 |
| `device` | str | `"auto"` | 推理设备 |
| `conf` | float | `0.25` | 置信度阈值 |
| `iou` | float | `0.45` | NMS IoU 阈值 |
| `track` | bool | `False` | 开启跟踪 |
| `tracker` | str | `"bytetrack"` | 跟踪器类型 |
| `segment` | bool | `False` | 开启分割 |
| `pose` | bool | `False` | 开启姿态估计 |

## run() 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source` | str/int/list/ndarray | - | 媒体源 |
| `recursive` | bool | `False` | 文件夹递归 |
| `skip_frames` | int | `0` | 视频跳帧数 |
| `start_frame` | int | `0` | 视频起始帧 |
| `end_frame` | int/None | `None` | 视频结束帧 |

## 返回值

`v.run(source)` 返回一个迭代器，每次迭代产生一个元组 `(frame, meta, result)`:

- **frame**: `np.ndarray` — BGR 格式图像
- **meta**: `dict` — 元数据 (`source_path`, `frame_index`, `is_video` 等)
- **result**: `dict` — 包含:
  - `"detections"`: `List[Detection]` — 检测结果
  - `"tracks"`: `List[Track]` — 跟踪结果 (需 `track=True`)
  - `"poses"`: `List[Pose]` — 姿态结果 (需 `pose=True`)

## 下一步

- 查看 [examples/](../examples/) 获取完整示例
- 阅读 [API 参考](API_REFERENCE.md) 了解完整接口
- 阅读 [功能特性](FEATURES.md) 了解全部能力
