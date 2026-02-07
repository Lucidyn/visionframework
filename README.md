# Vision Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

轻量、模块化的计算机视觉框架，支持目标检测、跟踪、实例分割、姿态估计与结果导出。

**v0.3.0 起，整个框架只有一个入口：`Vision` 类。**

## 安装

```bash
pip install -e .
```

## 两种 API

### 方式一：直接创建

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)

for frame, meta, result in v.run("video.mp4"):
    print(result["detections"])
    print(result["tracks"])
```

### 方式二：从配置文件创建

```json
{
    "model": "yolov8n.pt",
    "track": true,
    "conf": 0.25,
    "device": "auto"
}
```

```python
from visionframework import Vision

v = Vision.from_config("config.json")    # 也支持 .yaml / .yml / dict

for frame, meta, result in v.run("video.mp4"):
    print(result["detections"])
```

## `source` 支持一切

`v.run(source)` 中的 `source` 参数支持以下任意类型，**无需额外代码**：

| source 值 | 含义 |
|-----------|------|
| `"test.jpg"` | 单张图片 |
| `"video.mp4"` | 视频文件 |
| `0` | 摄像头 |
| `"rtsp://..."` | RTSP / HTTP 视频流 |
| `"images_folder/"` | 包含图片/视频的文件夹 |
| `["a.jpg", "b.mp4"]` | 多个路径组成的列表 |
| `np.ndarray` | BGR numpy 图像数组 |

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"yolov8n.pt"` | 模型路径或名称 |
| `model_type` | `"yolo"` | 检测器后端：`yolo` / `detr` / `rfdetr` |
| `device` | `"auto"` | 设备：`auto` / `cpu` / `cuda` / `cuda:0` |
| `conf` | `0.25` | 置信度阈值 |
| `iou` | `0.45` | NMS IoU 阈值 |
| `track` | `False` | 开启多目标跟踪 |
| `tracker` | `"bytetrack"` | 跟踪器：`bytetrack` / `ioutracker` / `reidtracker` |
| `segment` | `False` | 开启实例分割 |
| `pose` | `False` | 开启姿态估计 |

## 更多示例

### 姿态估计

```python
v = Vision(model="yolov8n-pose.pt", pose=True)
for frame, meta, result in v.run("test.jpg"):
    for pose in result["poses"]:
        print(f"{len(pose.keypoints)} keypoints")
```

### 实例分割

```python
v = Vision(model="yolov8n-seg.pt", segment=True)
for frame, meta, result in v.run("test.jpg"):
    for det in result["detections"]:
        print(f"{det.class_name}: mask={'yes' if det.mask is not None else 'no'}")
```

### 视频处理 + 可视化

```python
import cv2
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)
for frame, meta, result in v.run("video.mp4", skip_frames=2):
    annotated = v.draw(frame, result)
    cv2.imshow("Vision", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### 从 YAML 配置文件

```yaml
# config.yaml
model: yolov8n.pt
model_type: yolo
device: auto
conf: 0.3
track: true
tracker: bytetrack
pose: false
segment: false
```

```python
v = Vision.from_config("config.yaml")
```

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 安装与最小示例 |
| [功能特性](docs/FEATURES.md) | 功能一览与场景说明 |
| [API 参考](docs/API_REFERENCE.md) | 详细的 API 文档 |

示例脚本在 `examples/`，查看 `examples/README.md` 获取运行命令。

## 关键更新

**v0.3.0 - API 大幅简化** (2026-02-07):
- **全新 `Vision` 类**：一个类取代所有旧 API (`create_detector`, `create_pipeline`, `process_image` 等)
- **两种创建方式**：`Vision(...)` 关键字参数 + `Vision.from_config(path)` 配置文件
- **统一 `run()` 方法**：处理图片/视频/摄像头/RTSP/文件夹/列表
- **`draw()` 方法**：一行绘制检测/跟踪/姿态结果
- **向后兼容**：旧的内部类 (YOLODetector, VisionPipeline 等) 仍可导入
- 修复了 `core/pipelines/pipeline.py` 中的相对导入错误

**v0.2.15 - 核心代码优化与重构**:
- 共享跟踪器工具，消除 ~90 行重复代码
- 输入验证增强
- ByteTracker bug 修复
- ~250 行代码精简

<details>
<summary>更早版本</summary>

**v0.2.14 - 测试修复与功能扩展**:
- VisionPipeline 批处理增强、视频批处理、模型优化工具等

**v0.2.13 - 架构优化与功能增强**:
- 内存池管理、插件系统、统一错误处理、依赖管理优化等

</details>

## 依赖项

### 必需
- opencv-python >= 4.8.0
- numpy >= 1.24.0, < 2.0.0
- torch >= 2.0.0
- ultralytics >= 8.0.0

### 可选
- transformers (DETR/CLIP)
- segment-anything (SAM 分割)
- rfdetr (RF-DETR)
- scipy (匈牙利算法匹配)
- av (PyAV 视频处理)
- pyyaml (YAML 配置文件)

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
