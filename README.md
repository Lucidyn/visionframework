# Vision Framework

[![Python 版本](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![版本](https://img.shields.io/badge/version-0.4.0-orange.svg)](docs/CHANGELOG.md)

轻量、模块化的计算机视觉框架，支持目标检测、跟踪、实例分割、姿态估计、ROI 计数、模型优化与部署。

**整个框架只有一个入口：`Vision` 类。所有组件均可直接从 `visionframework` 导入。**

## 安装

```bash
git clone https://github.com/yourusername/visionframework.git
cd visionframework
pip install -e .

# 安装可选依赖（按需选择）
pip install -e ".[clip]"    # CLIP / DETR 支持
pip install -e ".[sam]"     # SAM 分割支持
pip install -e ".[dev]"     # 开发工具（pytest 等）
pip install -e ".[all]"     # 全部可选依赖
```

## 快速上手

### 方式一：关键字参数

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)

for frame, meta, result in v.run("video.mp4"):
    print(result["detections"])
    print(result["tracks"])
```

### 方式二：从配置文件

```python
from visionframework import Vision

v = Vision.from_config("config.json")   # 支持 .json / .yaml / dict

for frame, meta, result in v.run("video.mp4"):
    print(result["detections"])
```

**config.json 示例：**

```json
{
    "model": "yolov8n.pt",
    "track": true,
    "conf": 0.25,
    "device": "auto",
    "fp16": true
}
```

## `source` 支持一切

`v.run(source)` 无需额外代码，直接支持：

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
| `conf` | `0.25` | 全局置信度阈值 |
| `iou` | `0.45` | NMS IoU 阈值 |
| `track` | `False` | 开启多目标跟踪 |
| `tracker` | `"bytetrack"` | 跟踪器：`bytetrack` / `ioutracker` / `reidtracker` |
| `segment` | `False` | 开启实例分割 |
| `pose` | `False` | 开启姿态估计 |
| `fp16` | `False` | FP16 半精度推理（CUDA） |
| `batch_inference` | `False` | 启用批量推理 |
| `dynamic_batch` | `False` | 动态调整批量大小 |
| `max_batch_size` | `8` | 最大 batch 大小 |
| `category_thresholds` | `None` | 按类别阈值，如 `{"person": 0.5}` |

## 更多示例

### 姿态估计

```python
from visionframework import Vision

v = Vision(model="yolov8n-pose.pt", pose=True)
for frame, meta, result in v.run("test.jpg"):
    for pose in result["poses"]:
        print(f"{len(pose.keypoints)} 个关键点")
```

### 实例分割

```python
from visionframework import Vision

v = Vision(model="yolov8n-seg.pt", segment=True)
for frame, meta, result in v.run("test.jpg"):
    for det in result["detections"]:
        print(f"{det.class_name}: mask={'有' if det.mask is not None else '无'}")
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
v.cleanup()
```

### ROI 区域计数（v0.4.0 新增）

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)
v.add_roi("入口", [(100,100),(400,100),(400,400),(100,400)])

for frame, meta, result in v.run("video.mp4"):
    counts = result["counts"]
    # {"入口": {"inside": 3, "entering": 1, "exiting": 0, "total_entered": 12}}
```

### 批量图像处理（v0.4.0 新增）

```python
from visionframework import Vision

v = Vision(model="yolov8n.pt")
results = v.process_batch([img1, img2, img3])
for i, r in enumerate(results):
    print(f"图像 {i}: {len(r['detections'])} 个目标")
```

### 热力图可视化（v0.4.0 新增）

```python
from visionframework import Vision, Visualizer
import cv2

v = Vision(model="yolov8n.pt", track=True)
vis = Visualizer()
state = {}

for frame, meta, result in v.run("video.mp4"):
    heatmap = vis.draw_heatmap(frame, result["tracks"],
                               alpha=0.5, accumulate=True, _heat_state=state)
    cv2.imshow("热力图", heatmap)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### 模型优化

```python
from visionframework import QuantizationConfig, quantize_model, PruningConfig, prune_model
import torch

model = torch.load("model.pt").eval()

# 动态量化
q_model = quantize_model(model, QuantizationConfig(quantization_type="dynamic"))

# L1 剪枝
import torch.nn as nn
p_model = prune_model(model, PruningConfig(pruning_type="l1_unstructured", amount=0.3,
                                           target_modules=[nn.Linear]))
```

### 自定义插件

```python
from visionframework import BaseDetector, Detection, register_detector
import numpy as np

@register_detector("my_detector")
class MyDetector(BaseDetector):
    def initialize(self) -> bool:
        self._initialized = True
        return True

    def detect(self, image: np.ndarray) -> list:
        return []
```

## 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 安装与最小示例 |
| [功能特性](docs/FEATURES.md) | 功能一览与场景说明 |
| [API 参考](docs/API_REFERENCE.md) | 详细的 API 文档 |
| [高级指南](docs/ADVANCED.md) | 插件、优化、部署、微调 |
| [更新日志](docs/CHANGELOG.md) | 版本历史 |
| [贡献指南](docs/CONTRIBUTING.md) | 如何参与贡献 |

示例脚本在 `examples/`，查看 `examples/README.md` 获取运行命令。

## 版本历史

**v0.4.0（2026-02-26）**：
- `Vision.add_roi()` — ROI 区域计数
- `Vision.process_batch()` — 批量图像处理
- `Vision.info()` — 实例配置摘要
- `Visualizer.draw_heatmap()` — 轨迹热力图
- LoRA / QLoRA 微调完整实现
- 统一导入风格：所有组件直接从 `visionframework` 导入
- 新增 11 个测试文件，264 个测试全部通过

**v0.3.0（2026-02-07）**：
- 全新 `Vision` 类，一个入口取代所有旧 API
- 统一 `run()` 方法处理图片/视频/摄像头/RTSP/文件夹
- `draw()` 方法一行绘制所有结果

<details>
<summary>更早版本</summary>

**v0.2.15**：共享跟踪器工具、输入验证增强、ByteTracker 修复、~250 行代码精简

**v0.2.14**：VisionPipeline 批处理增强、视频批处理、模型优化工具

**v0.2.13**：内存池管理、插件系统、统一错误处理、依赖管理优化

</details>

## 依赖项

### 必需

- opencv-python >= 4.8.0
- numpy >= 1.24.0, < 2.0.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics >= 8.0.0
- scipy >= 1.10.0
- Pillow >= 10.0.0
- pyyaml >= 6.0
- pydantic >= 2.0.0
- huggingface_hub >= 0.14.0

### 可选

- transformers（DETR / CLIP 支持）
- segment-anything（SAM 分割）
- rfdetr（RF-DETR 检测器）
- onnx / onnxruntime（ONNX 模型转换与推理）
- av（PyAV 高性能视频处理）
- peft（LoRA / QLoRA 微调）
- bitsandbytes（QLoRA 量化微调）
- psutil（内存监控）

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
