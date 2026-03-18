# 示例教程

本目录包含 VisionFramework 的使用示例，全部通过 YAML 配置文件驱动。

## 示例列表

| 示例 | 文件 | 说明 |
|------|------|------|
| 目标检测 | `01_detection.py` | YOLO11 单图检测 |
| 多目标跟踪 | `02_tracking.py` | ByteTrack 多帧跟踪 |
| 语义分割 | `03_segmentation.py` | ResNet50 语义分割 |
| 可视化 | `04_visualization.py` | 检测结果绘制 |
| DETR 检测 | `05_detr_detection.py` | Facebook DETR 官方权重 |
| YOLO26 检测 | `06_yolo26_detection.py` | YOLO26 端到端检测（NMS-free）|
| RF-DETR 检测 | `07_rfdetr_detection.py` | RF-DETR 原生实现 |

## 运行方式

```bash
# 确保在项目根目录
cd visionframework

# 推荐：开发模式安装（确保可直接 import visionframework）
pip install -e .

# 也可以使用命令行入口（安装后）
# vf-test-yolo26 --quick

# 运行示例
python examples/01_detection.py
python examples/02_tracking.py
python examples/03_segmentation.py
python examples/04_visualization.py
python examples/05_detr_detection.py
python examples/06_yolo26_detection.py
python examples/07_rfdetr_detection.py
```

## 核心用法

框架唯一入口是 `TaskRunner(yaml_path)`：

```python
from visionframework import TaskRunner

# 加载 YAML 配置并运行
task = TaskRunner("runs/detection/yolo11/detect.yaml")

# 处理单张图片
result = task.process(image)

# 处理视频/摄像头
for frame, meta, result in task.run("video.mp4"):
    ...
```

## 权重转换工具

```bash
# ultralytics YOLO 权重
python -m visionframework.tools.convert_ultralytics --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth

# Facebook DETR 官方权重（458/458 keys 完美映射）
python -m visionframework.tools.convert_detr \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detection/detr/detr_r50.pth --verify
```

## RF-DETR

RF-DETR 默认加载官方 `.pth`（不存在会自动下载，需要安装 `rfdetr` 用于构建同构网络与下载权重）。

```bash
python examples/07_rfdetr_detection.py
```

## 配置与目录

- **runs/**：运行/流水线配置（入口 YAML），`TaskRunner("runs/...")` 加载的即此类文件。
- **configs/**：仅模型配置（backbone/neck/head 等），由 runs 内 `models.detector`、`tracker` 等引用。
- **weights/**：权重按 `weights/<任务>/<算法>/` 存放，路径在 runs 的 `weights` 字段中指定。详见项目根目录 `weights/README.md`。

## YAML 配置说明

### 运行配置 (runs/<task>/<algo>/)

```yaml
pipeline: detection          # detection / tracking / segmentation / reid_tracking
algorithm: DETRDetector      # 可选，指定检测算法类型

models:
  detector: configs/detection/yolo11/yolo11n.yaml  # 模型配置路径

device: auto                 # auto / cpu / cuda
fp16: false                  # 半精度推理

# 类别过滤：只检测指定类别，支持名称 (str) 和 ID (int) 混用
filter_classes:
  - person
  - car
  - 5                        # 也可以用类别 ID
```

### 模型配置 (configs/<task>/<algo>/)

```yaml
backbone:
  type: YOLOBackbone
  depth: 0.50
  width: 0.25

neck:
  type: YOLOPAN
  in_channels: [128, 128, 256]
  depth: 0.50
  c3k: false

head:
  type: YOLOHead
  in_channels: [128, 128, 256]
  num_classes: 80
  reg_max: 16

postprocess:
  conf: 0.25
  nms_iou: 0.45

# 可选：指定预训练权重路径
# weights: weights/yolo11n_vf.pt
```

### 内置运行配置一览（runs/）

入口统一为 `TaskRunner("runs/...")`，运行配置内引用 `configs/` 下的模型配置与 `weights/` 下的权重。

| 运行配置路径 | 说明 |
|--------------|------|
| `runs/detection/yolo11/detect.yaml` | 通用 YOLO11 检测 |
| `runs/detection/yolo11/detect_person.yaml` | 只检测行人（类别过滤） |
| `runs/detection/yolo11/detect_vehicles.yaml` | 只检测车辆（类别过滤） |
| `runs/detection/yolo26/detect.yaml` | YOLO26 端到端检测（NMS-free）|
| `runs/detection/detr/detect.yaml` | DETR 检测 |
| `runs/detection/rfdetr/detect_nano.yaml` | RF-DETR `.pth` 检测（nano） |
| `runs/detection/rfdetr/detect_small.yaml` | RF-DETR `.pth` 检测（small） |
| `runs/detection/rfdetr/detect_base.yaml` | RF-DETR `.pth` 检测（base） |
| `runs/detection/rfdetr/detect_medium.yaml` | RF-DETR `.pth` 检测（medium） |
| `runs/detection/rfdetr/detect_large.yaml` | RF-DETR `.pth` 检测（large） |
| `runs/tracking/bytetrack/tracking.yaml` | ByteTrack 多目标跟踪 |
| `runs/tracking/bytetrack/reid_tracking.yaml` | ReID 增强跟踪 |
| `runs/segmentation/resnet50/segmentation.yaml` | 语义分割 |
