# 示例教程

本目录包含 VisionFramework 的使用示例，全部通过 YAML 配置文件驱动。

## 示例列表

| 示例 | 文件 | 说明 |
|------|------|------|
| 目标检测 | `01_detection.py` | 使用 YOLO 进行单图检测 |
| 多目标跟踪 | `02_tracking.py` | ByteTrack 多帧跟踪 |
| 语义分割 | `03_segmentation.py` | ResNet50 语义分割 |
| 可视化 | `04_visualization.py` | 检测结果绘制 |
| 真实检测 | `05_real_detection.py` | ultralytics 预训练权重真实图片检测 |

## 运行方式

```bash
# 确保在项目根目录
cd visionframework

# 运行示例
python examples/01_detection.py
python examples/02_tracking.py
python examples/03_segmentation.py
python examples/04_visualization.py
```

## 核心用法

框架唯一入口是 `TaskRunner(yaml_path)`：

```python
from visionframework import TaskRunner

# 加载 YAML 配置并运行
task = TaskRunner("configs/runtime/detect.yaml")

# 处理单张图片
result = task.process(image)

# 处理视频/摄像头
for frame, meta, result in task.run("video.mp4"):
    ...
```

## 权重转换工具

```bash
# ultralytics YOLO 权重
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_vf.pt

# Facebook DETR 官方权重（458/458 keys 完美映射）
python tools/convert_detr.py \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detr_r50.pth --verify
```

## RF-DETR 适配器

RF-DETR 采用适配器模式，直接封装 `rfdetr` 包推理（需 `pip install rfdetr`）：

```bash
python tools/rfdetr_adapter.py --model base --image test.jpg --conf 0.5
```

```python
from tools.rfdetr_adapter import RFDETRAdapter
adapter = RFDETRAdapter(model_size="base", conf=0.5)
detections = adapter.predict(image_bgr)
```

## YAML 配置说明

### 运行时配置 (configs/runtime/)

```yaml
pipeline: detection          # detection / tracking / segmentation / reid_tracking
algorithm: DETRDetector      # 可选，指定检测算法类型

models:
  detector: configs/models/yolo11n.yaml  # 模型配置路径

device: auto                 # auto / cpu / cuda
fp16: false                  # 半精度推理

# 类别过滤：只检测指定类别，支持名称 (str) 和 ID (int) 混用
filter_classes:
  - person
  - car
  - 5                        # 也可以用类别 ID
```

### 模型配置 (configs/models/)

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

### 内置运行时配置一览

| 配置文件 | 说明 |
|----------|------|
| `detect.yaml` | 通用 YOLO 检测 |
| `detect_detr.yaml` | DETR 检测 |
| `detect_rfdetr.yaml` | RF-DETR 检测 |
| `detect_person.yaml` | 只检测行人（类别过滤） |
| `detect_vehicles.yaml` | 只检测车辆（类别过滤） |
| `tracking.yaml` | ByteTrack 多目标跟踪 |
| `reid_tracking.yaml` | ReID 增强跟踪 |
| `segmentation.yaml` | 语义分割 |
