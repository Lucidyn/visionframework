# VisionFramework v2.0

模块化、组件式计算机视觉框架。通过 YAML 配置驱动，支持目标检测、语义分割、多目标跟踪等任务。

## 特性

- **组件化架构** — Backbone / Neck / Head 自由组合，通过注册表动态实例化
- **YAML 驱动** — 唯一入口 `TaskRunner(yaml_path)`，零代码配置切换模型和任务
- **内置模型** — YOLO11、YOLO26、DETR、RF-DETR，以及 CSPDarknet、ResNet 等基础组件
- **官方权重** — 支持加载 Facebook DETR (458/458 完美映射)、ultralytics YOLO 官方预训练权重
- **类别过滤** — 通过 `filter_classes` 配置项指定只检测某些类别
- **多任务支持** — 检测、分割、跟踪、ReID 跟踪，统一 pipeline 管理
- **最少依赖** — 核心仅需 `torch`、`opencv-python`、`numpy`、`pyyaml`

## 快速开始

### 安装

```bash
pip install torch torchvision opencv-python numpy pyyaml
```

### 目标检测

```python
from visionframework import TaskRunner

task = TaskRunner("configs/runtime/detect.yaml")
result = task.process(image)
detections = result["detections"]
```

### 视频跟踪

```python
task = TaskRunner("configs/runtime/tracking.yaml")
for frame, meta, result in task.run("video.mp4"):
    tracks = result["tracks"]
```

### DETR 检测（使用官方预训练权重）

```python
# 1. 先转换官方权重
#    python tools/convert_detr.py \
#        --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
#        --output weights/detr_r50.pth --verify

# 2. 在 YAML 中指定权重路径，运行
task = TaskRunner("configs/runtime/detect_detr.yaml")
result = task.process(image)
```

### RF-DETR 检测（通过适配器）

```python
# 需要额外安装: pip install rfdetr
from tools.rfdetr_adapter import RFDETRAdapter

adapter = RFDETRAdapter(model_size="base", conf=0.5)
detections = adapter.predict(image_bgr)
```

### 类别过滤

在 runtime YAML 中添加 `filter_classes` 即可只检测指定类别，支持类别名称和 ID 混用：

```yaml
# configs/runtime/detect_person.yaml
pipeline: detection
models:
  detector: configs/models/yolo_n.yaml
filter_classes:
  - person
```

```yaml
# 同时过滤多个类别
filter_classes:
  - car
  - bus
  - truck
```

```yaml
# 按 ID 过滤（int / str 可混用）
filter_classes:
  - 0           # person (COCO class id)
  - bus         # 按名称
```

## 项目结构

```
visionframework/
├── layers/                # 原子层：ConvBNAct, C3k2, C2PSA, Attention, SPPF, ...
│   ├── conv.py            # 基础卷积块
│   ├── csp.py             # CSP 系列模块 (C3k2, C2PSA, PSABlock)
│   ├── pooling.py         # SPPF, SPP
│   ├── attention.py       # SEBlock, CBAM
│   ├── positional.py      # 2D 正弦位置编码（与官方 DETR 对齐）
│   └── deformable_attn.py # 可变形注意力
├── models/
│   ├── backbones/         # YOLOBackbone, ResNet, CSPDarknet, DINOv2Backbone
│   ├── necks/             # YOLOPAN, PAN, FPN, TransformerEncoderNeck, DeformableEncoderNeck
│   └── heads/             # YOLOHead, DETRHead, RFDETRHead, SegHead, ReIDHead
├── algorithms/
│   ├── detection/         # Detector (YOLO), DETRDetector (DETR)
│   ├── segmentation/      # Segmenter
│   ├── tracking/          # ByteTracker, IOUTracker
│   └── reid/              # Embedder
├── pipelines/             # DetectionPipeline, TrackingPipeline, ...
├── core/
│   ├── registry.py        # 组件注册表
│   ├── builder.py         # 模型/算法/管线构建器
│   └── config.py          # YAML 配置加载与解析
├── engine/                # 数据源处理 (图片/视频/摄像头)
├── task_api.py            # TaskRunner — 唯一公共入口
├── utils/                 # bbox, nms, device, visualization
└── data/                  # Detection, Track, Pose 等数据结构

configs/
├── models/                # 模型配置
│   ├── yolo_n.yaml        # CSPDarknet-YOLOHead (nano)
│   ├── yolo_s.yaml        # CSPDarknet-YOLOHead (small)
│   ├── yolo11n/s/m/l.yaml # YOLOBackbone+YOLOPAN (YOLO11 系列)
│   ├── yolo26n/s/m/l.yaml # YOLOBackbone+YOLOPAN(c3k) (YOLO26 系列)
│   ├── detr_r50.yaml      # ResNet-50 + TransformerEncoder + DETRHead
│   ├── rfdetr_base.yaml   # DINOv2 + DeformableEncoder + RFDETRHead
│   ├── resnet50_seg.yaml  # ResNet-50 语义分割
│   └── osnet_reid.yaml    # OSNet ReID 特征提取
├── runtime/               # 运行时配置
│   ├── detect.yaml        # 通用检测
│   ├── detect_detr.yaml   # DETR 检测
│   ├── detect_rfdetr.yaml # RF-DETR 检测
│   ├── detect_person.yaml # 只检测行人（类别过滤示例）
│   ├── detect_vehicles.yaml # 只检测车辆（类别过滤示例）
│   ├── tracking.yaml      # ByteTrack 跟踪
│   ├── reid_tracking.yaml # ReID 跟踪
│   └── segmentation.yaml  # 语义分割
└── components/            # 组件配置
    ├── bytetrack.yaml     # ByteTracker 参数
    └── iou_tracker.yaml   # IOUTracker 参数

tools/
├── convert_ultralytics.py # ultralytics YOLO 权重转换工具
├── convert_detr.py        # Facebook DETR 官方权重转换工具
└── rfdetr_adapter.py      # RF-DETR 适配器（封装 rfdetr 包）

examples/                  # 教程示例 (全部 YAML 驱动)
```

## 内置模型

| 模型 | 配置文件 | Backbone | Neck | Head | 特点 |
|------|----------|----------|------|------|------|
| YOLO-nano | `yolo_n.yaml` | CSPDarknet | PAN | YOLOHead | 基础轻量模型 |
| YOLO-small | `yolo_s.yaml` | CSPDarknet | PAN | YOLOHead | 基础小型模型 |
| YOLO11n | `yolo11n.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，轻量 |
| YOLO11s | `yolo11s.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，小型 |
| YOLO11m | `yolo11m.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，中型 |
| YOLO11l | `yolo11l.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，大型 |
| YOLO26n | `yolo26n.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free, reg_max=1 |
| YOLO26s | `yolo26s.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free, reg_max=1 |
| YOLO26m | `yolo26m.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free, reg_max=1 |
| YOLO26l | `yolo26l.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free, reg_max=1 |
| DETR-R50 | `detr_r50.yaml` | ResNet-50 | TransformerEncoderNeck | DETRHead | 无 NMS，集合预测 |
| RF-DETR | `rfdetr_base.yaml` | DINOv2Backbone | DeformableEncoderNeck | RFDETRHead | 可变形注意力 |
| 分割 | `resnet50_seg.yaml` | ResNet-50 | FPN | SegHead | 语义分割 |
| ReID | `osnet_reid.yaml` | OSNet | — | ReIDHead | 行人重识别 |

## 官方预训练权重

### DETR (Facebook)

框架的 DETR 实现与 Facebook 官方完全对齐，支持直接加载官方预训练权重：

```bash
# 下载并转换官方 DETR-R50 权重（458/458 keys 完美映射）
python tools/convert_detr.py \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detr_r50.pth \
    --verify

# 支持的官方 checkpoint:
#   DETR-R50:      https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
#   DETR-R101:     https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
#   DETR-DC5-R50:  https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
#   DETR-DC5-R101: https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
```

转换后在模型 YAML 中添加 `weights` 字段：

```yaml
# configs/models/detr_r50.yaml 中添加
weights: weights/detr_r50.pth
```

### YOLO (ultralytics)

```bash
# 转换 ultralytics YOLO11n 权重
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_vf.pt --test
```

### RF-DETR (Roboflow)

RF-DETR 架构复杂度较高，采用适配器模式直接调用 `rfdetr` 包推理：

```bash
# 安装依赖
pip install rfdetr

# 命令行测试
python tools/rfdetr_adapter.py --model base --image test.jpg --conf 0.5
```

```python
# 代码调用
from tools.rfdetr_adapter import RFDETRAdapter

adapter = RFDETRAdapter(model_size="base", conf=0.5)
detections = adapter.predict(image_bgr)
```

支持的模型: `nano`, `small`, `base`, `medium`, `large`。

## 运行时配置参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `pipeline` | str | 任务类型：`detection` / `tracking` / `segmentation` / `reid_tracking` |
| `algorithm` | str | 检测算法：`Detector`（默认）或 `DETRDetector` |
| `models.detector` | str | 检测模型配置路径 |
| `models.segmenter` | str | 分割模型配置路径 |
| `models.reid` | str | ReID 模型配置路径 |
| `tracker` | str/dict | 跟踪器配置路径或内联配置 |
| `device` | str | `auto` / `cpu` / `cuda` |
| `fp16` | bool | 半精度推理 |
| `filter_classes` | list | 类别过滤，支持类别名称 (str) 和 ID (int) 混用 |

## 自定义模型

通过 YAML 自由组合组件：

```yaml
# 自定义: ResNet-50 backbone + PAN neck + YOLO head
backbone:
  type: ResNet
  layers: 50

neck:
  type: PAN
  in_channels: [512, 1024, 2048]
  depth: 0.33

head:
  type: YOLOHead
  in_channels: [512, 1024, 2048]
  num_classes: 20
  reg_max: 16
```

## 依赖

| 包 | 版本 | 说明 |
|----|------|------|
| torch | ≥1.10 | 核心 |
| opencv-python | ≥4.5 | 图像处理 |
| numpy | ≥1.20 | 数值计算 |
| pyyaml | ≥5.0 | 配置文件 |
| scipy | 可选 | ByteTrack 匹配 |
| ultralytics | 可选 | YOLO 权重转换 |
| rfdetr | 可选 | RF-DETR 推理适配器 |

## 许可

MIT License
