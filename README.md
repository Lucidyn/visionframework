# VisionFramework v2.0

模块化、组件式计算机视觉框架。通过 YAML 配置驱动，支持目标检测、语义分割、多目标跟踪等任务。

## 特性

- **组件化架构** — Backbone / Neck / Head 自由组合，通过注册表动态实例化
- **YAML 驱动** — 唯一入口 `TaskRunner(yaml_path)`，零代码配置切换模型和任务
- **内置模型** — YOLO11、YOLO26、DETR、RF-DETR，以及 CSPDarknet、ResNet 等基础组件
- **官方权重** — 支持加载 Facebook DETR（458/458 完美映射）、ultralytics YOLO 官方预训练权重
- **YOLO26 端到端** — NMS-free one-to-one 检测头，`end2end: true` 一键启用
- **类别过滤** — 通过 `filter_classes` 配置项指定只检测某些类别
- **多任务支持** — 检测、分割、跟踪、ReID 跟踪，统一 pipeline 管理
- **最少依赖** — 核心仅需 `torch`、`opencv-python`、`numpy`、`pyyaml`

## 快速开始

### 安装

```bash
pip install torch torchvision opencv-python numpy pyyaml
```

所有示例的统一用法模式：**加载图片 → `TaskRunner` → 可视化保存**。

---

### YOLO11 目标检测

```bash
# 转换 ultralytics 官方权重
pip install ultralytics
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_converted.pth
```

```python
import cv2
from visionframework import TaskRunner, Visualizer

img = cv2.imread("test_bus.jpg")
task = TaskRunner("configs/runtime/detect.yaml")      # weights 字段已在 yaml 中指定
result = task.process(img)

vis = Visualizer()
cv2.imwrite("result.jpg", vis.draw_detections(img.copy(), result["detections"]))
```

> 参考示例：`examples/01_detection.py`

---

### YOLO26 端到端检测（NMS-free）

YOLO26 使用 one-to-one 检测头，无需 NMS 后处理，推理更快。

```bash
python tools/convert_ultralytics.py --model yolo26n.pt --out weights/yolo26n_converted.pth
```

```python
task = TaskRunner("configs/runtime/detect_yolo26.yaml")   # end2end 已在 yaml 中配置
result = task.process(img)
```

| 特性 | YOLO11 | YOLO26 |
|------|--------|--------|
| 检测头 | one-to-many + NMS | one-to-one（NMS-free）|
| reg_max | 16（DFL 解码）| 1（直接回归）|
| SPPF | cv1 有激活 | cv1 无激活 + 残差连接 |

> 参考示例：`examples/06_yolo26_detection.py`

---

### DETR 检测（Facebook 官方预训练权重）

```bash
python tools/convert_detr.py \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detr_r50.pth --verify
```

```python
task = TaskRunner("configs/runtime/detect_detr.yaml")
result = task.process(img)
```

> 参考示例：`examples/05_detr_detection.py`

---

### RF-DETR 检测（Roboflow，DINOv2 backbone）

```bash
pip install rfdetr
```

```python
from tools.rfdetr_adapter import RFDETRAdapter
from visionframework import Visualizer

adapter = RFDETRAdapter(model_size="base", conf=0.5)
detections = adapter.predict(img)
```

> 参考示例：`examples/07_rfdetr_detection.py`

---

### 视频跟踪

```python
task = TaskRunner("configs/runtime/tracking.yaml")
for frame, meta, result in task.run("video.mp4"):
    tracks = result["tracks"]
```

### 类别过滤

在 runtime YAML 中添加 `filter_classes` 即可只检测指定类别，支持名称和 ID 混用：

```yaml
filter_classes:
  - person
  - car
  - 7    # truck (COCO class id)
```

## 项目结构

```
visionframework/
├── layers/                   # 原子层
│   ├── conv.py               # ConvBNAct, DWConvBNAct, DepthwiseSepConv, Focus
│   ├── csp.py                # C3k2, C2PSA, PSABlock, Attention, C2f, CSPBlock
│   ├── pooling.py            # SPPF（支持 cv1_act/residual）, SPP
│   ├── mlp.py                # MLP（DETR/RF-DETR 共享）
│   ├── attention.py          # SEBlock, CBAM, TransformerBlock
│   ├── positional.py         # 2D 正弦位置编码
│   └── deformable_attn.py    # 可变形注意力
├── models/
│   ├── backbones/            # YOLOBackbone, ResNet, CSPDarknet, DINOv2Backbone
│   ├── necks/                # YOLOPAN, PAN, FPN, TransformerEncoderNeck, DeformableEncoderNeck
│   └── heads/                # YOLOHead, DETRHead, RFDETRHead, SegHead, ReIDHead
├── algorithms/
│   ├── base.py               # BaseAlgorithm（设备管理、fp16、predict_batch 共享基类）
│   ├── detection/            # Detector (YOLO/YOLO26), DETRDetector (DETR)
│   ├── segmentation/         # Segmenter
│   ├── tracking/             # ByteTracker, IOUTracker
│   └── reid/                 # Embedder
├── pipelines/                # DetectionPipeline, TrackingPipeline, ...
├── core/
│   ├── registry.py           # 组件注册表
│   ├── builder.py            # 模型/算法/管线构建器
│   └── config.py             # YAML 配置加载与解析
├── engine/                   # 数据源处理（图片/视频/摄像头）
├── task_api.py               # TaskRunner — 唯一公共入口
├── utils/
│   ├── filter.py             # resolve_filter_ids（类别过滤，各算法共享）
│   ├── bbox.py               # 边界框工具
│   ├── nms.py                # NMS（class-aware）
│   ├── device.py             # 设备解析
│   └── visualization/        # Visualizer（检测/跟踪/分割可视化）
└── data/                     # Detection, Track, Pose 等数据结构

configs/
├── models/                   # 模型配置
│   ├── yolo11n/s/m/l.yaml    # YOLO11 系列（YOLOBackbone + YOLOPAN）
│   ├── yolo26n/s/m/l.yaml    # YOLO26 系列（NMS-free，reg_max=1）
│   ├── detr_r50.yaml         # ResNet-50 + TransformerEncoder + DETRHead
│   ├── rfdetr_base.yaml      # DINOv2 + DeformableEncoder + RFDETRHead
│   ├── resnet50_seg.yaml     # ResNet-50 语义分割
│   └── osnet_reid.yaml       # OSNet ReID 特征提取
├── runtime/                  # 运行时配置
│   ├── detect.yaml           # YOLO11 通用检测
│   ├── detect_detr.yaml      # DETR 检测
│   ├── detect_rfdetr.yaml    # RF-DETR 检测
│   ├── detect_person.yaml    # 只检测行人（类别过滤示例）
│   ├── detect_vehicles.yaml  # 只检测车辆（类别过滤示例）
│   ├── tracking.yaml         # ByteTrack 跟踪
│   ├── reid_tracking.yaml    # ReID 跟踪
│   └── segmentation.yaml     # 语义分割
└── components/               # 组件配置
    ├── bytetrack.yaml        # ByteTracker 参数
    └── iou_tracker.yaml      # IOUTracker 参数

tools/
├── convert_ultralytics.py    # ultralytics YOLO11/YOLO26 权重转换
├── convert_detr.py           # Facebook DETR 官方权重转换
└── rfdetr_adapter.py         # RF-DETR 适配器（封装 rfdetr 包）

examples/
├── 01_detection.py           # YOLO11 目标检测
├── 02_tracking.py            # 多目标跟踪
├── 03_segmentation.py        # 语义分割
├── 04_visualization.py       # 可视化工具用法
├── 05_detr_detection.py      # DETR 检测（Facebook 官方权重）
├── 06_yolo26_detection.py    # YOLO26 端到端检测（NMS-free）
└── 07_rfdetr_detection.py    # RF-DETR 检测（Roboflow 适配器）
```

## 内置模型

| 模型 | 配置文件 | Backbone | Neck | Head | 特点 |
|------|----------|----------|------|------|------|
| YOLO11n | `yolo11n.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，轻量 |
| YOLO11s | `yolo11s.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，小型 |
| YOLO11m | `yolo11m.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，中型 |
| YOLO11l | `yolo11l.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，大型 |
| YOLO26n | `yolo26n.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26s | `yolo26s.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26m | `yolo26m.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26l | `yolo26l.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| DETR-R50 | `detr_r50.yaml` | ResNet-50 | TransformerEncoderNeck | DETRHead | 无 NMS，集合预测 |
| RF-DETR | `rfdetr_base.yaml` | DINOv2Backbone | DeformableEncoderNeck | RFDETRHead | 可变形注意力 |
| 分割 | `resnet50_seg.yaml` | ResNet-50 | FPN | SegHead | 语义分割 |
| ReID | `osnet_reid.yaml` | OSNet | — | ReIDHead | 行人重识别 |

## 官方预训练权重

权重路径在 **runtime YAML** 的 `weights` 字段中指定，`TaskRunner` 会自动加载，无需修改代码。

```yaml
# configs/runtime/detect.yaml
weights: weights/yolo11n_converted.pth   # 字符串：作用于 detector

# 多模型场景（tracking + reid）
weights:
  detector: weights/yolo11n_converted.pth
  reid: weights/osnet.pth
```

### YOLO11 / YOLO26 (ultralytics)

转换工具自动检测模型类型，YOLO26 会自动使用 one-to-one head 权重：

```bash
# YOLO11（支持 n/s/m/l）
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_converted.pth

# YOLO26（自动检测 one2one head，无需额外参数）
python tools/convert_ultralytics.py --model yolo26n.pt --out weights/yolo26n_converted.pth
```

### DETR (Facebook)

```bash
python tools/convert_detr.py \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detr_r50.pth --verify

# 支持的官方 checkpoint:
#   DETR-R50:      https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
#   DETR-R101:     https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
#   DETR-DC5-R50:  https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
#   DETR-DC5-R101: https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
```

### RF-DETR (Roboflow)

RF-DETR 架构复杂，采用适配器模式直接调用 `rfdetr` 包推理（权重自动下载），支持尺寸：`nano`, `small`, `base`, `medium`, `large`。

```bash
pip install rfdetr
```

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
| `filter_classes` | list | 类别过滤，支持名称 (str) 和 ID (int) 混用 |

## 模型 YAML 关键参数

### 通用参数

```yaml
task: detection          # 任务类型
weights: path/to/weights # 预训练权重路径（可选）

backbone:
  type: YOLOBackbone
  depth: 0.50
  width: 0.25
  max_channels: 1024

postprocess:
  conf: 0.25             # 置信度阈值
  nms_iou: 0.45          # NMS IoU 阈值
```

### YOLO26 专属参数

```yaml
backbone:
  sppf_cv1_act: false    # SPPF 第一个卷积无激活（YOLO26 特有）
  sppf_residual: true    # SPPF 残差连接（YOLO26 特有）

neck:
  c3k: true              # 使用 C3k 模块（YOLO26 neck）
  a2c2f: true            # 使用 A2C2f 模块（YOLO26 neck）

head:
  reg_max: 1             # 直接回归，不使用 DFL（YOLO26 特有）

postprocess:
  end2end: true          # 跳过 NMS（one-to-one head）
```

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
