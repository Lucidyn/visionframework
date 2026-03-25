# VisionFramework v2.0

模块化、组件式计算机视觉框架。通过 YAML 配置驱动，支持目标检测、语义分割、多目标跟踪等任务。

## 特性

- **组件化架构** — Backbone / Neck / Head 自由组合，通过注册表动态实例化
- **YAML 驱动** — 唯一入口 `TaskRunner(yaml_path)`，零代码配置切换模型和任务
- **内置模型** — YOLO11、YOLO26、DETR、RT-DETR（Ultralytics HGNet **rtdetr-l/x**），以及 CSPDarknet、ResNet 等基础组件
- **官方权重** — 支持加载 Facebook DETR（458/458完美映射）、ultralytics YOLO、RT-DETR HG（**rtdetr-l/x.pt** 转换）权重
- **YOLO26 端到端** — NMS-free one-to-one 检测头，`end2end: true` 一键启用
- **多任务支持** — 检测、分割、跟踪、ReID 跟踪，统一 pipeline 管理
- **最少依赖** — 核心仅需 `torch`、`opencv-python`、`numpy`、`pyyaml`

## 快速开始

### 安装

```bash
pip install torch torchvision opencv-python numpy pyyaml

# 推荐：开发模式安装（避免 tools/examples 依赖当前工作目录）
pip install -e .

# 可选：安装后可直接使用命令行入口
#   vf-test-yolo26 --quick
#   vf-convert-ultralytics --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth
```

所有示例的统一用法模式：**加载图片 → `TaskRunner` → 可视化保存**。仓库根目录提供真实样图 **`test_bus.jpg`**（Ultralytics 公开 [bus.jpg](https://ultralytics.com/images/bus.jpg)，便于检测/可视化示例）；若缺失可运行 `python examples/04_visualization.py`，脚本会尝试自动下载。`04_visualization.py` 保存的 `visualization_demo.jpg` 是 **手动示意框**（验证绘制管线），不是模型推理。

**重要：** `runs/.../detect.yaml` 里若配置了 `weights` 但对应文件不存在，框架不会中止，只会用**随机初始化**的权重推理，结果几乎总是 **0 个检测**，保存的图片上就像什么也没检测到。示例 `01` / `05` / `06` / `07` 在启动时调用 `visionframework.core.config.require_detector_weights` 检查权重；自定义脚本可同样复用该函数。请先按各示例注释转换或下载权重。

---

### YOLO11 目标检测

```bash
# 转换 ultralytics 官方权重
pip install ultralytics
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth
```

```python
import cv2
from visionframework import TaskRunner, Visualizer

img = cv2.imread("test_bus.jpg")
task = TaskRunner("runs/detection/yolo11/detect.yaml")      # weights 字段已在 yaml 中指定
result = task.process(img)

vis = Visualizer()
cv2.imwrite("result.jpg", vis.draw_detections(img.copy(), result["detections"]))
```

> 参考示例：`examples/01_detection.py`

---

### YOLO26 端到端检测（NMS-free）

YOLO26 使用 one-to-one 检测头，无需 NMS 后处理，推理更快。

```bash
python tools/convert_ultralytics.py --model yolo26n.pt --out weights/detection/yolo26/yolo26n_converted.pth
```

```python
task = TaskRunner("runs/detection/yolo26/detect.yaml")   # end2end 已在 yaml 中配置
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
    --output weights/detection/detr/detr_r50.pth --verify
```

```python
task = TaskRunner("runs/detection/detr/detect.yaml")
result = task.process(img)
```

> 参考示例：`examples/05_detr_detection.py`

---

### RT-DETR 检测（Ultralytics **rtdetr-l / rtdetr-x**，HGNet）

与 [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/) 官方 **COCO 预训练**（`rtdetr-l.pt` / `rtdetr-x.pt`）**权重布局**对齐：HGNet + PAN/AIFI + `RTDETRDecoder` 均为框架内 **纯 PyTorch** 实现，**推理与转换 `.pt` 不需要安装 `ultralytics`**。**无 NMS**。使用官方 `.pt` 时须遵守 **AGPL-3.0**（见 **`NOTICE`**）。可选：`pip install -e ".[rtdetr-verify]"` 以运行与 `ultralytics` 逐位对齐的测试。

```bash
python -m visionframework.tools.convert_ultralytics_rtdetr_hg \
    --weights path/to/rtdetr-l.pt --variant l \
    --out weights/detection/rtdetr/rtdetr_l_vf.pth --verify
python -m visionframework.tools.convert_ultralytics_rtdetr_hg \
    --weights path/to/rtdetr-x.pt --variant x \
    --out weights/detection/rtdetr/rtdetr_x_vf.pth --verify
```

```python
# l：runs/detection/rtdetr/detect.yaml  →  rtdetr_l_vf.pth
# x：runs/detection/rtdetr/detect_x.yaml →  rtdetr_x_vf.pth
task = TaskRunner("runs/detection/rtdetr/detect.yaml")
result = task.process(img)
vis = Visualizer()
cv2.imwrite("rtdetr_result_l.jpg", vis.draw_detections(img.copy(), result["detections"]))
```

> 参考示例：`examples/07_rtdetr_detection.py`（**一次运行**会依次推理 l 与 x，并写出 `rtdetr_result_l.jpg`、`rtdetr_result_x.jpg`，同时用 l 结果覆盖 `rtdetr_result.jpg`）。安装后也可用 `vf-convert-rtdetr --weights ... --variant l|x --out ...`。

---

### 视频跟踪

```python
task = TaskRunner("runs/tracking/bytetrack/tracking.yaml")
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
│   ├── mlp.py                # MLP（DETR 等 Transformer 头共享）
│   ├── attention.py          # SEBlock, CBAM, TransformerBlock
│   ├── positional.py         # 2D 正弦位置编码
│   └── deformable_attn.py    # 可变形注意力
├── models/
│   ├── backbones/            # YOLOBackbone, ResNet, RTDETRHGBackbone, CSPDarknet, DINOv2Backbone
│   ├── necks/                # YOLOPAN, PAN, FPN, TransformerEncoderNeck, …
│   ├── heads/                # YOLOHead, DETRHead, RTDETRHGDecoder, SegHead, ReIDHead
│   ├── layers/               # 共享块（随 YOLO/DETR 等组件使用）
│   └── ops/                  # 可变形注意力等（如 MSDeformAttn）
├── algorithms/
│   ├── base.py               # BaseAlgorithm（设备管理、fp16、predict_batch 共享基类）
│   ├── detection/            # Detector, DETRDetector, RTDETRDetector
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

configs/                      # 仅模型配置（结构/backbone/head）
├── detection/
│   ├── yolo11/               # yolo11n/s/m/l/x.yaml
│   ├── yolo26/               # yolo26n/s/m/l/x.yaml
│   ├── detr/                 # detr_r50.yaml
│   └── rtdetr/               # rtdetr_l.yaml, rtdetr_x.yaml
├── tracking/bytetrack/       # bytetrack.yaml（跟踪器参数）
├── segmentation/resnet50/    # resnet50_seg.yaml
└── reid/osnet/               # osnet_reid.yaml

runs/                         # 仅运行/流水线配置（入口 YAML，TaskRunner 加载）
├── detection/
│   ├── yolo11/               # detect.yaml, detect_person.yaml, detect_vehicles.yaml
│   ├── yolo26/               # detect.yaml
│   ├── detr/                 # detect.yaml
│   └── rtdetr/               # detect.yaml（l）, detect_x.yaml
├── tracking/bytetrack/       # tracking.yaml, reid_tracking.yaml
└── segmentation/resnet50/   # segmentation.yaml

weights/                      # 权重按 任务/算法 存放（见 weights/README.md）
├── detection/yolo11/         # yolo11n_converted.pth 等
├── detection/yolo26/
├── detection/detr/
├── detection/rtdetr/
├── segmentation/resnet50/
└── reid/osnet/

tools/
├── convert_ultralytics.py           # ultralytics YOLO11/YOLO26 权重转换
├── convert_ultralytics_rtdetr_hg.py # Ultralytics rtdetr-l/x .pt → 框架 .pth
├── test_yolo26.py                   # YOLO11/YOLO26 与 Ultralytics 对齐测试
└── convert_detr.py                  # Facebook DETR 官方权重转换

examples/
├── 01_detection.py           # YOLO11 目标检测
├── 02_tracking.py            # 多目标跟踪
├── 03_segmentation.py        # 语义分割
├── 04_visualization.py       # 可视化工具用法
├── 05_detr_detection.py      # DETR 检测（Facebook 官方权重）
├── 06_yolo26_detection.py    # YOLO26 端到端检测（NMS-free）
└── 07_rtdetr_detection.py    # RT-DETR（HGNet l/x；输出 rtdetr_result_{l,x}.jpg）

test/                         # pytest；说明见 test/README.md
```

## 内置模型

| 模型 | 模型配置（configs/） | Backbone | Neck | Head | 特点 |
|------|----------------------|----------|------|------|------|
| YOLO11n | `detection/yolo11/yolo11n.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，轻量 |
| YOLO11s | `detection/yolo11/yolo11s.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，小型 |
| YOLO11m | `detection/yolo11/yolo11m.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，中型 |
| YOLO11l | `detection/yolo11/yolo11l.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，大型 |
| YOLO11x | `detection/yolo11/yolo11x.yaml` | YOLOBackbone | YOLOPAN | YOLOHead | C3k2+C2PSA，超大 |
| YOLO26n | `detection/yolo26/yolo26n.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26s | `detection/yolo26/yolo26s.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26m | `detection/yolo26/yolo26m.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26l | `detection/yolo26/yolo26l.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| YOLO26x | `detection/yolo26/yolo26x.yaml` | YOLOBackbone | YOLOPAN(c3k) | YOLOHead | NMS-free，reg_max=1 |
| DETR-R50 | `detection/detr/detr_r50.yaml` | ResNet-50 | TransformerEncoderNeck | DETRHead | 无 NMS，集合预测 |
| RT-DETR-l | `detection/rtdetr/rtdetr_l.yaml` | RTDETRHGBackbone（含 PAN/AIFI） | — | RTDETRHGDecoder | 无 NMS；纯 PyTorch；官方 `.pt` 见 `NOTICE` |
| RT-DETR-x | `detection/rtdetr/rtdetr_x.yaml` | RTDETRHGBackbone（同上，宽版） | — | RTDETRHGDecoder | 同上 |
| 分割 | `segmentation/resnet50/resnet50_seg.yaml` | ResNet-50 | FPN | SegHead | 语义分割 |
| ReID | `reid/osnet/osnet_reid.yaml` | OSNet | — | ReIDHead | 行人重识别 |

## 官方预训练权重

权重路径在 **运行配置**（`runs/<task>/<algo>/*.yaml`）的 `weights` 字段中指定，`TaskRunner` 会自动加载，无需修改代码。运行配置内通过 `models.detector` 等引用 `configs/` 下的模型配置。

```yaml
# runs/detection/yolo11/detect.yaml
weights: weights/detection/yolo11/yolo11n_converted.pth   # 字符串：作用于 detector

# 多模型场景（tracking + reid）
weights:
  detector: weights/detection/yolo11/yolo11n_converted.pth
  reid: weights/reid/osnet/osnet_reid.pth
```

### YOLO11 / YOLO26 (ultralytics)

转换工具自动检测模型类型，YOLO26 会自动使用 one-to-one head 权重：

```bash
# YOLO11（支持 n/s/m/l/x）
python tools/convert_ultralytics.py --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth

# YOLO26（支持 n/s/m/l/x，自动检测 one2one head）
python tools/convert_ultralytics.py --model yolo26n.pt --out weights/detection/yolo26/yolo26n_converted.pth

# 验证与 Ultralytics 推理一致（10 个模型：11n/s/m/l/x、26n/s/m/l/x）
python tools/test_yolo26.py
```

### DETR (Facebook)

```bash
python tools/convert_detr.py \
    --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output weights/detection/detr/detr_r50.pth --verify

# 支持的官方 checkpoint:
#   DETR-R50:      https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
#   DETR-R101:     https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
#   DETR-DC5-R50:  https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
#   DETR-DC5-R101: https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
```

### RT-DETR HG（Ultralytics **rtdetr-l / rtdetr-x**）

```bash
python -m visionframework.tools.convert_ultralytics_rtdetr_hg \
    --weights path/to/rtdetr-l.pt --variant l \
    --out weights/detection/rtdetr/rtdetr_l_vf.pth --verify
python -m visionframework.tools.convert_ultralytics_rtdetr_hg \
    --weights path/to/rtdetr-x.pt --variant x \
    --out weights/detection/rtdetr/rtdetr_x_vf.pth --verify

# 安装后也可: vf-convert-rtdetr --weights ... --variant l|x --out ...
```

转换后的 `.pth` 需与 `configs/detection/rtdetr/rtdetr_{l,x}.yaml` 一致。**推理与上述转换命令不需要安装 `ultralytics`**。使用官方 `.pt` 须遵守 **AGPL-3.0**（见 **`NOTICE`**）。可选安装 **`pip install -e ".[rtdetr-verify]"`** 以运行与 Ultralytics 对齐的额外测试。

## 运行时配置参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `pipeline` | str | 任务类型：`detection` / `tracking` / `segmentation` / `reid_tracking` |
| `algorithm` | str | 检测算法：`Detector`（默认）、`DETRDetector` 或 `RTDETRDetector` |
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

## 测试

在项目根目录执行（需开发依赖）：

```bash
pip install -e ".[dev]"
pytest
```

- **`test/`**：`test/core`、`test/models`、`test/algorithms`、`test/utils`、`test/pipelines` 等，覆盖配置解析、注册表、各检测/跟踪算法与可视化等。
- **RT-DETR**：`test/models/test_rtdetr.py` 为结构/构建测试；`test/algorithms/test_rtdetr_pretrained.py` 中  
  - 默认可跑的 **smoke**：随机权重 + `TaskRunner`，不要求预训练文件；  
  - 标记 **`rtdetr_official`**：需环境变量 **`RTDETR_L_PT`**、**`RTDETR_X_PT`** 分别指向官方 COCO 的 `rtdetr-l.pt`、`rtdetr-x.pt`（用于现场转换并断言 bus 图至少检出目标）。示例：
  ```bash
  set RTDETR_L_PT=C:\path\to\rtdetr-l.pt
  set RTDETR_X_PT=C:\path\to\rtdetr-x.pt
  pytest -m rtdetr_official
  ```
  （Linux/macOS 使用 `export`。）未设置时对应用例 **skip**。
- 更多说明见 **`test/README.md`**；标记定义见 **`pyproject.toml`** 中 `[tool.pytest.ini_options]`。

## 依赖

| 包 | 版本 | 说明 |
|----|------|------|
| torch | ≥1.10 | 核心 |
| opencv-python | ≥4.5 | 图像处理 |
| numpy | ≥1.20 | 数值计算 |
| pyyaml | ≥5.0 | 配置文件 |
| scipy | 可选 | ByteTrack 匹配 |
| ultralytics | 可选 | YOLO 权重转换；RT-DETR 对齐测试（`pip install -e ".[rtdetr-verify]"`） |

## 许可

VisionFramework 以 **MIT License** 发布。使用 RT-DETR HG（`RTDETRHGBackbone` / `RTDETRHGDecoder`）及 Ultralytics 官方权重时，还须遵守 **Ultralytics AGPL-3.0**（见 **`NOTICE`**）。
