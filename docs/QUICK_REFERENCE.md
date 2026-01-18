# 快速参考

本页列出最常用的类、函数和配置项，便于快速查阅。

核心接口

- `Detector(config)`
  - 主要方法: `initialize()`, `detect(image, categories=None)`
  - 关键配置: `model_type` (yolo|detr|rfdetr), `model_path`, `conf_threshold`, `device`, `enable_segmentation`, `categories`
  - `categories`: 可在配置或调用时传入，用于按类别过滤返回结果（支持类别名或 id）。

- `VisionPipeline(config)`
  - 集成检测、跟踪、可选姿态、分割等模块。
  - 主要方法: `initialize()`, `process(frame)` → 返回字典包含 `detections`, `tracks`, `pose` 等。

- `Tracker` / 跟踪器
  - 配置: `tracker_type` (byte/reid/sort), `max_age`, `min_hits`, `iou_threshold`。

- `Visualizer`
  - 绘制检测框、轨迹、骨架、实例掩码。

常用返回类型（数据结构）

- `Detection`: {`bbox`, `score`, `class_id`, `class_name`, `mask`(可选)}
- `Track`: {`track_id`, `bbox`, `class_id`, `score`, `history`}

配置速查（YAML / dict）

```yaml
detector:
  model_type: "yolo"
  model_path: "yolov8n.pt"
  conf_threshold: 0.25
  device: "cpu"
  categories: ['person', 'car']  # 可选，按类别过滤

pipeline:
  enable_tracking: true
  detector_config: ${detector}
  tracker_config:
    tracker_type: "reid"
    max_age: 30
```

性能选项

- `performance.batch_inference`：是否启用批量推理（受后端支持）。
- `performance.use_fp16`：GPU 下半精度加速。

更多细节请参阅 `docs/QUICKSTART.md` 和 `docs/PROJECT_STRUCTURE.md`。
