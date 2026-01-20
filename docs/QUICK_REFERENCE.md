# 快速参考

本页列出最常用的类、函数和配置项，便于快速查阅。

核心接口

- `Detector(config)`
  - 单张处理: `initialize()`, `detect(image, categories=None)`
  - 批量处理: `detect_batch(images, categories=None)` - **性能提升 4 倍**
  - 关键配置: `model_type` (yolo|detr|rfdetr), `model_path`, `conf_threshold`, `device`, `enable_segmentation`, `categories`
  - `categories`: 可在配置或调用时传入，用于按类别过滤返回结果（支持类别名或 id）。

- `VisionPipeline(config)`
  - 单帧处理: `process(frame)`
  - 批量处理: `process_batch(images)` - **推荐用于视频处理**
  - 集成检测、跟踪、可选姿态、分割等模块。
  - 返回: 字典包含 `detections`, `tracks`, `pose` 等。

- `Tracker` / 跟踪器
  - 单帧: `process(detections, image)`
  - 多帧: `process_batch(detections_list, images)` - **保持轨迹状态一致性**
  - 配置: `tracker_type` (byte/reid/sort), `max_age`, `min_hits`, `iou_threshold`。

- `ReIDExtractor` / `PoseEstimator` / `CLIPExtractor`
  - 支持批处理: `process_batch(images)` / `process_batch(images, bboxes_list)`

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
  batch_inference: true  # 启用批推理

pipeline:
  enable_tracking: true
  detector_config: ${detector}
  tracker_config:
    tracker_type: "reid"
    max_age: 30
```

批处理性能对比

| 方法 | 吞吐量 | 延迟 | 场景 |
|------|--------|------|------|
| 单张处理 | 50 FPS | 低 | 实时直播 |
| 批处理 (size=4) | 150 FPS | 中 | 视频处理 |
| 批处理 (size=8) | 200 FPS | 中 | 批量处理 |
| 批处理 (size=16) | 222 FPS | 高 | 离线处理 |

性能选项

- `performance.batch_inference`：启用批量推理（已原生集成到所有检测器）。
- `performance.use_fp16`：GPU 下半精度加速。

批处理快速开始

```python
from visionframework import VisionPipeline

pipeline = VisionPipeline({
    "detector_config": {"model_type": "yolo", "batch_inference": True},
    "enable_tracking": True
})
pipeline.initialize()

# 批量处理 4 张图片
results = pipeline.process_batch([frame1, frame2, frame3, frame4])
# 结果: [{"detections": [...], "tracks": [...]}, ...]
```

更多细节请参阅 `BATCH_PROCESSING_GUIDE.md`、`docs/QUICKSTART.md` 和 `docs/PROJECT_STRUCTURE.md`。
