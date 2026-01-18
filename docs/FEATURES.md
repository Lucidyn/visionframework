# 功能特性

Vision Framework 提供面向工程的视觉能力，主要功能：

- 检测：支持 YOLO、DETR、RF-DETR 等后端，统一 `Detector` 接口。
- 实例/语义分割：在支持模型上提供掩码输出与可视化。
- 跟踪：ByteTrack、IOU 跟踪、基于 ReID 的跟踪器。
- 姿态估计：关键点检测与骨架可视化（YOLO Pose 等）。
- 区域检测（ROI/Zone）：矩形、多边形、圆形等形状支持。
- 结果导出：JSON、CSV、COCO 等格式。
- 可视化工具：检测框、轨迹、骨架、热力图、实例掩码。
- 性能与工程化：FPS 统计、批量推理选项、FP16 支持（GPU）。

设计原则：模块化设计、轻量高层 API、工程友好的配置与示例（YAML/dict）。

