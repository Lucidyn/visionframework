# 架构概览（v0.2.8）

Vision Framework 采用模块化管线架构：

- 数据流：输入帧 → Detector（检测）→ Processor（如分割/姿态/ReID）→ Tracker（匹配/关联）→ Exporter/Visualizer
- 每个能力均实现统一接口以便替换后端模型（例如 YOLO/DETR/RF-DETR）。
- 配置中心：使用统一配置（YAML 或 dict），运行时根据配置动态构建组件。

关键设计点：
- 松耦合：组件通过数据结构（`Detection`, `Track`）交换信息。
- 可插拔：可替换检测与跟踪后端，无需改动上层代码。
- 工程化：支持批量推理、FP16、设备选择等性能配置。

本版本变更（节选）:
- 添加 `categories` 参数，允许在框架层面过滤返回检测。
- 优化示例目录，增加按功能的演示脚本。
