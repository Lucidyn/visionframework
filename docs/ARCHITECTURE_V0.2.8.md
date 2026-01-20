# 架构概览（v0.2.9）

Vision Framework 采用模块化管线架构，原生支持批处理推理：

## 数据流

**单帧处理**：
```
输入帧 → Detector → Processor → Tracker → Exporter/Visualizer
```

**批量处理** ⭐ **推荐**：
```
输入帧批 → Detector.detect_batch() → Pipeline → Tracker.process_batch() → 结果批
```

## 关键设计

- **模块化**：每个能力均实现统一接口以便替换后端模型（例如 YOLO/DETR/RF-DETR）。
- **配置中心**：使用统一配置（YAML 或 dict），运行时根据配置动态构建组件。
- **松耦合**：组件通过数据结构（`Detection`, `Track`）交换信息。
- **可插拔**：可替换检测与跟踪后端，无需改动上层代码。
- **工程化**：支持批量推理、FP16、设备选择等性能配置。

## 批处理架构

**有状态 vs 无状态**：

| 组件 | 处理方式 | 方法 | 性能 |
|------|---------|------|------|
| Detector | 无状态（可并行） | `detect_batch()` | **4x 提升** |
| Tracker | 有状态（顺序） | `process_batch()` | 便利性↑ |
| Processor | 半状态（按batch） | `process_batch()` | **2-3x 提升** |

**为什么追踪器是顺序的？**
- 追踪维持跨帧状态（轨迹历史、ID 分配）
- 必须按时间顺序处理以保持一致性
- `process_batch()` 仍然提供便利的 API 来处理多帧

## 本版本变更（v0.2.9）

✨ **批处理推理优化**：
- 所有检测器支持 `detect_batch()` 方法
- 追踪器支持 `process_batch()` 用于多帧处理
- 处理器（ReID、Pose）支持批处理
- `VisionPipeline.process_batch()` 端到端批处理
- **性能提升：4 倍吞吐量**

🔍 **懒加载保护**：
- 防止模块导入时加载重型库导致崩溃
- 通过 `__getattr__` 延迟加载

📚 **完整文档**：
- `BATCH_PROCESSING_GUIDE.md` - 详细使用指南
- `BATCH_PROCESSING_SUMMARY.md` - 实现总结

## v0.2.8 变更（节选）
- 添加 `categories` 参数，允许在框架层面过滤返回检测。
- 优化示例目录，增加按功能的演示脚本。
