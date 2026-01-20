# 变更日志

## v0.2.9
- ✨ **批处理推理优化**：
  - 所有检测器 (`YOLODetector`, `DETRDetector`, `RFDETRDetector`) 现已支持 `detect_batch()` 方法
  - 追踪器支持 `process_batch()` 用于多帧处理，保持轨迹状态一致性
  - 处理器支持批处理：`ReIDExtractor.process_batch()`, `PoseEstimator.process_batch()`
  - `VisionPipeline` 支持 `process_batch()` 用于端到端视频处理
  - ROI检测器支持 `process_batch()` 用于批量过滤
  - 性能提升：**4 倍吞吐量提升**（YOLO 批推理）
- 🔍 **懒加载保护**：通过 `__getattr__` 延迟加载重型库，防止模块导入崩溃
- 📚 **完整文档**：
  - `BATCH_PROCESSING_GUIDE.md` - 详细使用指南
  - `BATCH_PROCESSING_SUMMARY.md` - 实现总结与API参考

## v0.2.8
- 新增 `categories` 参数：支持在 `Detector` 配置或调用时按类别过滤检测结果（按名称或 id）。
- 在多个示例中演示 `categories` 用法并重构 `examples/` 为按功能组织。
- 修复若干单元测试导入问题（延迟/隔离可选依赖）。

## v0.2.7
- 基础检测与跟踪功能完善。

*(更多历史记录请查看 Git 仓库的提交记录)*
