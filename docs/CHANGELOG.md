# 更新日志

## v0.2.8 - 架构重构与模块管理

### 新功能

#### 1. 统一异常系统
- ✅ 创建 `visionframework/exceptions.py` 定义异常层级
- ✅ 基础异常类：`VisionFrameworkError`
- ✅ 专项异常类：
  - `DetectorInitializationError`, `DetectorInferenceError`
  - `TrackerInitializationError`, `TrackerUpdateError`
  - `ConfigurationError`, `ModelNotFoundError`, `ModelLoadError`
  - `DeviceError`, `DependencyError`, `DataFormatError`, `ProcessingError`
- ✅ 所有异常可在主包中导入

#### 2. 模型管理器
- ✅ 创建 `visionframework/models/model_manager.py`
- ✅ `ModelManager` 类用于模型缓存、下载和版本管理
- ✅ 全局 `get_model_manager()` 函数
- ✅ 模型注册表和源管理

#### 3. 特征提取器重构
- ✅ 创建 `visionframework/core/processors/` 目录
- ✅ `FeatureExtractor` 基类定义统一接口
- ✅ 迁移模块：
  - `CLIPExtractor` → `core/processors/clip_extractor.py`
  - `ReIDExtractor` → `core/processors/reid_extractor.py` 
  - `PoseEstimator` → `core/processors/pose_estimator.py`
- ✅ 统一的特征提取接口

#### 4. 导入更新
- ✅ `visionframework/__init__.py` 更新至 v0.2.8
- ✅ 新增异常和模型管理器导出
- ✅ 保持向后兼容性

## v0.2.7 - CLIP 集成、性能优化和评估工具

### 新功能

#### 1. CLIP 零样本分类集成
- ✅ 新增 `CLIPExtractor` 类，支持图像-文本匹配和零样本分类
- ✅ 支持可选的 FP16 推理和自定义模型选择
- ✅ 完整的示例代码 (`clip_example.py`) 和单元测试
- ✅ 集成到主包导出，可直接 `from visionframework import CLIPExtractor` 导入

#### 2. 性能优化与批量推理
- ✅ YOLO/DETR/RF-DETR 检测器支持批量推理 (`batch_inference=True`)
- ✅ 所有检测器支持 FP16 推理 (`use_fp16=True`)
- ✅ 使用 `torch.no_grad()` 和 `torch.cuda.amp.autocast()` 优化推理
- ✅ 完整的批量和 FP16 单元测试

#### 3. ReID 跟踪器增强
- ✅ 参数化配置：`track_history_length`, `embedding_dim`, `matching_strategy`
- ✅ 支持 Hungarian 和 Greedy 匹配策略
- ✅ 可配置 ID 激活阈值和匹配成本阈值
- ✅ 改进的 scipy 回退实现（无需强制依赖）

#### 4. 完整的跟踪评估工具
- ✅ `TrackingEvaluator` 实现标准 MOT 指标：
  - **MOTA** (Multiple Object Tracking Accuracy)
  - **MOTP** (Multiple Object Tracking Precision)
  - **IDF1** (ID F1 Score)
- ✅ 基于 IoU 的检测匹配（Hungarian 算法）
- ✅ ID 切换检测和精准度/召回率计算
- ✅ 完整示例代码和 5 个单元测试

### 依赖管理优化

- ✅ 将 `transformers`, `rfdetr`, `supervision` 移至 `extras_require`
- ✅ 定义功能组：`clip`, `detr`, `rfdetr`, `dev`, `all`
- ✅ 用户可选择性安装：`pip install -e ".[clip]"` 或 `pip install -e ".[all]"`
- ✅ 更新 requirements.txt 和 setup.py

### 文档与示例

- ✅ 新增 `tracking_evaluation_example.py` 展示评估工具使用
- ✅ 新增 `clip_example.py` 展示 CLIP 零样本分类
- ✅ 更新 README.md 记录新功能和安装选项
- ✅ 更新 QUICKSTART.md 展示 CLIP 和性能配置使用

## v0.2.6 - ReID 和实例分割支持

### 新功能

#### 1. ReID (Re-Identification) 跟踪
- ✅ 新增 `ReIDExtractor` 类，基于 ResNet50 提取外观特征
- ✅ 新增 `ReIDTracker` 类，结合 IoU 和外观特征进行目标匹配
- ✅ 支持 `tracker_type="reid"` 配置选项
- ✅ 有效处理遮挡和目标重找回场景

#### 2. 实例分割 (Instance Segmentation)
- ✅ 完善 `YOLODetector` 对分割模型的支持
- ✅ 支持 `enable_segmentation=True` 配置
- ✅ `Detection` 对象新增 `mask` 属性
- ✅ `Visualizer` 支持绘制分割掩码

#### 3. 管道增强
- ✅ `VisionPipeline` 支持将图像帧传递给跟踪器（用于特征提取）
- ✅ 统一了跟踪器接口，支持可选的 `image` 参数

### 文档更新
- ✅ 更新 README.md 添加 ReID 和分割配置示例
- ✅ 更新 QUICKSTART.md 添加新功能使用指南
- ✅ 更新 FEATURES.md 详细介绍 ReID 和分割功能
- ✅ 更新 PROJECT_STRUCTURE.md 反映新增文件

## v0.2.5 - 配置文件和代码清理

### 新功能

#### 1. 配置文件支持
- ✅ Config 类支持 YAML 和 JSON 格式
- ✅ 新增 `config_example.py` 示例展示配置文件使用
- ✅ 自动检测文件格式，优雅处理缺失的依赖

### 代码清理

#### 1. 删除无用示例
- ✅ 删除 `complete_video_processing.py`（功能已在其他示例覆盖）
- ✅ 删除 `model_comparison.py`（非核心功能）
- ✅ 删除 `detr_pose_examples.py`（功能已整合）
- ✅ 删除 `roi_counting_example.py`（功能已在 advanced_features.py 中）

#### 2. 文档精简
- ✅ 删除 `docs/CONTRIBUTING.md`（简化项目结构）

#### 3. 文档更新
- ✅ 更新 README.md 反映新的示例列表
- ✅ 更新 QUICKSTART.md 添加配置文件使用说明
- ✅ 更新 PROJECT_STRUCTURE.md 更新文件列表
- ✅ 更新 OPTIMIZATION_SUGGESTIONS.md 反映已完成优化

### 影响文件
- `visionframework/utils/config.py` - 添加 YAML 支持
- `examples/config_example.py` - 新增配置文件示例
- 多个文档文件更新

## v0.2.4 - 代码质量优化

### 代码优化

#### 1. 日志系统优化
- ✅ 替换所有核心模块中的 `print()` 为日志调用
- ✅ 统一使用 `logger` 模块进行日志记录
- ✅ 添加异常堆栈信息到日志（`exc_info=True`）
- ✅ 使用适当的日志级别（INFO、ERROR、WARNING、DEBUG）

#### 2. 异常处理优化
- ✅ 细化异常处理，使用具体异常类型
- ✅ 区分 ImportError、ValueError、RuntimeError、OSError 等
- ✅ 改进错误信息的详细程度

#### 3. 配置验证
- ✅ 在 `BaseModule` 中添加 `validate_config()` 方法
- ✅ 为 `Detector`、`Tracker`、`VisionPipeline` 实现配置验证
- ✅ 验证参数范围、类型和有效性
- ✅ 自动验证配置并在无效时记录警告

#### 4. 类型提示完善
- ✅ 为核心模块添加完整的类型提示
- ✅ 使用 `Optional`、`List`、`Dict`、`Tuple` 等类型
- ✅ 为类属性添加类型注解

#### 5. 文档字符串完善
- ✅ 为所有公共方法添加详细的文档字符串
- ✅ 包含 Args、Returns、Raises、Note、Example 部分
- ✅ 添加使用示例和参数说明

#### 6. 错误处理统一化
- ✅ 在 `BaseModule` 中添加 `handle_errors()` 装饰器
- ✅ 统一错误处理模式
- ✅ 改进错误处理的一致性

### 影响文件
- `visionframework/core/base.py` - 添加配置验证和错误处理工具
- `visionframework/core/detector.py` - 全面优化
- `visionframework/core/tracker.py` - 全面优化
- `visionframework/core/pipeline.py` - 全面优化
- `visionframework/core/pose_estimator.py` - 全面优化
- `visionframework/core/detectors/*.py` - 所有检测器实现优化
- `visionframework/utils/export.py` - 优化

### 测试
- ✅ 新增 `tests/test_logging_exceptions.py` - 第一阶段优化测试
- ✅ 新增 `tests/test_stage2_optimization.py` - 第二阶段优化测试
- ✅ 所有测试通过

### 向后兼容性
- ✅ 所有 API 保持向后兼容
- ✅ 现有代码无需修改即可使用
- ✅ 配置验证为可选，不影响现有代码

## v0.2.3 - RF-DETR 检测器支持和文档完善

### 新增功能

#### 1. RF-DETR 检测器支持
- ✅ 新增 RFDETRDetector 类
- ✅ 支持 Roboflow 开发的 RF-DETR 高性能实时目标检测模型
- ✅ 集成到统一的 Detector 接口，可通过 model_type="rfdetr" 使用
- ✅ 支持自定义置信度阈值和设备选择
- ✅ 自动模型下载和初始化

#### 2. 文档和测试完善
- ✅ 更新 README.md 添加 RF-DETR 使用说明和示例
- ✅ 更新 FEATURES.md 添加 RF-DETR 详细文档
- ✅ 新增 tests/test_rfdetr.py 测试文件
- ✅ 新增 examples/rfdetr_example.py 使用示例（详细中文注释）
- ✅ 新增 examples/rfdetr_tracking.py 跟踪示例（详细中文注释）
- ✅ 更新项目结构测试

#### 3. 示例代码完善
- ✅ 为所有基础示例添加详细中文注释
  - `basic_usage.py`: 基本使用示例（详细注释）
  - `video_tracking.py`: 视频跟踪示例（详细注释）
  - `advanced_features.py`: 高级功能示例（详细注释）
- ✅ 新增多个实用示例
  - `model_comparison.py`: 多模型性能对比示例
  - `complete_video_processing.py`: 完整视频处理流程示例
  - `batch_processing.py`: 批量图像处理示例
  - `roi_counting_example.py`: ROI 检测和计数综合示例

#### 4. 文档更新
- ✅ 更新所有文档确保内容准确、完整、一致
- ✅ 修复 README 版本徽章链接
- ✅ 更新项目结构文档
- ✅ 完善快速开始指南

### 依赖更新
- ✅ 添加 rfdetr>=0.1.0 依赖
- ✅ 添加 supervision>=0.18.0 依赖（RF-DETR 所需）

### 代码改进
- ✅ 完善检测器接口，支持三种模型类型：yolo, detr, rfdetr
- ✅ 改进错误处理和初始化流程

## v0.2.2 - 第一阶段功能扩展

### 新增功能

#### 1. ByteTrack 跟踪算法
- ✅ 实现 ByteTrack 多目标跟踪算法
- ✅ 支持高/低置信度检测的关联
- ✅ 更好的跟踪稳定性和准确性
- ✅ STrack 数据结构

#### 2. 实例分割支持
- ✅ 支持 YOLO 实例分割模型
- ✅ Detection 对象添加 mask 属性
- ✅ 掩码可视化支持

#### 3. 轨迹分析工具
- ✅ TrajectoryAnalyzer 类
- ✅ 速度计算（像素/帧或 m/s）
- ✅ 方向计算（角度）
- ✅ 距离计算
- ✅ 轨迹平滑
- ✅ 位置预测

#### 4. 评估工具
- ✅ DetectionEvaluator：检测评估
  - Precision, Recall, F1
  - mAP (mean Average Precision)
  - 支持多类别评估
- ✅ TrackingEvaluator：跟踪评估
  - MOTA (Multiple Object Tracking Accuracy)
  - IDF1 (ID F1 Score)

### 文档更新
- ✅ 更新 README 添加新功能说明
- ✅ 新增 stage1_features.py 示例

## v0.2.1 - DETR 和姿态估计支持

### 新增功能

#### 1. DETR 检测支持
- ✅ 支持 DETR (Detection Transformer) 模型
- ✅ 支持 facebook/detr-resnet-50 和 detr-resnet-101
- ✅ 集成到 Detector 类，可通过 model_type="detr" 使用

#### 2. 姿态估计功能
- ✅ 新增 PoseEstimator 类
- ✅ 支持 YOLO Pose 模型
- ✅ 关键点检测（17 个 COCO 关键点）
- ✅ 骨架绘制功能
- ✅ Pose 和 KeyPoint 数据结构

#### 3. 可视化增强
- ✅ 新增 draw_pose() 和 draw_poses() 方法
- ✅ 支持关键点可视化
- ✅ 支持骨架连接绘制

### 依赖更新
- ✅ 添加 transformers>=4.30.0 依赖

### 文档更新
- ✅ 更新 README 添加 DETR 和姿态估计说明
- ✅ 新增 detr_pose_examples.py 示例
- ✅ 更新 FEATURES.md 文档

## v0.2.0 - 高级功能扩展

### 新增功能

#### 1. 结果导出功能 (ResultExporter)
- ✅ 支持导出检测结果到 JSON 格式
- ✅ 支持导出跟踪结果到 JSON 格式
- ✅ 支持导出检测结果到 CSV 格式
- ✅ 支持导出跟踪结果到 CSV 格式
- ✅ 支持导出视频处理结果到 JSON
- ✅ 支持导出为 COCO 格式（用于评估和训练）

**文件**: `visionframework/utils/export.py`

#### 2. 性能分析工具 (PerformanceMonitor)
- ✅ 实时 FPS 监控（当前、平均、最小、最大）
- ✅ 处理时间统计（每帧平均、最小、最大时间）
- ✅ 组件时间分析（检测、跟踪、可视化各组件耗时）
- ✅ 性能摘要打印
- ✅ 滑动窗口统计
- ✅ 简单的 Timer 上下文管理器

**文件**: `visionframework/utils/performance.py`

#### 3. 视频处理工具 (VideoProcessor, VideoWriter)
- ✅ 便捷的视频读取接口
- ✅ 视频写入功能
- ✅ 视频信息获取（FPS、尺寸、总帧数等）
- ✅ 批量视频处理函数
- ✅ 支持摄像头输入
- ✅ 支持跳帧处理

**文件**: `visionframework/utils/video_utils.py`

#### 4. 区域检测 (ROIDetector)
- ✅ 支持矩形 ROI
- ✅ 支持多边形 ROI
- ✅ 支持圆形 ROI
- ✅ 检测/跟踪结果过滤
- ✅ 按 ROI 分组结果
- ✅ 点/边界框包含检测
- ✅ ROI 掩码生成

**文件**: `visionframework/core/roi_detector.py`

#### 5. 计数功能 (Counter)
- ✅ 进入 ROI 计数
- ✅ 离开 ROI 计数
- ✅ 当前在 ROI 内计数
- ✅ 多 ROI 支持
- ✅ 基于跟踪的计数
- ✅ 计数状态重置

**文件**: `visionframework/core/counter.py`

### 文档更新
- ✅ 新增 `FEATURES.md` - 高级功能详细文档
- ✅ 更新 `README.md` - 添加新功能说明
- ✅ 新增 `examples/advanced_features.py` - 高级功能使用示例

### 示例代码
- ✅ ROI 检测示例
- ✅ 对象计数示例
- ✅ 性能监控示例
- ✅ 结果导出示例
- ✅ 视频处理示例

## v0.1.0 - 初始版本

### 基础功能
- ✅ 目标检测 (Detector)
- ✅ 目标跟踪 (Tracker)
- ✅ 完整管道 (VisionPipeline)
- ✅ 可视化工具 (Visualizer)
- ✅ 配置管理 (Config)
- ✅ 图像工具 (ImageUtils)

