Examples

This `examples/` folder contains simple, focused example scripts. Each file
demonstrates one feature and is intentionally minimal so new users can run
and understand them quickly.

## 目录结构

```
examples/
├── basic/                  # 基础功能示例
├── video/                  # 视频处理示例
├── advanced/               # 高级功能示例
├── models/                 # 模型相关示例
├── config/                 # 配置相关示例
├── system/                 # 系统功能示例
└── README.md              # 示例说明
```

## 示例分类

### 基础功能示例 (basic/)
- **00_basic_detection.py** — 基础目标检测示例
- **01_detection_with_tracking.py** — 带跟踪的目标检测示例
- **02_simplified_api.py** — 简化API使用示例

### 视频处理示例 (video/)
- **03_video_processing.py** — 视频文件处理示例
- **04_stream_processing.py** — 视频流处理示例
- **12_pyav_video_processing.py** — PyAV视频处理示例，展示高性能视频处理、与OpenCV的性能对比以及RTSP流处理
- **13_vision_pipeline_pyav.py** — VisionPipeline PyAV集成示例，展示在管道中使用PyAV的方法，包括RTSP流支持

### 高级功能示例 (advanced/)
- **05_advanced_features.py** — 高级功能示例
- **06_tools_usage.py** — 工具类使用示例
- **07_enhanced_features.py** — 增强功能示例，包括ReID跟踪和性能监控

### 模型相关示例 (models/)
- **08_segmentation_sam.py** — SAM分割示例，展示自动分割和交互式分割
- **09_clip_features.py** — CLIP特征示例，包括图像-文本相似度和零样本分类
- **10_pose_estimation.py** — 姿态估计示例，支持YOLO Pose和MediaPipe Pose

### 配置相关示例 (config/)
- **11_config_based_processing.py** — 配置文件驱动的视觉处理示例
- **my_config.json** — 配置文件示例

### 系统功能示例 (system/)
- **14_plugin_system_example.py** — 插件系统使用示例，展示如何注册和使用自定义检测器、跟踪器和模型
- **15_memory_pool_example.py** — 内存池管理示例，展示内存分配、释放和优化
- **16_error_handling_example.py** — 统一错误处理示例，展示错误处理、包装和输入验证
- **17_dependency_management_example.py** — 依赖管理示例，展示依赖检测、懒加载和安装建议

## 运行示例

在项目根目录执行，例如：

### 基础功能示例
```bash
python examples/basic/00_basic_detection.py
python examples/basic/01_detection_with_tracking.py
python examples/basic/02_simplified_api.py
```

### 视频处理示例
```bash
python examples/video/03_video_processing.py
python examples/video/04_stream_processing.py
python examples/video/12_pyav_video_processing.py
python examples/video/13_vision_pipeline_pyav.py
```

### 高级功能示例
```bash
python examples/advanced/05_advanced_features.py
python examples/advanced/06_tools_usage.py
python examples/advanced/07_enhanced_features.py
```

### 模型相关示例
```bash
python examples/models/08_segmentation_sam.py
python examples/models/09_clip_features.py
python examples/models/10_pose_estimation.py
```

### 配置相关示例
```bash
python examples/config/11_config_based_processing.py
```

### 系统功能示例
```bash
python examples/system/14_plugin_system_example.py
python examples/system/15_memory_pool_example.py
python examples/system/16_error_handling_example.py
python examples/system/17_dependency_management_example.py
```

## 示例说明

- **基础功能示例**：适合初学者，展示框架的基本使用方法
- **视频处理示例**：展示视频文件和流的处理方法，包括高性能处理
- **高级功能示例**：展示框架的高级特性和工具类使用
- **模型相关示例**：展示各种模型的使用方法，包括分割、CLIP和姿态估计
- **配置相关示例**：展示如何使用配置文件驱动视觉处理
- **系统功能示例**：展示框架的系统级功能，包括插件系统、内存管理、错误处理和依赖管理

## 依赖说明

- **基础功能**：仅依赖核心库（OpenCV、NumPy、YOLOv8）
- **视频处理**：可选依赖PyAV（用于高性能视频处理）
- **模型相关**：
  - SAM分割：依赖segment_anything
  - CLIP特征：依赖transformers
  - 姿态估计：依赖YOLOv8或MediaPipe
- **系统功能**：依赖核心库

## 注意事项

- 所有示例都设计为可以独立运行
- 示例中使用的模型会在首次运行时自动下载
- 对于需要额外依赖的示例，会提供安装建议
- 示例中的配置可以根据实际需求进行修改

这些示例展示了VisionFramework的不同功能和使用方式，从基础的目标检测到高级的视频处理、模型应用和系统功能。