Examples
========

该目录包含按功能分类的示例脚本，方便快速上手和查阅具体用法。

## 目录结构

```text
examples/
├── basic/                  # 基础功能示例（检测 / 跟踪 / 简化 API / 姿态估计 / 分割）
│   ├── 00_basic_detection.py
│   ├── 01_detection_with_tracking.py
│   ├── 02_simplified_api.py
│   ├── 03_pose_estimation.py
│   ├── 04_segmentation.py
│   └── 05_video_processing.py
├── advanced/               # 高级功能示例（多模态 / 批处理 / 自定义组件 / 结果导出 / 模型工具）
│   ├── 08_model_tools_example.py
│   ├── 09_multimodal_processing.py
│   ├── 10_batch_processing.py
│   ├── 11_custom_component.py
│   └── 12_result_export.py
└── README.md               # 示例说明文档
```

## 基础功能示例（basic/）

- `00_basic_detection.py`  
  使用 `YOLODetector` 对单张图片进行目标检测，演示最基础的检测用法。

- `01_detection_with_tracking.py`  
  使用 `VisionPipeline` 同时完成检测和多目标跟踪，演示管道式 API。

- `02_simplified_api.py`  
  使用高层函数 `process_image` 一行完成检测/跟踪，演示最简化的调用方式。

- `03_pose_estimation.py`  
  演示使用 `PoseEstimator` 进行人体姿态估计，支持关键点检测和骨架绘制。

- `04_segmentation.py`  
  演示使用 `SAMSegmenter` 进行图像分割，支持自动分割和交互式分割（点/框提示）。

- `05_video_processing.py`  
  演示使用 `process_video` 函数进行视频处理，支持视频文件和摄像头输入。

运行方式示例（在项目根目录）：

```bash
conda activate frametest   # 如使用 Conda 环境

python examples/basic/00_basic_detection.py
python examples/basic/01_detection_with_tracking.py
python examples/basic/02_simplified_api.py
python examples/basic/03_pose_estimation.py
python examples/basic/04_segmentation.py
python examples/basic/05_video_processing.py
```

> 提示：基础示例需要本地存在 YOLO 权重（如 `yolov8n.pt`），并安装 `ultralytics` 等依赖。

## 高级功能示例（advanced/）

- `08_model_tools_example.py`  
  演示模型工具功能，包括模型优化（量化、剪枝）、微调配置、数据增强和轨迹分析，展示如何使用框架的高级模型工具。

- `09_multimodal_processing.py`  
  演示多模态处理，结合目标检测、跟踪、姿态估计和特征提取，展示如何构建完整的视觉处理系统。

- `10_batch_processing.py`  
  演示批量处理功能，比较批量处理与单张处理的性能差异，展示如何使用 `BatchPipeline` 进行高效的批量处理。

- `11_custom_component.py`  
  演示如何创建和注册自定义组件，包括自定义检测器和处理器，展示框架的可扩展性。

- `12_result_export.py`  
  演示结果导出功能，支持将检测、跟踪和姿态估计结果导出为多种格式（JSON、CSV、COCO等）。

运行方式：

```bash
conda activate frametest
python examples/advanced/08_model_tools_example.py
python examples/advanced/09_multimodal_processing.py
python examples/advanced/10_batch_processing.py
python examples/advanced/11_custom_component.py
python examples/advanced/12_result_export.py
```

