# 项目结构

```
visionframework/
├── visionframework/          # 主包目录
│   ├── __init__.py          # 包初始化文件
│   ├── core/                # 核心功能模块
│   │   ├── __init__.py
│   │   ├── base.py          # 基础模块类
    │   │   ├── reid.py          # ReID特征提取
    │   │   ├── detector.py      # 统一检测器接口
    │   │   ├── tracker.py       # 目标跟踪器
│   │   ├── pipeline.py      # 完整管道
│   │   ├── roi_detector.py  # ROI区域检测
│   │   ├── counter.py       # 计数功能
│   │   ├── pose_estimator.py # 姿态估计
│   │   ├── detectors/       # 检测器实现
│   │   │   ├── __init__.py
│   │   │   ├── base_detector.py  # 检测器基类
│   │   │   ├── yolo_detector.py  # YOLO 检测器
│   │   │   ├── detr_detector.py  # DETR 检测器
│   │   │   └── rfdetr_detector.py # RF-DETR 检测器
│   │   └── trackers/         # 跟踪器实现
│   │       ├── __init__.py
│   │       ├── base_tracker.py   # 跟踪器基类
    │   │       ├── iou_tracker.py    # IoU 跟踪器
    │   │       ├── byte_tracker.py   # ByteTrack 跟踪器
    │   │       └── reid_tracker.py   # ReID 跟踪器
    │   ├── data/                # 数据结构模块
│   │   ├── __init__.py
│   │   ├── detection.py     # Detection 数据结构
│   │   ├── track.py         # Track 和 STrack 数据结构
│   │   ├── pose.py          # Pose 和 KeyPoint 数据结构
│   │   └── roi.py           # ROI 数据结构
│   └── utils/               # 工具模块
│       ├── __init__.py
│       ├── config.py        # 配置管理
│       ├── visualization/   # 可视化工具模块
│       ├── image_utils.py    # 图像处理工具
│       ├── export.py         # 结果导出
│       ├── performance.py    # 性能分析
│       └── video_utils.py    # 视频处理工具
│
├── examples/                # 示例代码目录
│   ├── basic_usage.py       # 基本使用示例（详细注释）
│   ├── video_tracking.py    # 视频跟踪示例（详细注释）
│   ├── config_example.py    # 使用配置文件示例（推荐）
│   ├── advanced_features.py # 高级功能示例（详细注释）
│   ├── rfdetr_example.py    # RF-DETR 检测器示例
│   ├── rfdetr_tracking.py   # RF-DETR 检测 + 跟踪示例
│   ├── yolo_pose_example.py # YOLO Pose 姿态估计示例
│   └── batch_processing.py  # 批量图像处理示例
│
├── tests/                   # 测试文件目录
│   ├── __init__.py
│   ├── conftest.py          # pytest 配置
│   ├── quick_test.py        # 快速测试
│   ├── test_structure.py    # 结构测试
│   ├── test_utilities.py    # 工具和组件测试
│   ├── test_code_quality.py # 代码质量测试
│   └── test_rfdetr.py      # RF-DETR 测试
│
├── docs/                    # 文档目录
│   ├── QUICKSTART.md        # 快速开始指南
│   ├── FEATURES.md          # 高级功能文档
│   ├── CHANGELOG.md         # 更新日志
│   └── PROJECT_STRUCTURE.md # 项目结构说明（本文件）
│
├── README.md                # 项目主文档
├── requirements.txt         # 依赖列表
├── setup.py                 # 安装脚本
└── .gitignore              # Git忽略文件
```

## 目录说明

### visionframework/
主包目录，包含所有核心功能代码。

- **core/**: 核心功能模块
  - `base.py`: 所有模块的基类，定义通用接口
  - `reid.py`: ReID 特征提取模块
  - `detector.py`: 统一检测器接口，支持多种检测模型
  - `tracker.py`: 统一跟踪器接口
  - `pipeline.py`: 完整的检测+跟踪管道
  - `roi_detector.py`: ROI区域检测和过滤
  - `counter.py`: 对象计数功能
  - `pose_estimator.py`: 姿态估计功能
  - `detectors/`: 检测器实现目录
    - `base_detector.py`: 检测器基类
    - `yolo_detector.py`: YOLO 检测器实现
    - `detr_detector.py`: DETR 检测器实现
    - `rfdetr_detector.py`: RF-DETR 检测器实现
  - `trackers/`: 跟踪器实现目录
    - `base_tracker.py`: 跟踪器基类
    - `iou_tracker.py`: IoU 跟踪器实现
    - `byte_tracker.py`: ByteTrack 跟踪器实现
    - `reid_tracker.py`: ReID 跟踪器实现

- **data/**: 数据结构模块
  - `detection.py`: Detection 类，表示检测结果（边界框、置信度、类别等）
  - `track.py`: Track 和 STrack 类，表示跟踪轨迹
  - `pose.py`: Pose 和 KeyPoint 类，表示姿态估计结果
  - `roi.py`: ROI 类，表示感兴趣区域

- **utils/**: 工具模块
  - `config.py`: 配置管理工具
  - `image_utils.py`: 图像处理工具函数
  - `export.py`: 结果导出工具（JSON/CSV/COCO）
  - `performance.py`: 性能监控和分析工具
  - `video_utils.py`: 视频处理工具
  - `logger.py`: 日志工具
  - `trajectory_analyzer.py`: 轨迹分析工具
  - `visualization/`: 可视化子模块
    - `unified_visualizer.py`: 统一可视化器
  - `evaluation/`: 评估子模块
    - `detection_evaluator.py`: 检测评估器
    - `tracking_evaluator.py`: 跟踪评估器

### examples/
示例代码目录，包含各种使用示例。

- `basic_usage.py`: 基本功能使用示例（包含详细中文注释）
- `video_tracking.py`: 视频跟踪示例（包含详细中文注释）
- `config_example.py`: 使用配置文件示例（推荐，支持 YAML/JSON）
- `advanced_features.py`: 高级功能使用示例（ROI、计数、性能监控等，详细注释）
- `rfdetr_example.py`: RF-DETR 检测器使用示例（详细注释）
- `rfdetr_tracking.py`: RF-DETR 检测 + 跟踪示例（详细注释）
- `yolo_pose_example.py`: YOLO Pose 姿态估计示例
- `batch_processing.py`: 批量图像处理示例

### tests/
测试文件目录，包含各种测试脚本。

- `quick_test.py`: 快速功能测试
- `test_structure.py`: 项目结构测试
- `test_new_features.py`: 新功能测试
- `test_rfdetr.py`: RF-DETR 检测器测试

### docs/
文档目录，包含项目文档。

- `QUICKSTART.md`: 快速开始指南
- `FEATURES.md`: 高级功能详细文档
- `CHANGELOG.md`: 版本更新日志
- `PROJECT_STRUCTURE.md`: 项目结构说明（本文件）
- `OPTIMIZATION_SUGGESTIONS.md`: 项目优化建议和完成情况

## 文件命名规范

- Python模块文件使用小写字母和下划线：`detector.py`, `roi_detector.py`
- 类名使用大驼峰命名：`Detector`, `ROIDetector`, `PerformanceMonitor`
- 函数和变量使用小写字母和下划线：`get_fps()`, `frame_count`
- 常量使用大写字母和下划线：`YOLO_AVAILABLE`, `MAX_AGE`

## 导入规范

所有公共API都通过 `visionframework/__init__.py` 导出，用户可以直接从主包导入：

```python
from visionframework import Detector, Tracker, VisionPipeline
from visionframework import ROIDetector, Counter
from visionframework import ResultExporter, PerformanceMonitor
```

