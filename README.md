# Vision Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

轻量、模块化的计算机视觉框架，支持目标检测、跟踪、实例分割、姿态估计与结果导出。该仓库提供统一的高层 API，便于在工程中快速集成多种视觉能力。

主要目标：易用、模块化、可扩展。核心接口示例与快速上手指南见下文与 `docs/`。

最短快速开始：

```bash
pip install -e .
```

运行示例（仅检测）：

```python
from visionframework import Detector
import cv2

det = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
det.initialize()
img = cv2.imread("your_image.jpg")
print(len(det.detect(img)))
```

## 文档

仓库的完整文档位于 `docs/`（已重建）。主要入口：

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICKSTART.md) | 安装与最小示例 |
| [快速参考](docs/QUICK_REFERENCE.md) | 常用 API 与配置速查 |
| [功能特性](docs/FEATURES.md) | 功能一览与场景说明 |
| [项目结构](docs/PROJECT_STRUCTURE.md) | 代码组织与模块说明 |
| [架构概览](docs/ARCHITECTURE_V0.2.8.md) | 高层架构与组件交互 |
| [迁移指南](docs/MIGRATION_GUIDE.md) | 从旧版本迁移要点 |
| [变更日志](docs/CHANGELOG.md) | 版本历史 |

示例脚本在 `examples/` 下，推荐先查看 `examples/README.md` 获取运行命令。

## 关键更新

- 新增 `categories` 参数：可在 `Detector` 配置或调用 `detect(image, categories=[...])` 时使用，用于在框架层面过滤返回结果（按类别名或 id）。

## 贡献与支持

欢迎通过 Issue/PR 贡献。有关开发依赖、测试和本地运行，请参阅 `pyproject.toml` 与 `requirements.txt`。

---

（下面为详细 API/示例与配置仍保留于仓库文档；如需更紧凑的 README 可告知我将进一步裁剪）

```python
from visionframework import Config

# 获取各模块默认配置
detector_config = Config.get_default_detector_config()
tracker_config = Config.get_default_tracker_config()
pipeline_config = Config.get_default_pipeline_config()
```

## 示例代码

查看 `examples/` 目录获取更多使用示例：

### 基础示例
- `basic_usage.py`: 基本使用示例（详细注释）
- `video_tracking.py`: 视频跟踪示例（详细注释）
- **`config_example.py`**: 使用配置文件示例（推荐，支持 YAML/JSON）

### 检测器示例
- `rfdetr_example.py`: RF-DETR 检测器示例
- `rfdetr_tracking.py`: RF-DETR 检测 + 跟踪示例
- `yolo_pose_example.py`: YOLO Pose 姿态估计示例

### 高级功能示例
- `advanced_features.py`: 高级功能示例（ROI、计数、性能监控等，详细注释）
- `batch_processing.py`: 批量图像处理示例

## 示例代码

查看 `examples/` 目录获取完整示例代码：

| 示例 | 说明 |
|------|------|
| `basic_usage.py` | 基本使用示例 |
| `config_example.py` | 配置文件用法（推荐） |
| `video_tracking.py` | 视频跟踪示例 |
| `advanced_features.py` | 高级功能（ROI、计数等） |
| `batch_processing.py` | 批量处理示例 |
| `yolo_pose_example.py` | 姿态估计示例 |
| `rfdetr_example.py` | RF-DETR 检测器示例 |
| `rfdetr_tracking.py` | RF-DETR 跟踪示例 |
| `clip_example.py` | CLIP 零样本分类示例 |
| `tracking_evaluation_example.py` | 跟踪评估示例 |

## 依赖项

### 必需依赖
- opencv-python >= 4.8.0
- numpy >= 1.24.0, < 2.0.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics >= 8.0.0
- scipy >= 1.10.0
- Pillow >= 10.0.0

### 可选依赖
- transformers >= 4.30.0 (用于 DETR/CLIP 模型)
- rfdetr >= 0.1.0 (用于 RF-DETR 模型)
- supervision >= 0.18.0 (用于 RF-DETR 模型)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 常见问题

**Q: 如何选择检测器？**  
A: YOLO 最快，DETR 精度最高，RF-DETR 平衡两者。根据需求选择。

**Q: 支持 GPU 加速吗？**  
A: 是的，所有模块都支持 CUDA。设置 `device: "cuda"` 即可。

**Q: 如何使用自定义模型？**  
A: 通过 `model_path` 参数指定模型文件路径即可。

**Q: 能扩展新功能吗？**  
A: 可以，所有模块都是可扩展的，支持继承和定制。

## 支持

- 阅读 [文档](docs/)
- 查看 [示例代码](examples/)
- 运行 [测试](tests/)
- 提出 [问题/建议](https://github.com/visionframework/visionframework/issues)

---

**Vision Framework v0.2.8** | 架构优化版本 | 生产就绪
- pyyaml >= 6.0 (用于 YAML 配置文件支持)

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。
