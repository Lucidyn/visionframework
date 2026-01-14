# Vision Framework v0.2.8 - 快速参考指南

## 📚 文档导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目概览 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速开始指南 |
| [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) | 迁移指南 |
| [docs/ARCHITECTURE_V0.2.8.md](docs/ARCHITECTURE_V0.2.8.md) | 架构设计 |
| [docs/FEATURES.md](docs/FEATURES.md) | 功能特性 |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | 版本历史 |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 项目结构 |

## 🚀 快速开始

### 1. 基本导入
```python
# 主要类
from visionframework import (
    CLIPExtractor,
    PoseEstimator,
    ReIDExtractor,
    YOLODetector,
    IOUTracker,
    VisionPipeline
)

# 异常处理
from visionframework import (
    VisionFrameworkError,
    DetectorInitializationError,
    TrackerInitializationError
)

# 模型管理
from visionframework import (
    ModelManager,
    get_model_manager
)
```

### 2. 异常处理示例
```python
from visionframework import DetectorInitializationError, VisionFrameworkError

try:
    detector = YOLODetector(config)
    detections = detector.detect(image)
except DetectorInitializationError as e:
    print(f"检测器初始化失败: {e}")
except VisionFrameworkError as e:
    print(f"Vision Framework 错误: {e}")
```

### 3. 特征提取示例
```python
# CLIP 零样本分类
clip = CLIPExtractor()
clip.initialize()
scores = clip.zero_shot_classify(image, ["cat", "dog", "bird"])

# 姿态估计
pose_est = PoseEstimator()
pose_est.initialize()
poses = pose_est.process(image)

# ReID 特征提取
reid = ReIDExtractor()
reid.initialize()
features = reid.extract(image, bboxes)
```

### 4. 模型管理示例
```python
from visionframework import get_model_manager

manager = get_model_manager()

# 获取模型路径（自动下载）
model_path = manager.get_model_path("yolov8n.pt", download=True)

# 列表缓存模型
cached = manager.list_cached_models()

# 清除缓存
manager.clear_cache("yolov8n.pt")
```

### 5. 自定义特征提取器
```python
from visionframework.core.processors import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    def initialize(self) -> None:
        # 加载你的模型
        self.model = ...
        self._initialized = True
    
    def extract(self, input_data):
        # 实现特征提取
        with torch.no_grad():
            features = self.model(input_data)
        return features
    
    def _move_to_device(self, device: str) -> None:
        # 移动模型到设备
        self.model.to(device)

# 使用
extractor = MyFeatureExtractor("my_model")
extractor.initialize()
features = extractor.extract(image)
```

## 📖 异常类参考

| 异常类 | 用途 | 继承自 |
|--------|------|--------|
| `VisionFrameworkError` | 基础异常 | `Exception` |
| `DetectorInitializationError` | 检测器初始化失败 | `VisionFrameworkError` |
| `DetectorInferenceError` | 检测器推理失败 | `VisionFrameworkError` |
| `TrackerInitializationError` | 跟踪器初始化失败 | `VisionFrameworkError` |
| `TrackerUpdateError` | 跟踪器更新失败 | `VisionFrameworkError` |
| `ConfigurationError` | 配置错误 | `VisionFrameworkError` |
| `ModelNotFoundError` | 模型未找到 | `VisionFrameworkError` |
| `ModelLoadError` | 模型加载失败 | `VisionFrameworkError` |
| `DeviceError` | 设备错误 | `VisionFrameworkError` |
| `DependencyError` | 依赖缺失 | `VisionFrameworkError` |
| `DataFormatError` | 数据格式错误 | `VisionFrameworkError` |
| `ProcessingError` | 处理错误 | `VisionFrameworkError` |

## 🔧 模块结构

```
visionframework/
├── core/
│   ├── processors/          # 特征提取器 (新)
│   ├── detectors/           # 检测器
│   └── trackers/            # 跟踪器
├── models/                  # 模型管理 (新)
├── utils/
│   ├── evaluation/          # 评估工具
│   ├── visualization/       # 可视化
│   └── ...
├── data/                    # 数据结构
├── exceptions.py            # 异常类 (新)
└── __init__.py              # 主包 (更新)
```

## 📋 特征提取器接口

所有特征提取器实现以下接口：

```python
class FeatureExtractor(ABC):
    def initialize(self) -> None:
        """初始化提取器并加载模型"""
        pass
    
    def extract(self, input_data) -> Union[np.ndarray, dict]:
        """从输入数据提取特征"""
        pass
    
    def is_initialized(self) -> bool:
        """检查提取器是否已初始化"""
        pass
    
    def to(self, device: str) -> None:
        """将提取器移动到指定设备"""
        pass
    
    def _move_to_device(self, device: str) -> None:
        """内部实现：设备移动逻辑"""
        pass
```

## 🔄 迁移检查表

如果你使用的是 v0.2.7 或更早版本：

- [ ] 更新 CLIP 导入：`visionframework.core.clip` → `visionframework`
- [ ] 更新 PoseEstimator 导入：`visionframework.core.pose_estimator` → `visionframework`
- [ ] 更新 ReIDExtractor 导入：`visionframework.core.reid` → `visionframework`
- [ ] 添加异常处理代码（可选但推荐）
- [ ] 考虑使用 ModelManager（可选）
- [ ] 测试所有更新后的代码

详见 [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)

## 🧪 测试

运行所有测试：
```bash
python -m pytest tests/ -v
```

运行特定测试：
```bash
# CLIP 测试
python -m pytest tests/test_clip_integration.py -v

# 追踪评估器测试
python -m pytest tests/test_tracking_evaluator.py -v
```

## 📊 版本信息

```python
import visionframework
print(visionframework.__version__)  # "0.2.8"
```

## 🔗 相关链接

- [快速开始](docs/QUICKSTART.md)
- [迁移指南](docs/MIGRATION_GUIDE.md)
- [项目结构](docs/PROJECT_STRUCTURE.md)
- [变更日志](docs/CHANGELOG.md)

## ❓ 常见问题

### Q: 旧的导入路径还能用吗？
A: 是的，旧的导入路径仍然有效但已弃用。建议更新到新的导入路径。

### Q: 如何创建自定义特征提取器？
A: 继承 `FeatureExtractor` 并实现必要的抽象方法。参见上面的示例。

### Q: ModelManager 是必需的吗？
A: 不是，但建议使用它来管理模型缓存。

### Q: 如何处理异常？
A: 使用特定的异常类或捕获 `VisionFrameworkError` 基类。

### Q: 支持哪些 Python 版本？
A: Python 3.8+

## 📞 支持

- 查看文档：`docs/` 目录
- 查看示例：`examples/` 目录
- 运行测试：`tests/` 目录

---

**最后更新**: 2024年  
**Vision Framework 版本**: v0.2.8  
**状态**: ✅ 生产就绪
