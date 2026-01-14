# Vision Framework v0.2.8 - 完整的架构重构

## 项目概述

Vision Framework 是一个全面的计算机视觉框架，支持目标检测、跟踪、姿态估计和其他 CV 任务。

当前版本：**v0.2.8**

## 实施的改进

### 1️⃣ 统一异常系统

**文件**: `visionframework/exceptions.py`

创建了完整的异常层级系统，包含以下类：

```
VisionFrameworkError (基础异常)
├── DetectorInitializationError
├── DetectorInferenceError
├── TrackerInitializationError
├── TrackerUpdateError
├── ConfigurationError
├── ModelNotFoundError
├── ModelLoadError
├── DeviceError
├── DependencyError
├── DataFormatError
└── ProcessingError
```

**优势**：
- ✅ 统一的异常处理
- ✅ 更好的错误诊断
- ✅ 清晰的异常层级
- ✅ 易于扩展新的异常类型

### 2️⃣ 模型管理器

**文件**: `visionframework/models/__init__.py`

创建了 `ModelManager` 类用于：
- 模型缓存管理
- 模型下载和加载
- 版本管理
- 模型注册表

**关键特性**：
- 全局 `get_model_manager()` 函数
- 自定义缓存目录支持
- 模型清理和列表功能

### 3️⃣ 特征提取器重构

**目录**: `visionframework/core/processors/`

#### 基类: `FeatureExtractor`
提供统一的特征提取器接口：

```python
class FeatureExtractor(ABC):
    def initialize(self) -> None: ...
    def extract(self, input_data) -> Union[np.ndarray, dict]: ...
    def is_initialized(self) -> bool: ...
    def to(self, device: str) -> None: ...
    def _move_to_device(self, device: str) -> None: ...
```

#### 迁移的模块

**CLIPExtractor** → `core/processors/clip_extractor.py`
- 图像-文本匹配
- 零样本分类
- FP16 支持

**ReIDExtractor** → `core/processors/reid_extractor.py`
- 人员重识别特征提取
- ResNet50 主干网络
- L2 归一化

**PoseEstimator** → `core/processors/pose_estimator.py`
- YOLO Pose 模型集成
- COCO 17 关键点检测
- 置信度阈值过滤

### 4️⃣ 导入更新

所有相关文件的导入已更新：

```
✅ visionframework/__init__.py - 导出所有新类
✅ visionframework/core/__init__.py - 从 processors 导入
✅ visionframework/core/trackers/reid_tracker.py - 导入更新
✅ examples/clip_example.py - 更新示例代码
```

### 5️⃣ 版本号更新

- README.md: v0.2.5 → v0.2.8
- setup.py: v0.2.5 → v0.2.8
- visionframework/__init__.py: v0.2.5 → v0.2.8

### 6️⃣ 文档更新

#### 新增文档
- **MIGRATION_GUIDE.md**: 导入迁移指南
- **ARCHITECTURE_V0.2.8.md**: 架构改进详细说明
- **PROJECT_STRUCTURE.md**: 更新的项目结构

#### 更新文档
- **CHANGELOG.md**: 添加 v0.2.8 变更记录

## 使用示例

### 特征提取

```python
from visionframework import CLIPExtractor, PoseEstimator, ReIDExtractor

# CLIP 零样本分类
clip = CLIPExtractor()
clip.initialize()
scores = clip.zero_shot_classify(image, ["cat", "dog"])

# 姿态估计
pose_est = PoseEstimator()
pose_est.initialize()
poses = pose_est.process(image)

# ReID 特征提取
reid = ReIDExtractor()
reid.initialize()
features = reid.extract(image, bboxes)
```

### 异常处理

```python
from visionframework import DetectorInitializationError, VisionFrameworkError

try:
    detector = YOLODetector(config)
except DetectorInitializationError as e:
    print(f"检测器初始化失败: {e}")
except VisionFrameworkError as e:
    print(f"框架错误: {e}")
```

### 模型管理

```python
from visionframework import get_model_manager

manager = get_model_manager()
model_path = manager.get_model_path("yolov8n.pt", download=True)
cached_models = manager.list_cached_models()
manager.clear_cache("yolov8n.pt")
```

### 自定义特征提取器

```python
from visionframework.core.processors import FeatureExtractor

class MyExtractor(FeatureExtractor):
    def initialize(self) -> None:
        # 加载模型
        pass
    
    def extract(self, image):
        # 实现特征提取
        pass
    
    def _move_to_device(self, device: str) -> None:
        # 移动到设备
        pass

extractor = MyExtractor("my_model")
extractor.initialize()
features = extractor.extract(image)
```

## 测试状态

✅ **所有测试通过**

```
tests/test_clip_integration.py::test_clip_wrapper_smoke PASSED
tests/test_tracking_evaluator.py::test_tracking_evaluator_basic PASSED
tests/test_tracking_evaluator.py::test_tracking_evaluator_mota PASSED
tests/test_tracking_evaluator.py::test_tracking_evaluator_idf1 PASSED
tests/test_tracking_evaluator.py::test_tracking_evaluator_motp PASSED
tests/test_tracking_evaluator.py::test_tracking_evaluator_comprehensive PASSED
```

## 向后兼容性

⚠️ **导入路径变更** - 旧路径仍然可用但已弃用：

```python
# 已弃用（仍然有效但不推荐）
from visionframework.core.clip import CLIPExtractor
from visionframework.core.pose_estimator import PoseEstimator
from visionframework.core.reid import ReIDExtractor

# 推荐方式
from visionframework import CLIPExtractor, PoseEstimator, ReIDExtractor
```

## Git 提交

```
commit 6969268
feat: architecture restructuring with unified exceptions, model manager, and feature processors
```

包含以下更改：
- 17 个文件修改/新增
- 1274 行代码更改

## 后续计划

### 可能的未来改进
- [ ] 更多特殊异常类型
- [ ] 高级模型缓存策略
- [ ] 模型性能基准测试
- [ ] 自定义处理器注册表
- [ ] 处理器管道
- [ ] 分布式处理支持
- [ ] 模型量化和优化
- [ ] 动态模型加载

## 开发者指南

### 添加新的异常类型

```python
# 在 visionframework/exceptions.py 中
class MyCustomError(VisionFrameworkError):
    """自定义错误描述"""
    pass
```

### 添加新的特征提取器

```python
# 在 visionframework/core/processors/ 中创建新文件
from .feature_extractor import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    def initialize(self) -> None:
        # 实现初始化
        pass
    
    def extract(self, input_data):
        # 实现特征提取
        pass
    
    def _move_to_device(self, device: str) -> None:
        # 实现设备移动
        pass
```

### 注册模型

```python
from visionframework import get_model_manager

manager = get_model_manager()
manager.register_model(
    name="my_model",
    source="custom",
    config={"key": "value"}
)
```

## 项目统计

- **代码行数**: ~3500+ 行
- **模块数**: 25+
- **测试数**: 10+
- **示例代码**: 10+
- **文档页面**: 8+

## 许可证

MIT License

## 致谢

感谢所有贡献者和用户的支持！

---

**最后更新**: 2024年 (v0.2.8)
