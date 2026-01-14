"""
架构重构总结 - v0.2.8

本文档总结了 v0.2.8 中的主要架构改进和好处。
"""

# ============================================================================
# 主要改进
# ============================================================================

## 1. 统一异常系统

### 改进前（v0.2.7）
- 异常定义分散在各个模块
- 没有统一的异常层级
- 难以编写通用的异常处理代码

### 改进后（v0.2.8）
```
visionframework/exceptions.py
├── VisionFrameworkError (基础)
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

### 优势
- ✅ 清晰的异常层级
- ✅ 更好的错误诊断
- ✅ 简化异常处理代码
- ✅ 统一的错误消息格式

## 2. 模型管理

### 改进前（v0.2.7）
- 没有集中的模型管理
- 模型缓存位置不一致
- 没有模型版本管理
- 下载逻辑分散在各个模块

### 改进后（v0.2.8）
```
visionframework/models/
└── __init__.py (ModelManager 类)
```

**功能**
- 集中的模型缓存管理
- 模型注册表
- 版本管理支持
- 统一的下载接口

**使用示例**
```python
from visionframework import get_model_manager

manager = get_model_manager()
model_path = manager.get_model_path("yolov8n.pt", download=True)
manager.clear_cache("yolov8n.pt")  # 清除特定模型
cached_models = manager.list_cached_models()  # 列出所有缓存
```

## 3. 特征提取器重构

### 改进前（v0.2.7）
- 特征提取器位置不一致
  - `core/clip.py` - CLIP
  - `core/reid.py` - ReID
  - `core/pose_estimator.py` - 姿态估计
- 没有统一的特征提取接口
- 难以扩展新的特征提取器

### 改进后（v0.2.8）
```
visionframework/core/processors/
├── __init__.py
├── feature_extractor.py (基类)
├── clip_extractor.py
├── reid_extractor.py
└── pose_estimator.py
```

**统一的 FeatureExtractor 基类**
```python
class FeatureExtractor(ABC):
    def initialize(self) -> None: ...
    def extract(self, input_data) -> Union[np.ndarray, dict]: ...
    def is_initialized(self) -> bool: ...
    def to(self, device: str) -> None: ...
    def _move_to_device(self, device: str) -> None: ...
```

### 优势
- ✅ 一致的模块位置
- ✅ 统一的接口
- ✅ 更容易扩展
- ✅ 改进的类型提示

## 4. 版本和导出更新

### 改进前（v0.2.7）
- 主包导出不包括异常和模型管理器
- 版本号：0.2.5（不同步）

### 改进后（v0.2.8）
- 所有主要类都在主包中导出
- 版本号统一为 0.2.8
- 完整的 `__all__` 列表

**新导出**
```python
from visionframework import (
    # ... 现有的导出
    # 新增
    VisionFrameworkError,
    DetectorInitializationError,
    TrackerInitializationError,
    # ... 其他异常
    ModelManager,
    get_model_manager,
    ReIDExtractor,
)
```

# ============================================================================
# 迁移影响
# ============================================================================

## 破坏性变化
- ❌ 特征提取器导入路径已更改
  - OLD: `from visionframework.core.clip import CLIPExtractor`
  - NEW: `from visionframework import CLIPExtractor`
  - 但旧路径仍然可用（deprecated）

## 非破坏性变化
- ✅ 所有新的异常和模型管理功能
- ✅ 特征提取器中增加的方法
- ✅ 改进的类型提示和文档

# ============================================================================
# 建议的最佳实践
# ============================================================================

## 1. 使用主包导入
```python
from visionframework import CLIPExtractor, PoseEstimator, ReIDExtractor
```

## 2. 处理异常
```python
from visionframework import DetectorInitializationError

try:
    detector = YOLODetector(config)
except DetectorInitializationError as e:
    logger.error(f"Failed to init detector: {e}")
```

## 3. 使用模型管理器
```python
from visionframework import get_model_manager

manager = get_model_manager()
model_path = manager.get_model_path("yolov8n.pt")
```

## 4. 扩展特征提取器
```python
from visionframework.core.processors import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    def initialize(self) -> None:
        # 初始化模型
        pass
    
    def extract(self, input_data):
        # 实现特征提取
        pass
```

# ============================================================================
# 性能和维护改进
# ============================================================================

### 开发者体验
- 更清晰的代码组织
- 更容易的错误诊断
- 更一致的 API
- 更好的文档

### 维护性
- 集中的异常处理
- 集中的模型管理
- 模块化的特征提取器
- 更容易的代码重构

### 扩展性
- 易于添加新的异常类型
- 易于集成新的特征提取器
- 易于支持新的模型源
- 易于实现自定义处理器

# ============================================================================
# 后续计划
# ============================================================================

未来版本可能包括：
- [ ] 更多异常类型和错误信息
- [ ] 高级模型缓存策略
- [ ] 模型性能基准测试
- [ ] 自定义处理器注册表
- [ ] 处理器管道
- [ ] 分布式处理支持
