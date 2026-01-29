# 快速开始指南

## 安装

### 方法 1: 使用 pip 安装

1. **克隆仓库**:
   ```bash
   git clone https://github.com/yourusername/visionframework.git
   cd visionframework
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **安装包**:
   ```bash
   pip install -e .
   ```

### 方法 2: 使用 Conda 环境（推荐）

1. **创建并激活 Conda 环境**:
   ```bash
   conda create -n frametest python=3.8
   conda activate frametest
   ```

2. **克隆仓库**:
   ```bash
   git clone https://github.com/yourusername/visionframework.git
   cd visionframework
   ```

3. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

4. **安装包**:
   ```bash
   pip install -e .
   ```

5. **安装可选依赖** (根据需要):
   ```bash
   # 安装所有可选依赖
   pip install -e "[all]"
   
   # 或者安装特定功能组
   pip install -e "[clip]"       # 用于 CLIP 支持
   pip install -e "[sam]"        # 用于 SAM 分割支持
   pip install -e "[rfdetr]"     # 用于 RF-DETR 支持
   pip install -e "[pyav]"       # 用于 PyAV 高性能视频处理支持
   ```

## 基本使用

### 示例 1: 基本目标检测

```python
from visionframework.core.pipeline import VisionPipeline
import cv2

# 使用配置字典初始化管道
pipeline = VisionPipeline({
    "detector_config": {"model_path": "yolov8n.pt"}
})

# 加载图像
image = cv2.imread("test.jpg")

# 处理图像
results = pipeline.process(image)

# 手动绘制边界框
for detection in results["detections"]:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, detection.class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow("检测结果", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例 2: 简化 API

```python
from visionframework.core.pipeline import VisionPipeline
import cv2
import numpy as np

# 加载图像
image = cv2.imread("test.jpg")

# 使用静态方法快速处理
results = VisionPipeline.process_image(image, model_path="yolov8n.pt")

# 绘制边界框
for detection in results["detections"]:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, detection.class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow("结果", image)
cv2.waitKey(0)
```

### 示例 3: 视频处理

```python
from visionframework.core.pipeline import VisionPipeline

# 使用配置字典处理视频文件
pipeline = VisionPipeline({
    "detector_config": {"model_path": "yolov8n.pt"}
})
pipeline.process_video("input.mp4", "output.mp4")

# 或者使用静态方法
VisionPipeline.run_video("input.mp4", "output.mp4", model_path="yolov8n.pt")
```

### 示例 4: 视频流处理

```python
from visionframework.core.pipeline import VisionPipeline

# 使用配置字典处理 RTSP 流
pipeline = VisionPipeline({
    "detector_config": {"model_path": "yolov8n.pt"}
})
pipeline.process_video("rtsp://example.com/stream", "output.mp4")

# 或者使用静态方法
VisionPipeline.run_video("rtsp://example.com/stream", "output.mp4", model_path="yolov8n.pt")
```

### 示例 5: 姿态估计

```python
from visionframework.core.pipeline import VisionPipeline
import cv2

# 初始化带姿态估计的管道
pipeline = VisionPipeline({
    "enable_pose_estimation": True,
    "detector_config": {"model_path": "yolov8n.pt"},
    "pose_estimator_config": {"model_path": "yolov8n-pose.pt"}
})

# 加载图像
image = cv2.imread("person.jpg")

# 处理图像
results = pipeline.process(image)

# 绘制姿态关键点和骨骼
from visionframework.utils.visualization import Visualizer
viz = Visualizer()
result_image = viz.draw_poses(image.copy(), results["poses"])

# 显示结果
cv2.imshow("姿态估计结果", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例 6: 批处理（提高性能）

```python
from visionframework.core.pipeline import VisionPipeline
import cv2
import numpy as np

# 初始化支持批处理的管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "yolov8n.pt",
        "batch_inference": True
    }
})

# 加载多张图像
images = [
    cv2.imread("image1.jpg"),
    cv2.imread("image2.jpg"),
    cv2.imread("image3.jpg"),
    cv2.imread("image4.jpg")
]

# 批处理图像
results = pipeline.process_batch(images)

# 处理结果
for i, result in enumerate(results):
    print(f"Image {i+1}: {len(result['detections'])} detections")
```

### 示例 7: SAM 分割

```python
from visionframework.core.segmenters.sam_segmenter import SAMSegmenter
import cv2

# 初始化 SAM 分割器
segmenter = SAMSegmenter({"model_path": "sam_vit_h_4b8939.pth"})
segmenter.initialize()

# 加载图像
image = cv2.imread("object.jpg")

# 自动分割
results = segmenter.segment(image)

# 绘制分割结果
from visionframework.utils.visualization import Visualizer
viz = Visualizer()
result_image = viz.draw_segmentations(image.copy(), results)

# 显示结果
cv2.imshow("分割结果", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例 8: CLIP 特征

```python
from visionframework.core.processors.clip_extractor import CLIPExtractor
import cv2

# 初始化 CLIP 提取器
clip_extractor = CLIPExtractor({"model_path": "ViT-B/32"})
clip_extractor.initialize()

# 加载图像
image = cv2.imread("cat.jpg")

# 提取图像特征
image_features = clip_extractor.extract_image_features(image)
print(f"图像特征维度: {image_features.shape}")

# 计算图像-文本相似度
texts = ["a cat", "a dog", "a car"]
similarities = clip_extractor.compute_similarity(image, texts)
print("文本相似度:")
for text, sim in zip(texts, similarities):
    print(f"{text}: {sim:.4f}")
```

### 示例 9: PyAV 视频处理

```python
from visionframework.core.pipeline import VisionPipeline

# 使用配置字典处理视频文件（使用PyAV）
pipeline = VisionPipeline({
    "detector_config": {"model_path": "yolov8n.pt"}
})
pipeline.process_video("input.mp4", "output.mp4", use_pyav=True)

# 处理RTSP流（使用PyAV）
pipeline.process_video("rtsp://example.com/stream", "output_rtsp.mp4", use_pyav=True)

# 或者使用静态方法（使用PyAV）
VisionPipeline.run_video("input.mp4", "output.mp4", model_path="yolov8n.pt", use_pyav=True)

# 使用静态方法处理RTSP流（使用PyAV）
VisionPipeline.run_video("rtsp://example.com/stream", "output_rtsp.mp4", model_path="yolov8n.pt", use_pyav=True)
```

### 示例 10: 内存池管理

```python
from visionframework.utils.memory.memory_manager import MemoryManager
import numpy as np
import cv2

# 初始化全局内存池
memory_pool = MemoryManager.get_global_memory_pool()
memory_pool.initialize(
    min_blocks=4,      # 最小内存块数量
    block_size=(480, 640, 3),  # 每个内存块的形状
    max_blocks=8       # 最大内存块数量
)

# 分配内存
memory = memory_pool.acquire()
print(f"分配的内存形状: {memory.shape}")
print(f"内存池状态: {memory_pool.get_status()}")

# 使用内存
image = cv2.imread("test.jpg")
if image is not None:
    # 将图像复制到分配的内存中
    memory[:image.shape[0], :image.shape[1]] = image[:memory.shape[0], :memory.shape[1]]
    
    # 处理图像
    # ...

# 释放内存
memory_pool.release(memory)
print(f"释放内存后的内存池状态: {memory_pool.get_status()}")

# 优化内存使用
memory_pool.optimize()
print(f"优化后的内存池状态: {memory_pool.get_status()}")

# 获取内存池统计信息
stats = memory_pool.get_stats()
print(f"内存池统计信息: {stats}")
```

### 示例 11: 插件系统

```python
from visionframework.core.plugin_system import (
    register_detector, register_tracker, plugin_registry,
    register_segmenter, register_model
)
from visionframework.core.pipeline import VisionPipeline
import cv2
import numpy as np

# 注册自定义检测器
@register_detector("simple_detector")
class SimpleDetector:
    def __init__(self, config):
        self.config = config
        self.conf_threshold = config.get("conf_threshold", 0.5)
    
    def initialize(self):
        print("初始化简单检测器")
        return True
    
    def detect(self, image):
        # 简单的检测实现
        from visionframework.data.detection import Detection
        detections = []
        # 模拟检测结果
        height, width = image.shape[:2]
        detections.append(Detection(
            bbox=(width//4, height//4, 3*width//4, 3*height//4),
            confidence=0.9,
            class_id=0,
            class_name="object"
        ))
        return detections

# 注册自定义跟踪器
@register_tracker("simple_tracker")
class SimpleTracker:
    def __init__(self, config):
        self.config = config
        self.id_counter = 0
    
    def initialize(self):
        print("初始化简单跟踪器")
        return True
    
    def update(self, detections):
        # 简单的跟踪实现
        for i, detection in enumerate(detections):
            detection.track_id = i
        return detections

# 列出所有注册的组件
print("注册的检测器:", plugin_registry.list_detectors())
print("注册的跟踪器:", plugin_registry.list_trackers())

# 使用自定义组件创建管道
pipeline = VisionPipeline({
    "detector_config": {
        "model_path": "simple_detector"  # 使用自定义检测器
    },
    "tracker_config": {
        "tracker_type": "simple_tracker"  # 使用自定义跟踪器
    },
    "enable_tracking": True
})

# 初始化管道
pipeline.initialize()

# 测试自定义组件
image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1)

results = pipeline.process(image)
print(f"检测结果数量: {len(results['detections'])}")
for detection in results['detections']:
    print(f"检测目标: {detection.class_name}, 置信度: {detection.confidence}, 跟踪ID: {detection.track_id}")
```

### 示例 12: 统一错误处理

```python
from visionframework.utils.error_handling import ErrorHandler

# 创建错误处理器
handler = ErrorHandler()

# 1. 处理错误
def risky_operation():
    raise ValueError("测试错误")

try:
    # 模拟错误
    error = ValueError("测试错误")
    result = handler.handle_error(
        error=error,
        error_type=Exception,
        message="测试错误处理"
    )
    print(f"错误处理结果: {result}")
except Exception as e:
    print(f"错误处理失败: {e}")

# 2. 包装错误
print("\n测试错误包装:")
wrapped_func = handler.wrap_error(
    func=risky_operation,
    error_type=Exception,
    message="测试错误包装"
)
result = wrapped_func()
print(f"包装函数结果: {result}")

# 3. 输入验证
print("\n测试输入验证:")
# 有效输入
valid_input = {"key": "value"}
is_valid, error_msg = handler.validate_input(
    input_value=valid_input,
    expected_type=dict,
    param_name="input"
)
print(f"有效输入验证结果: {is_valid}, 错误消息: {error_msg}")

# 无效输入
invalid_input = "not a dict"
is_valid, error_msg = handler.validate_input(
    input_value=invalid_input,
    expected_type=dict,
    param_name="input"
)
print(f"无效输入验证结果: {is_valid}, 错误消息: {error_msg}")

# 4. 错误消息格式化
print("\n测试错误消息格式化:")
error = ValueError("测试错误")
error_message = handler.format_error_message(
    message="测试操作",
    error=error,
    context={"param": "value"}
)
print(f"格式化的错误消息: {error_message}")
```

### 示例 13: 依赖管理

```python
from visionframework.utils.dependency_manager import (
    DependencyManager, is_dependency_available, 
    import_optional_dependency, get_available_dependencies,
    get_missing_dependencies
)

# 创建依赖管理器
manager = DependencyManager()

# 1. 检查依赖可用性
print("检查依赖可用性:")
dependencies = ["clip", "sam", "rfdetr", "pyav"]
for dep in dependencies:
    available = is_dependency_available(dep)
    print(f"{dep}: {available}")

# 2. 获取依赖信息
print("\n获取依赖信息:")
clip_info = manager.get_dependency_info("clip")
print(f"CLIP 依赖信息: {clip_info}")

# 3. 获取安装命令
print("\n获取安装命令:")
install_command = manager.get_install_command("clip")
print(f"CLIP 安装命令: {install_command}")

# 4. 导入可选依赖
print("\n导入可选依赖:")
module = import_optional_dependency("clip", "transformers")
print(f"导入 transformers 模块: {module is not None}")

# 5. 获取可用和缺失的依赖
print("\n依赖状态:")
available = get_available_dependencies()
print(f"可用的依赖: {available}")

missing = get_missing_dependencies()
print(f"缺失的依赖: {missing}")

# 6. 获取所有依赖状态
print("\n所有依赖状态:")
all_status = manager.get_all_dependency_status()
for dep, status in all_status.items():
    print(f"  {dep}: {status['available']} - {status['message']}")


## 验证

要验证安装是否成功，运行其中一个示例脚本：

```bash
# 激活 Conda 环境（如果使用）
conda activate frametest

# 运行基础检测示例
python examples/basic/00_basic_detection.py

# 运行姿态估计示例
python examples/models/10_pose_estimation.py

# 运行 SAM 分割示例
python examples/models/08_segmentation_sam.py

# 运行 CLIP 特征示例
python examples/models/09_clip_features.py

# 运行 PyAV 视频处理示例
python examples/video/12_pyav_video_processing.py

# 运行 VisionPipeline PyAV 集成示例
python examples/video/13_vision_pipeline_pyav.py

# 运行插件系统示例
python examples/system/14_plugin_system_example.py

# 运行内存池管理示例
python examples/system/15_memory_pool_example.py

# 运行统一错误处理示例
python examples/system/16_error_handling_example.py

# 运行依赖管理示例
python examples/system/17_dependency_management_example.py
```

## 下一步

- 在 `examples/` 目录中探索更多示例
- 查看 `FEATURES.md` 了解所有可用功能
- 参考 `API_REFERENCE.md` 获取详细的 API 文档
- 运行测试以确保所有功能正常工作：
  ```bash
  python -m pytest test/
  ```
