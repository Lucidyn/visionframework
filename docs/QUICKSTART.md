# 快速开始指南

## 安装

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

# 手动绘制边界框（VisionPipeline 没有内置的 visualize 方法）
for detection in results["detections"]:
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, detection["class_name"], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

# 使用静态方法快速处理，带配置字典
results = VisionPipeline.process_image(image, {
    "detector_config": {"model_path": "yolov8n.pt"}
})

# 绘制边界框
for detection in results["detections"]:
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, detection["class_name"], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

# 或者使用静态方法，带配置字典
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

## 验证

要验证安装是否成功，运行其中一个示例脚本：

```bash
python examples/00_basic_detection.py
```

你应该会看到一个窗口显示示例图像上的目标检测结果。

## 下一步

- 在 `examples/` 目录中探索更多示例
- 查看 `FEATURES.md` 了解所有可用功能
- 参考 `API_REFERENCE.md` 获取详细的 API 文档
