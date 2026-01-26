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

## 验证

要验证安装是否成功，运行其中一个示例脚本：

```bash
# 激活 Conda 环境（如果使用）
conda activate frametest

# 运行基础检测示例
python examples/00_basic_detection.py

# 运行姿态估计示例
python examples/10_pose_estimation.py

# 运行 SAM 分割示例
python examples/08_segmentation_sam.py

# 运行 CLIP 特征示例
python examples/09_clip_features.py
```

## 下一步

- 在 `examples/` 目录中探索更多示例
- 查看 `FEATURES.md` 了解所有可用功能
- 参考 `API_REFERENCE.md` 获取详细的 API 文档
- 运行测试以确保所有功能正常工作：
  ```bash
  python -m pytest test/
  ```
