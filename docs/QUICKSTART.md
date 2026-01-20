# 快速开始

本页提供最短路径让你在本地运行一次检测示例。

## 安装

推荐使用虚拟环境：

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e .
```

根据需要安装可选功能，例如 DETR、RF-DETR、CLIP：

```bash
pip install -e "[detr]"   # 可选：DETR 后端
pip install -e "[clip]"   # 可选：CLIP 零样本支持
```

## 最小示例（单张检测）

```python
from visionframework import Detector
import cv2

det = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
det.initialize()
img = cv2.imread("path/to/image.jpg")
detections = det.detect(img)
print(f"Found {len(detections)} detections")
```

## 批处理示例（**推荐用于视频**）

```python
from visionframework import VisionPipeline
import cv2

# 初始化带批处理的管道
pipeline = VisionPipeline({
    "detector_config": {"model_type": "yolo", "batch_inference": True},
    "enable_tracking": True
})
pipeline.initialize()

# 批量处理多帧 - 性能提升 4 倍！
frames = [cv2.imread(f"frame_{i}.jpg") for i in range(4)]
results = pipeline.process_batch(frames)

for i, result in enumerate(results):
    print(f"Frame {i}: {len(result['detections'])} detections, {len(result['tracks'])} tracks")
```

注意：首次运行若缺模型会自动下载（需联网）。

## 运行示例脚本

仓库中的 `examples/` 包含按功能组织的示例，推荐从 `examples/README.md` 查看说明。

```bash
# 运行检测示例
python examples/detect_basic.py

# 运行视频追踪示例（自动使用批处理）
python examples/video_tracking.py
```

## 性能提示

- 💡 **视频处理**：使用 `pipeline.process_batch()` 而不是逐帧 `process()`，性能提升 **4 倍**
- 💡 **GPU 加速**：设置 `device: "cuda"` 以充分利用 GPU 批处理能力
- 💡 **FP16 加速**：在 GPU 上启用 `use_fp16: true` 以进一步加速

更多 API 细节请参阅 `docs/QUICK_REFERENCE.md` 和 `BATCH_PROCESSING_GUIDE.md`。
