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

## 最小示例（检测）

```python
from visionframework import Detector
import cv2

det = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
det.initialize()
img = cv2.imread("path/to/image.jpg")
detections = det.detect(img)
print(f"Found {len(detections)} detections")
```

注意：首次运行若缺模型会自动下载（需联网）。

## 运行示例脚本

仓库中的 `examples/` 包含按功能组织的示例，推荐从 `examples/README.md` 查看说明。

```bash
# 运行检测示例
python examples/detect_basic.py
```

更多 API 细节请参阅 `docs/QUICK_REFERENCE.md`。
