"""
02 - 从配置文件加载
===================
使用 JSON/YAML 配置文件创建 Vision 实例，便于管理不同场景的配置。
"""

import json
import tempfile
from pathlib import Path
from visionframework import Vision

# ── 1. 从 dict 创建（适合代码中动态配置）──
v1 = Vision.from_config({
    "model": "yolov8n.pt",
    "track": True,
    "conf": 0.3,
})
print(f"v1: {v1}")

# ── 2. 从 JSON 文件创建（适合运维/部署场景）──
config = {
    "model": "yolov8n.pt",
    "model_type": "yolo",
    "device": "auto",
    "conf": 0.25,
    "track": True,
    "tracker": "bytetrack",
    "pose": False,
    "segment": False,
}

# 写一个临时配置文件做演示
config_path = Path(tempfile.gettempdir()) / "vision_config.json"
config_path.write_text(json.dumps(config, indent=2))
print(f"配置文件: {config_path}")

v2 = Vision.from_config(config_path)
print(f"v2: {v2}")

# ── 3. 处理媒体 ──
source = "test.jpg"
for frame, meta, result in v2.run(source):
    print(f"检测到 {len(result['detections'])} 个物体")
