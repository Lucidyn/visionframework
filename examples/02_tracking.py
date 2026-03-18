"""
示例 02 — 多目标跟踪

通过 YAML 配置文件驱动 ByteTrack 跟踪器。
可处理视频文件、摄像头或图片序列。
"""

import numpy as np
from visionframework import TaskRunner

task = TaskRunner("runs/tracking/bytetrack/tracking.yaml")

# 模拟处理多帧
for i in range(5):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = task.process(frame)
    tracks = result.get("tracks", [])
    print(f"帧 {i}: {len(tracks)} 个跟踪目标")

# 重置跟踪器状态
task.reset()
print("跟踪器已重置")
