"""
01 - 检测 + 跟踪
=================
开启 track=True 即可在检测基础上进行多目标跟踪。
"""

from visionframework import Vision

# ── 创建带跟踪的 Vision ──
v = Vision(model="yolov8n.pt", track=True)

# ── 对视频进行检测 + 跟踪 ──
source = "video.mp4"  # 也可以是 0 (摄像头) 或 "rtsp://..."

for frame, meta, result in v.run(source):
    detections = result["detections"]
    tracks = result["tracks"]
    print(f"帧 {meta.get('frame_index', '?')}: "
          f"{len(detections)} 检测, {len(tracks)} 跟踪目标")
