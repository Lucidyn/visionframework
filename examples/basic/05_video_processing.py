"""
05 - 视频 / 摄像头 / RTSP 流处理
=================================
run() 方法统一处理所有媒体类型。
"""

import cv2
from visionframework import Vision

v = Vision(model="yolov8n.pt", track=True)

# ── 处理视频文件 ──
source = "video.mp4"  # 也可以是: 0 (摄像头), "rtsp://...", "folder/"

for frame, meta, result in v.run(source, skip_frames=2):
    detections = result["detections"]
    tracks = result["tracks"]

    # 在帧上绘制结果
    annotated = v.draw(frame, result)

    cv2.imshow("Vision Framework", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
v.cleanup()
