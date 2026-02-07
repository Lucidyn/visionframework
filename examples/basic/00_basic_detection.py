"""
00 - 基本目标检测
=================
最简单的用法：3 行代码检测图片中的物体。

source 参数支持：图片路径、视频路径、摄像头 (0)、RTSP 流、文件夹、路径列表。
"""

from visionframework import Vision

# ── 创建 Vision 实例 ──
v = Vision(model="yolov8n.pt")

# ── 运行检测 ──
source = "test.jpg"  # 换成你的图片 / 视频 / 摄像头 / 文件夹

for frame, meta, result in v.run(source):
    detections = result["detections"]
    print(f"[{meta.get('source_path', 'frame')}] 检测到 {len(detections)} 个物体")
    for det in detections:
        print(f"  {det.class_name}: {det.confidence:.2f}  bbox={det.bbox}")
