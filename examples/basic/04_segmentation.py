"""
04 - 实例分割
=============
开启 segment=True 即可进行实例分割。
"""

from visionframework import Vision

# ── 创建带分割的 Vision ──
v = Vision(model="yolov8n-seg.pt", segment=True)

source = "test.jpg"

for frame, meta, result in v.run(source):
    detections = result["detections"]
    print(f"检测到 {len(detections)} 个物体")
    for det in detections:
        has_mask = det.mask is not None
        print(f"  {det.class_name}: conf={det.confidence:.2f}, mask={'有' if has_mask else '无'}")
