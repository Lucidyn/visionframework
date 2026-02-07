"""
03 - 姿态估计
=============
开启 pose=True 即可检测人体关键点。
"""

from visionframework import Vision

# ── 创建带姿态估计的 Vision ──
v = Vision(model="yolov8n-pose.pt", pose=True)

source = "test.jpg"

for frame, meta, result in v.run(source):
    poses = result["poses"]
    print(f"检测到 {len(poses)} 个人体姿态")
    for i, pose in enumerate(poses):
        print(f"  人物 {i}: {len(pose.keypoints)} 个关键点, 置信度={pose.confidence:.2f}")
