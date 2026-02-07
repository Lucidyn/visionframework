"""
09 - 多模态处理
===============
结合检测、跟踪、姿态估计等多种能力。

最简单的方式是直接用 Vision 类开启所有功能：
"""

import cv2
from visionframework import Vision

def main() -> None:
    # ── 一行开启全部能力 ──
    v = Vision(model="yolov8n.pt", track=True, pose=True, conf=0.25)

    source = "test.jpg"  # 图片 / 视频 / 摄像头 / 文件夹

    for frame, meta, result in v.run(source):
        detections = result["detections"]
        tracks = result["tracks"]
        poses = result["poses"]

        print(f"[{meta.get('source_path', 'frame')}] "
              f"检测 {len(detections)}, 轨迹 {len(tracks)}, 姿态 {len(poses)}")

        annotated = v.draw(frame, result)
        cv2.imshow("Multimodal", annotated)
        if cv2.waitKey(1 if meta.get("is_video") else 0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    v.cleanup()


if __name__ == "__main__":
    main()
