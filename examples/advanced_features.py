"""
advanced_features.py
高级功能示例：ROI 检测、简单计数、性能统计示例

说明：本示例演示如何在连续帧上维持简单计数器（进入/离开），以及记录处理时间以做性能监控。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    from visionframework import Detector
except Exception:
    Detector = None

import cv2
import time


def main():
    if Detector is None:
        print("visionframework 未安装或无法导入。请先安装：pip install -e .")
        return

    det = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
    det.initialize()

    # 简单 ROI：矩形 (x1, y1, x2, y2)
    roi = (100, 100, 400, 300)

    # 进入计数器：如果检测框中心进入 ROI 则计数（演示用途）
    counter = 0

    # 打开摄像头并运行若干帧
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    times = []
    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.time()
        detections = det.detect(frame)
        elapsed = time.time() - t0
        times.append(elapsed)

        # 检测是否有对象进入 ROI
        for d in detections:
            # 假设 Detection 有 bbox 属性 [x1,y1,x2,y2]
            bbox = getattr(d, "bbox", None) or d.get("bbox") if isinstance(d, dict) else None
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                counter += 1
                break

    cap.release()
    avg = sum(times) / len(times) if times else 0
    print(f"处理 {frame_idx} 帧，平均检测耗时：{avg:.3f}s，简单进入计数（可能重复统计）：{counter}")


if __name__ == "__main__":
    main()
