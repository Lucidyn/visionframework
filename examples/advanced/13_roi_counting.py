"""
13 - ROI 区域计数
=================
演示如何定义感兴趣区域 (ROI) 并统计进出人数。

新 API：
    v.add_roi(name, points)          # 注册多边形区域
    result["counts"]                 # 每帧的计数结果
"""

import cv2
import numpy as np
from visionframework import Vision

# ── 定义 ROI 区域 ──
# 矩形区域：左上 (100,100) → 右下 (500,400)
ENTRANCE_ZONE = [(100, 100), (500, 100), (500, 400), (100, 400)]
EXIT_ZONE     = [(520, 100), (900, 100), (900, 400), (520, 400)]


def draw_roi_overlay(frame: np.ndarray, roi_points: list, color=(0, 255, 0),
                     label: str = "") -> np.ndarray:
    """在帧上绘制 ROI 多边形。"""
    pts = np.array(roi_points, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    cv2.polylines(frame, [pts], True, color, 2)
    if label:
        x, y = pts[0]
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main() -> None:
    # ── 创建 Vision 实例，开启跟踪 ──
    v = Vision(model="yolov8n.pt", track=True, conf=0.3)

    # ── 注册 ROI 区域 ──
    v.add_roi("entrance", ENTRANCE_ZONE)
    v.add_roi("exit",     EXIT_ZONE)

    print("Vision 配置:", v.info())

    # ── 处理视频 ──
    source = "video.mp4"   # 替换为你的视频路径

    for frame, meta, result in v.run(source, skip_frames=1):
        # 绘制 ROI 区域
        frame = draw_roi_overlay(frame, ENTRANCE_ZONE, (0, 255, 0), "Entrance")
        frame = draw_roi_overlay(frame, EXIT_ZONE,     (0, 0, 255), "Exit")

        # 绘制检测/跟踪结果
        frame = v.draw(frame, result)

        # 显示计数
        counts = result.get("counts", {})
        y_offset = 30
        for zone_name, zone_counts in counts.items():
            text = (f"{zone_name}: inside={zone_counts.get('inside', 0)}  "
                    f"entered={zone_counts.get('total_entered', 0)}  "
                    f"exited={zone_counts.get('total_exited', 0)}")
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        cv2.imshow("ROI Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    v.cleanup()


if __name__ == "__main__":
    main()
