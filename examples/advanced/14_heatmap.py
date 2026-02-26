"""
14 - 轨迹热力图
================
使用 Visualizer.draw_heatmap() 生成目标运动热力图。

支持两种模式：
  - 单帧热力图：每帧独立渲染
  - 累积热力图：跨帧叠加，展示整体运动轨迹
"""

import cv2
import numpy as np
from visionframework import Vision, Visualizer


def main() -> None:
    v = Vision(model="yolov8n.pt", track=True, conf=0.3)
    vis = Visualizer()

    source = "video.mp4"   # 替换为你的视频路径

    # 累积热力图状态（跨帧共享）
    heat_state: dict = {}

    for frame, meta, result in v.run(source, skip_frames=1):
        tracks = result.get("tracks", [])

        # ── 1. 单帧热力图 ──
        single_heatmap = vis.draw_heatmap(frame, tracks, alpha=0.5, radius=25)

        # ── 2. 累积热力图 ──
        accum_heatmap = vis.draw_heatmap(
            frame, tracks,
            alpha=0.4,
            radius=20,
            accumulate=True,
            _heat_state=heat_state,
        )

        # 拼接两种模式并排显示
        combined = np.hstack([single_heatmap, accum_heatmap])
        h, w = combined.shape[:2]
        cv2.putText(combined, "Single Frame", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Accumulated", (w // 2 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Heatmap", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    v.cleanup()


if __name__ == "__main__":
    main()
