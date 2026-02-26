"""
17 - 性能监控
=============
演示如何使用 PerformanceMonitor 监控推理性能。

指标：
  - FPS（当前/平均/最大/最小）
  - 帧时间统计
  - 组件级耗时（检测、跟踪等）
  - 内存使用（需 psutil）
"""

import time
import numpy as np
import cv2

from visionframework import Vision, PerformanceMonitor


def main() -> None:
    # ── 创建 Vision 实例 ──
    v = Vision(model="yolov8n.pt", track=True, conf=0.25)

    # ── 创建性能监控器 ──
    monitor = PerformanceMonitor(window_size=30)
    monitor.start()

    source = "video.mp4"   # 替换为你的视频路径

    frame_count = 0
    max_frames = 100  # 演示时只处理 100 帧

    for frame, meta, result in v.run(source):
        t_start = time.perf_counter()

        # ── 记录检测耗时 ──
        t_det = time.perf_counter()
        annotated = v.draw(frame, result)
        monitor.record_component_time("detection", time.perf_counter() - t_det)

        # ── 记录跟踪耗时（模拟） ──
        monitor.record_component_time("tracking", 0.002)

        # ── 打点 ──
        monitor.tick()

        # ── 每 10 帧打印一次指标 ──
        if frame_count % 10 == 0:
            metrics = monitor.get_metrics()
            print(f"帧 {frame_count:4d} | "
                  f"FPS: {metrics.fps:6.1f} | "
                  f"avg: {metrics.avg_fps:6.1f} | "
                  f"帧时间: {metrics.avg_time_per_frame*1000:5.1f}ms | "
                  f"内存: {metrics.avg_memory_usage:.0f}MB")

        # 显示
        cv2.putText(annotated,
                    f"FPS: {monitor.get_metrics().fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Performance Monitor", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
        if frame_count >= max_frames:
            break

    # ── 最终报告 ──
    print("\n── 性能报告 ──")
    report = monitor.get_detailed_report()
    if isinstance(report, dict):
        for section, data in report.items():
            print(f"  {section}: {data}")
    else:
        print(report)

    cv2.destroyAllWindows()
    v.cleanup()


if __name__ == "__main__":
    main()
