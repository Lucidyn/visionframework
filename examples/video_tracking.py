"""
video_tracking.py
视频跟踪示例（简单、注释）

用法:
  python examples/video_tracking.py /path/to/video.mp4

示例说明：使用 `VisionPipeline` 初始化并在视频上运行检测+跟踪，处理有限帧并保存首个可视化结果。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    from visionframework import VisionPipeline, Visualizer
except Exception:
    VisionPipeline = None
    Visualizer = None

import cv2


def main():
    if VisionPipeline is None:
        print("visionframework 未安装或无法导入。请在项目根目录运行：\npython -m pip install -e .")
        return

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default=None)
    args = p.parse_args()

    # 简单配置：启用跟踪
    cfg = {
        "enable_tracking": True,
        "detector_config": {"model_path": "yolov8n.pt", "conf_threshold": 0.25},
        "tracker_config": {"tracker_type": "byte", "max_age": 30}
    }

    pipeline = VisionPipeline(cfg)
    pipeline.initialize()

    # 打开视频（或摄像头）
    if args.video is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    viz = Visualizer() if Visualizer is not None else None

    saved = False
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 200:
            break
        frame_count += 1

        results = pipeline.process(frame)
        tracks = results.get("tracks", [])

        if viz is not None and not saved:
            # 将首帧可视化并保存为示例输出
            out = viz.draw_tracks(frame.copy(), tracks, draw_history=True)
            cv2.imwrite("video_tracking_out.jpg", out)
            saved = True

    cap.release()
    print(f"处理完成，共处理 {frame_count} 帧；结果样本保存到 video_tracking_out.jpg（若可视化可用）。")


if __name__ == "__main__":
    main()
