"""
basic_usage.py
基本使用示例（详细注释）

用法:
  python examples/basic_usage.py /path/to/image.jpg

此示例展示如何用最少的代码初始化 `Detector`、运行检测并将结果可视化保存。
"""
import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，方便直接运行 examples
sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    from visionframework import Detector, Visualizer
except Exception:
    Detector = None
    Visualizer = None

import cv2
import numpy as np


def main():
    if Detector is None:
        print("visionframework 未安装或无法导入。请在项目根目录运行：\npython -m pip install -e .")
        return

    # 从命令行读取图像路径（可选）
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default=None)
    args = p.parse_args()

    if args.image is None:
        # 没有传入图片时，创建一张灰色占位图用于演示
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
    else:
        img = cv2.imread(args.image)
        if img is None:
            print("无法读取图片，已退出。")
            return

    # 初始化检测器，最小配置示例
    det = Detector({
        "model_path": "yolov8n.pt",  # 首次会自动下载（需联网）
        "conf_threshold": 0.25,
        "device": "cpu"
    })
    det.initialize()

    # 运行检测（单张图）
    detections = det.detect(img)
    print(f"检测到 {len(detections)} 个对象")

    # 可视化结果并保存
    if Visualizer is not None:
        viz = Visualizer()
        out = viz.draw_detections(img.copy(), detections)
        out_path = "basic_usage_out.jpg"
        cv2.imwrite(out_path, out)
        print(f"可视化结果已保存到 {out_path}")
    else:
        print("Visualizer 不可用，跳过可视化。")


if __name__ == "__main__":
    main()
