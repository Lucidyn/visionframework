"""
batch_processing.py
批量图像处理示例：展示 `Detector.detect` 接受图像列表（若后端支持批量）并返回按图像分组的检测结果。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    from visionframework import Detector
except Exception:
    Detector = None

import cv2


def main():
    if Detector is None:
        print("visionframework 未安装或无法导入。请先安装：pip install -e .")
        return

    det = Detector({"model_path": "yolov8n.pt", "conf_threshold": 0.25})
    det.initialize()

    # 加载多张图片（示例用同一张图复制）
    img = cv2.imread("your_image.jpg")
    if img is None:
        print("未找到 your_image.jpg，示例结束。")
        return

    imgs = [img, img.copy(), img.copy()]

    # 若后端不支持批量推理，框架应回退到逐张调用；这里示例调用统一接口
    results = det.detect(imgs)

    # 结果为按图像分组的检测列表
    for i, r in enumerate(results):
        print(f"Image {i}: {len(r)} detections")


if __name__ == "__main__":
    main()
