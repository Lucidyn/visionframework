"""
RF-DETR 适配器 — 封装 Roboflow 的 rfdetr 包用于 VisionFramework。

RF-DETR 的模型架构非常复杂（多尺度 Deformable DETR + DINOv2），
直接映射权重到我们框架不现实，因此采用适配器模式直接调用 rfdetr 包推理。

依赖: pip install rfdetr

用法:
    python tools/rfdetr_adapter.py --model base --image test_bus.jpg
    python tools/rfdetr_adapter.py --model large --image test_bus.jpg --conf 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionframework.data.detection import Detection


def _get_coco_class_map() -> dict:
    """获取 COCO class id → 名称映射。优先使用 rfdetr 内置的映射（1-indexed）。"""
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES
        return COCO_CLASSES
    except ImportError:
        pass
    # 回退: COCO 80 类 1-indexed
    names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    coco_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    return {cid: name for cid, name in zip(coco_ids, names)}


class RFDETRAdapter:
    """RF-DETR 适配器 — 将 rfdetr 推理结果转为 VisionFramework Detection。

    Parameters
    ----------
    model_size : str
        模型大小: "nano", "small", "base", "medium", "large"。
    conf : float
        置信度阈值。
    resolution : int
        输入分辨率。
    """

    def __init__(self, model_size: str = "base", conf: float = 0.5, resolution: int = 560):
        try:
            import rfdetr
        except ImportError:
            raise ImportError(
                "RF-DETR 适配器需要安装 rfdetr 包: pip install rfdetr"
            )

        if model_size == "large":
            from rfdetr import RFDETRLarge
            self.model = RFDETRLarge(resolution=resolution)
        else:
            from rfdetr import RFDETRBase
            self.model = RFDETRBase(resolution=resolution)

        self.conf = conf
        self.model_size = model_size
        self.class_map = _get_coco_class_map()

    def predict(self, img: np.ndarray) -> list:
        """检测。

        Parameters
        ----------
        img : np.ndarray
            BGR 图片 (H, W, 3)。

        Returns
        -------
        list[Detection]
        """
        from PIL import Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        sv_dets = self.model.predict(pil_img, threshold=self.conf)

        detections = []
        if sv_dets.xyxy is not None and len(sv_dets.xyxy) > 0:
            for i in range(len(sv_dets.xyxy)):
                x1, y1, x2, y2 = sv_dets.xyxy[i].tolist()
                conf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 1.0
                cid = int(sv_dets.class_id[i]) if sv_dets.class_id is not None else -1
                name = self.class_map.get(cid)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cid,
                    class_name=name,
                ))

        return detections

    def predict_batch(self, images: list) -> list:
        return [self.predict(img) for img in images]


def main():
    parser = argparse.ArgumentParser(description="RF-DETR 适配器测试")
    parser.add_argument("--model", type=str, default="base",
                        choices=["nano", "small", "base", "medium", "large"])
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="rfdetr_result.jpg")
    args = parser.parse_args()

    print(f"加载 RF-DETR {args.model}...")
    adapter = RFDETRAdapter(model_size=args.model, conf=args.conf)

    img = cv2.imread(args.image)
    if img is None:
        print(f"错误: 无法读取图片 {args.image}")
        sys.exit(1)
    print(f"图片: {args.image} ({img.shape[1]}x{img.shape[0]})")

    dets = adapter.predict(img)
    print(f"\n检测到 {len(dets)} 个目标 (conf > {args.conf}):")
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        print(f"  [{d.class_name}] conf={d.confidence:.3f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    from visionframework.utils.visualization import Visualizer
    vis = Visualizer()
    drawn = vis.draw_detections(img.copy(), dets)
    cv2.imwrite(args.output, drawn)
    print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
