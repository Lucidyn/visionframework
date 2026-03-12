"""
Facebook DETR 官方权重转换工具。

将官方 DETR checkpoint 转换为 VisionFramework 兼容格式。
支持 DETR-R50、DETR-R101 等 ResNet backbone 的变体。

用法:
    # 下载并转换
    python tools/convert_detr.py --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output weights/detr_r50.pth

    # 从本地文件转换
    python tools/convert_detr.py --input detr-r50.pth --output weights/detr_r50.pth

    # 转换并验证
    python tools/convert_detr.py --input detr-r50.pth --output weights/detr_r50.pth --verify --image test.jpg

已知支持的官方 checkpoint:
    DETR-R50:  https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
    DETR-R101: https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth
    DETR-DC5-R50:  https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth
    DETR-DC5-R101: https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth
"""

import re
import argparse
import sys
from pathlib import Path
from collections import OrderedDict
import urllib.request

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_mapping(official_sd: dict, our_sd: dict) -> OrderedDict:
    """构建官方 → VisionFramework 的 key 映射。

    Returns
    -------
    mapping : OrderedDict
        {official_key: framework_key} 且 shape 匹配。
    """
    mapping = OrderedDict()

    for ok in official_sd.keys():
        vk = None

        m = re.match(r"backbone\.0\.body\.(.*)", ok)
        if m:
            vk = f"backbone.{m.group(1)}"

        if vk is None:
            m = re.match(r"input_proj\.(.*)", ok)
            if m:
                vk = f"neck.input_proj.{m.group(1)}"

        if vk is None:
            m = re.match(r"transformer\.encoder\.(.*)", ok)
            if m:
                vk = f"neck.encoder.{m.group(1)}"

        if vk is None:
            m = re.match(r"transformer\.decoder\.(.*)", ok)
            if m:
                vk = f"head.decoder.{m.group(1)}"

        if ok == "query_embed.weight":
            vk = "head.query_embed.weight"

        if vk is None:
            m = re.match(r"class_embed\.(.*)", ok)
            if m:
                vk = f"head.class_head.{m.group(1)}"

        if vk is None:
            m = re.match(r"bbox_embed\.layers\.(\d+)\.(.*)", ok)
            if m:
                linear_idx = int(m.group(1)) * 2
                vk = f"head.bbox_head.net.{linear_idx}.{m.group(2)}"

        if vk is not None and vk in our_sd:
            if official_sd[ok].shape == our_sd[vk].shape:
                mapping[ok] = vk

    return mapping


def convert_weights(official_path: str, config_yaml: str = "configs/models/detr_r50.yaml"):
    """转换官方权重为 VisionFramework 格式。"""
    from visionframework.core.builder import build_model_from_file

    ckpt = torch.load(official_path, map_location="cpu", weights_only=False)
    official_sd = ckpt["model"] if "model" in ckpt else ckpt

    model = build_model_from_file(config_yaml)
    our_sd = model.state_dict()

    mapping = build_mapping(official_sd, our_sd)

    converted = OrderedDict()
    for ok, vk in mapping.items():
        converted[vk] = official_sd[ok]

    result = model.load_state_dict(converted, strict=False)
    n_total = len(official_sd)
    n_matched = len(mapping)
    n_missing = len(result.missing_keys)

    print(f"官方 key 数: {n_total}")
    print(f"成功映射: {n_matched}/{n_total}")
    print(f"框架 missing: {n_missing}")

    return converted, model


def main():
    parser = argparse.ArgumentParser(description="DETR 官方权重转换")
    parser.add_argument("--input", type=str, help="官方 checkpoint 路径")
    parser.add_argument("--url", type=str, help="官方 checkpoint URL (自动下载)")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--config", type=str, default="configs/models/detr_r50.yaml",
                        help="框架模型配置文件")
    parser.add_argument("--verify", action="store_true", help="转换后验证检测")
    parser.add_argument("--image", type=str, default=None, help="验证用图片路径")
    args = parser.parse_args()

    if args.url and not args.input:
        args.input = Path(args.url).name
        if not Path(args.input).exists():
            print(f"下载: {args.url}")
            urllib.request.urlretrieve(args.url, args.input)

    if not args.input or not Path(args.input).exists():
        print("错误: 请指定 --input 或 --url")
        sys.exit(1)

    converted_sd, model = convert_weights(args.input, args.config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_sd, args.output)
    print(f"已保存至: {args.output}")

    if args.verify:
        import cv2
        import numpy as np
        from visionframework.algorithms.detection.detr_detector import DETRDetector

        COCO_CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        img_path = args.image or "test_bus.jpg"
        if not Path(img_path).exists():
            url = "https://ultralytics.com/images/bus.jpg"
            urllib.request.urlretrieve(url, img_path)
        img = cv2.imread(img_path)

        detector = DETRDetector(
            model=model, device="cpu", conf=0.7,
            input_size=(800, 800), class_names=COCO_CLASSES,
        )
        dets = detector.predict(img)
        print(f"\n验证: 检测到 {len(dets)} 个目标 (conf>0.7)")
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            print(f"  [{d.class_name}] conf={d.confidence:.3f}")


if __name__ == "__main__":
    main()
