"""
将 YOLO11 / YOLO26 全尺寸实例分割结果用 Visualizer 绘制并保存为 PNG。

默认输出目录：仓库根下 ``outputs/segmentation_viz/``（见 ``.gitignore``）。

用法::

    python -m visionframework.tools.save_yolo_seg_visualization
    python -m visionframework.tools.save_yolo_seg_visualization --out D:/viz --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]

YOLO_SEG_CASES = [
    ("YOLO11Segmenter", "yolo11n-seg.pt"),
    ("YOLO11Segmenter", "yolo11s-seg.pt"),
    ("YOLO11Segmenter", "yolo11m-seg.pt"),
    ("YOLO11Segmenter", "yolo11l-seg.pt"),
    ("YOLO11Segmenter", "yolo11x-seg.pt"),
    ("YOLO26Segmenter", "yolo26n-seg.pt"),
    ("YOLO26Segmenter", "yolo26s-seg.pt"),
    ("YOLO26Segmenter", "yolo26m-seg.pt"),
    ("YOLO26Segmenter", "yolo26l-seg.pt"),
    ("YOLO26Segmenter", "yolo26x-seg.pt"),
]

QUICK_CASES = [
    ("YOLO11Segmenter", "yolo11n-seg.pt"),
    ("YOLO26Segmenter", "yolo26n-seg.pt"),
]


def _load_bus_bgr() -> "object":
    for p in (REPO_ROOT / "test_bus.jpg", REPO_ROOT / "test" / "fixtures" / "bus.jpg"):
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    print("未找到 test_bus.jpg 或 test/fixtures/bus.jpg", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="保存 YOLO seg 可视化 PNG")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "segmentation_viz",
        help="输出目录",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="仅 yolo11n-seg + yolo26n-seg",
    )
    args = p.parse_args(argv)

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("请先安装: pip install ultralytics", file=sys.stderr)
        return 1

    import visionframework.algorithms.segmentation.yolo_segmenter  # noqa: F401
    from visionframework import Visualizer
    from visionframework.core.registry import ALGORITHMS

    img = _load_bus_bgr()
    cases = QUICK_CASES if args.quick else YOLO_SEG_CASES
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = Visualizer()
    ok, skip = 0, 0
    for cls_name, hub_id in cases:
        cls = ALGORITHMS.get(cls_name)
        try:
            seg = cls(weights=hub_id, device="cpu", conf=0.25)
            dets = seg.predict(img)
        except RuntimeError as e:
            msg = str(e).lower()
            if "zip" in msg or "central directory" in msg or "pytorchstreamreader" in msg:
                print(f"[skip] {hub_id}: 权重缓存损坏 — {e}")
                skip += 1
                continue
            raise
        stem = hub_id.replace(".pt", "").replace("-", "_")
        fname = f"{stem}_viz.png"
        drawn = vis.draw(img.copy(), {"detections": dets})
        path = out_dir / fname
        if cv2.imwrite(str(path), drawn):
            print(f"[ok] {fname}  det={len(dets)}  -> {path}")
            ok += 1
        else:
            print(f"[fail] imwrite {path}", file=sys.stderr)
            return 1

    print(f"完成: 保存 {ok} 张，跳过 {skip} 个，目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
