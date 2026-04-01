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
import tempfile
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

_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
_ASSETS_Y26 = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def _seg_yaml_for_hub(hub_id: str, cls_name: str) -> Path:
    stem = Path(hub_id).stem
    base = stem.replace("-seg", "")
    family = "yolo11" if cls_name == "YOLO11Segmenter" else "yolo26"
    return REPO_ROOT / "configs" / "segmentation" / family / f"{base}_seg.yaml"


def _download_seg_pt(name: str, dest: Path) -> bool:
    import urllib.request

    urls = (
        [f"{_ASSETS_Y26}/{name}", f"{_ASSETS_BASE}/{name}"]
        if name.startswith("yolo26")
        else [f"{_ASSETS_BASE}/{name}", f"{_ASSETS_Y26}/{name}"]
    )
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(dest))
            return dest.is_file() and dest.stat().st_size > 1000
        except OSError:
            continue
    return False


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

    import visionframework.algorithms.segmentation.yolo_segmenter  # noqa: F401
    from visionframework import Visualizer
    from visionframework.core.registry import ALGORITHMS

    img = _load_bus_bgr()
    cases = QUICK_CASES if args.quick else YOLO_SEG_CASES
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="vf_seg_w_"))

    vis = Visualizer()
    ok, skip = 0, 0
    for cls_name, hub_id in cases:
        cls = ALGORITHMS.get(cls_name)
        wpath = Path(hub_id)
        if not wpath.is_file():
            wpath = cache_dir / hub_id
            if not _download_seg_pt(hub_id, wpath):
                print(f"[skip] {hub_id}: 无法下载权重", file=sys.stderr)
                skip += 1
                continue
        ypath = _seg_yaml_for_hub(hub_id, cls_name)
        if not ypath.is_file():
            print(f"[skip] {hub_id}: 缺少配置 {ypath}", file=sys.stderr)
            skip += 1
            continue
        try:
            seg = cls(
                weights=str(wpath),
                device="cpu",
                conf=0.25,
                model_yaml=str(ypath),
            )
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
