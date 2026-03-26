"""
命令行推理入口：单图 / 目录 / 视频 / 摄像头，可选保存可视化结果。

用法::

    vf-run -c runs/detection/yolo11/detect.yaml -s test_bus.jpg -o out_dir
    python -m visionframework.tools.run_inference --config ... --source ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from visionframework import TaskRunner, Visualizer
from visionframework.utils.logging_config import configure_visionframework_logging


def _parse_source(s: str):
    s = s.strip()
    if s.isdigit():
        return int(s)
    return Path(s)


def main(argv: list[str] | None = None) -> int:
    configure_visionframework_logging()
    p = argparse.ArgumentParser(
        description="VisionFramework: run TaskRunner on image, directory, video, or camera index.",
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Runtime YAML (e.g. runs/detection/yolo11/detect.yaml)",
    )
    p.add_argument(
        "-s",
        "--source",
        required=True,
        help="Image path, directory of images, video path, or camera index (e.g. 0)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output directory for visualized images (image / directory sources only)",
    )
    p.add_argument(
        "--strict-weights",
        action="store_true",
        help="Fail if weights path in config is missing",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (for video / long runs)",
    )
    args = p.parse_args(argv)

    if not args.config.is_file():
        print(f"Config not found: {args.config.resolve()}", file=sys.stderr)
        return 1

    task = TaskRunner(args.config, strict_weights=args.strict_weights)
    source = _parse_source(args.source)
    vis = Visualizer() if args.out else None
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    for frame, meta, result in task.run(source):
        if args.max_frames is not None and n >= args.max_frames:
            break
        dets = result.get("detections") or []
        tracks = result.get("tracks")
        seg = "detections" if tracks is None else "tracks"
        count = len(dets) if tracks is None else len(tracks)
        path_hint = meta.get("path", "")
        print(f"[{n}] {meta.get('type', '?')} {path_hint} {seg}={count}")

        if vis is not None and args.out is not None:
            drawn = vis.draw(frame, result)
            if meta.get("type") == "image" and meta.get("path"):
                stem = Path(meta["path"]).stem
                out_path = args.out / f"{stem}_vf.jpg"
            else:
                out_path = args.out / f"frame_{n:06d}_vf.jpg"
            cv2.imwrite(str(out_path), drawn)
        n += 1

    print(f"Done. Processed {n} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
