"""
Runner: reads data sources and feeds frames to a pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from visionframework.pipelines.base import BasePipeline


class Runner:
    """Iterate over a data source and run a pipeline on every frame.

    Supported source types
    ----------------------
    * ``str`` / ``Path`` — image file, video file, RTSP/HTTP stream, or directory of images.
    * ``int`` — camera index.
    * ``np.ndarray`` — single BGR image.
    * ``list`` — list of any of the above.
    """

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, pipeline: BasePipeline):
        self.pipeline = pipeline

    def run(self, source) -> Iterator[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        """Yield ``(frame, meta, result)`` for every frame in *source*."""
        for frame, meta in self._iter_frames(source):
            result = self.pipeline.process(frame)
            yield frame, meta, result

    def _iter_frames(self, source) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        if isinstance(source, np.ndarray):
            yield source, {"type": "image", "frame_index": 0}
            return

        if isinstance(source, (list, tuple)):
            for idx, item in enumerate(source):
                for frame, meta in self._iter_frames(item):
                    meta["source_index"] = idx
                    yield frame, meta
            return

        if isinstance(source, int):
            yield from self._read_video(source, is_camera=True)
            return

        path = Path(str(source))
        if path.is_dir():
            files = sorted(
                f for f in path.rglob("*") if f.suffix.lower() in self._IMAGE_EXTS
            )
            for idx, f in enumerate(files):
                img = cv2.imread(str(f))
                if img is not None:
                    yield img, {"type": "image", "path": str(f), "frame_index": idx}
            return

        if path.suffix.lower() in self._IMAGE_EXTS:
            img = cv2.imread(str(path))
            if img is not None:
                yield img, {"type": "image", "path": str(path), "frame_index": 0}
            return

        yield from self._read_video(str(source))

    def _read_video(self, source, is_camera=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                meta = {
                    "type": "camera" if is_camera else "video",
                    "path": str(source),
                    "frame_index": idx,
                    "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1,
                }
                yield frame, meta
                idx += 1
        finally:
            cap.release()
