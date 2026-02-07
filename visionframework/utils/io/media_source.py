"""
Unified media input source for images, videos, streams, folders, and lists.

Supports:
- Single image path (e.g. "test.jpg")
- Single video path or stream URL (e.g. "video.mp4", "rtsp://...")
- Camera index (e.g. 0)
- Folder path: all images and videos inside (recursive optional)
- List of paths or camera indices (e.g. ["a.jpg", "b.mp4", 0])
- Single numpy array (BGR image) for API compatibility

Example:
    from visionframework.utils.io.media_source import iter_frames

    for frame, meta in iter_frames("test.jpg"):
        print(meta["source_path"], meta["frame_index"])
    for frame, meta in iter_frames(["img1.jpg", "img2.jpg"]):
        ...
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Iterator, Tuple, Optional, Any, Dict

from .video_utils import VideoProcessor
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

# Supported extensions (lowercase)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _is_stream_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(_STREAM_PREFIXES)


def _classify_file(path: Path) -> Optional[str]:
    """Return 'image', 'video', or None based on file extension."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _expand_folder(folder: Path, recursive: bool = False) -> List[Path]:
    """Return sorted list of media file paths under *folder*."""
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in _MEDIA_EXTENSIONS)


def _classify_str_item(item: str) -> Optional[Tuple[str, Any]]:
    """Classify a single string item into (type, value) or None."""
    path = Path(item).resolve()
    if path.is_file():
        kind = _classify_file(path)
        return (kind, str(path)) if kind else None
    if _is_stream_url(item):
        return ("video", item)
    return None


def _normalize_source(
    source: Union[str, int, List[Union[str, int]], np.ndarray, Path],
    recursive_folder: bool = False,
) -> List[Tuple[str, Any]]:
    """
    Normalize user input to a list of (type, item).
    type: "image" | "video" | "camera" | "array"
    """
    out: List[Tuple[str, Any]] = []

    def _expand_dir(path: Path) -> None:
        for p in _expand_folder(path, recursive=recursive_folder):
            kind = _classify_file(p)
            if kind:
                out.append((kind, str(p)))

    def _add_one(item: Any) -> None:
        if isinstance(item, np.ndarray):
            out.append(("array", item))
        elif isinstance(item, (int, np.integer)):
            out.append(("camera", int(item)))
        elif isinstance(item, (str, Path)):
            path = Path(item).resolve()
            if path.is_dir():
                _expand_dir(path)
            elif path.is_file():
                kind = _classify_file(path)
                if kind:
                    out.append((kind, str(path)))
                else:
                    logger.warning(f"Unsupported file extension, skipped: {path}")
            elif _is_stream_url(str(item)):
                out.append(("video", str(item)))
            else:
                # Might be a digit-string camera index like "0"
                s = str(item).strip()
                if s.isdigit():
                    out.append(("camera", int(s)))
                else:
                    logger.warning(f"Path does not exist, skipped: {item}")
        else:
            logger.warning(f"Unsupported source type, skipped: {type(item)}")

    # --- dispatch ---
    if isinstance(source, np.ndarray):
        out.append(("array", source))
    elif isinstance(source, (list, tuple)):
        for item in source:
            _add_one(item)
    else:
        _add_one(source)

    return out


# ---------------------------------------------------------------------------
#  Internal: iterate video/camera frames via VideoProcessor
# ---------------------------------------------------------------------------

def _iter_video_frames(
    proc: VideoProcessor,
    source_path: str,
    source_index: int,
    total_frames: int,
    start: int,
    end: Optional[int],
    skip: int,
) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    """Shared frame-reading loop for both video files and cameras."""
    frame_idx = -1
    try:
        while True:
            ret, frame = proc.read_frame()
            if not ret or frame is None:
                break
            frame_idx += 1
            if frame_idx < start:
                continue
            if end is not None and frame_idx > end:
                break
            if skip > 0 and (frame_idx - start) % (skip + 1) != 0:
                continue
            yield frame, {
                "source_path": source_path,
                "source_index": source_index,
                "frame_index": frame_idx,
                "is_video": True,
                "total_frames": total_frames,
            }
    finally:
        proc.close()


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def iter_frames(
    source: Union[str, int, List[Union[str, int]], np.ndarray, Path],
    *,
    recursive_folder: bool = False,
    video_skip_frames: int = 0,
    video_start_frame: int = 0,
    video_end_frame: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Iterate over frames from a unified media source.

    Yields ``(frame, meta)`` for each frame.

    ``meta`` dict keys:
        - source_path: str  (path, "camera:<index>", or "array")
        - source_index: int (index of this source in the source list)
        - frame_index: int  (frame index within current source; 0 for images)
        - is_video: bool
        - total_frames: int (-1 if unknown, e.g. camera)

    Args:
        source: Image path, video path/URL, camera index (int or "0"),
                folder path, list of the above, or single BGR numpy array.
        recursive_folder: If source is a folder, include subfolders.
        video_skip_frames: For video, skip N frames between reads (0 = every frame).
        video_start_frame: For video, start at this frame index (0-based).
        video_end_frame: For video, stop at this frame index (None = to end).

    Yields:
        (frame, meta)
    """
    items = _normalize_source(source, recursive_folder=recursive_folder)

    for source_index, (item_type, item) in enumerate(items):
        # --- numpy array ---
        if item_type == "array":
            yield item, {
                "source_path": "array",
                "source_index": source_index,
                "frame_index": 0,
                "is_video": False,
                "total_frames": 1,
            }
            continue

        # --- single image ---
        if item_type == "image":
            img = cv2.imread(str(item))
            if img is None:
                logger.warning(f"Failed to read image: {item}")
                continue
            yield img, {
                "source_path": str(item),
                "source_index": source_index,
                "frame_index": 0,
                "is_video": False,
                "total_frames": 1,
            }
            continue

        # --- camera or video ---
        is_camera = item_type == "camera"
        proc = VideoProcessor(item)
        if not proc.open():
            logger.warning(f"Failed to open {'camera' if is_camera else 'video'}: {item}")
            continue
        info = proc.get_info()
        total = -1 if is_camera else max(info.get("total_frames", -1) or -1, -1)
        label = f"camera:{item}" if is_camera else str(item)

        yield from _iter_video_frames(
            proc, label, source_index, total,
            start=video_start_frame,
            end=video_end_frame,
            skip=video_skip_frames,
        )
