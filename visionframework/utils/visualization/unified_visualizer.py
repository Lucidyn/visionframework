"""
Unified visualizer combining all visualization capabilities
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .base_visualizer import BaseVisualizer
from .detection_visualizer import DetectionVisualizer
from .track_visualizer import TrackVisualizer
from .pose_visualizer import PoseVisualizer
from ...data.detection import Detection
from ...data.track import Track
from ...data.pose import Pose


class Visualizer(BaseVisualizer):
    """Unified visualizer for all vision results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.detection_viz = DetectionVisualizer(config)
        self.track_viz = TrackVisualizer(config)
        self.pose_viz = PoseVisualizer(config)
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw detections"""
        return self.detection_viz.draw_detections(image, detections)
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Track],
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw tracks"""
        return self.track_viz.draw_tracks(image, tracks, draw_history)
    
    def draw_poses(
        self,
        image: np.ndarray,
        poses: List[Pose],
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draw poses"""
        return self.pose_viz.draw_poses(image, poses, draw_skeleton, draw_keypoints, draw_bbox)
    
    def draw_results(
        self,
        image: np.ndarray,
        detections: Optional[List[Detection]] = None,
        tracks: Optional[List[Track]] = None,
        poses: Optional[List[Pose]] = None,
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw all results on image"""
        result = image.copy()
        
        # Draw tracks first (they have history trails)
        if tracks:
            result = self.draw_tracks(result, tracks, draw_history=draw_history)
        
        # Draw poses
        if poses:
            result = self.draw_poses(result, poses)
        
        # Draw detections if no tracks (or if explicitly requested)
        if detections and not tracks:
            result = self.draw_detections(result, detections)

        return result

    def draw_heatmap(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        *,
        alpha: float = 0.5,
        radius: int = 20,
        colormap: int = cv2.COLORMAP_JET,
        accumulate: bool = False,
        _heat_state: Optional[Dict] = None,
    ) -> np.ndarray:
        """Overlay a trajectory heatmap on *frame*.

        Each track's bounding-box centre is used as a heat point.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw on.
        tracks : list[Track]
            Current frame's tracks.
        alpha : float
            Blend weight for the heatmap overlay (0 = invisible, 1 = full).
        radius : int
            Gaussian blob radius in pixels.
        colormap : int
            OpenCV colormap constant (default ``cv2.COLORMAP_JET``).
        accumulate : bool
            If *True* and *_heat_state* is provided, accumulate heat across
            frames instead of resetting each call.
        _heat_state : dict | None
            Mutable dict ``{"heat": np.ndarray}`` for cross-frame accumulation.
            Pass the same dict on every call to build up a persistent heatmap.

        Returns
        -------
        np.ndarray
            Copy of *frame* with heatmap blended in.
        """
        h, w = frame.shape[:2]

        if accumulate and _heat_state is not None:
            if "heat" not in _heat_state or _heat_state["heat"].shape != (h, w):
                _heat_state["heat"] = np.zeros((h, w), dtype=np.float32)
            heat = _heat_state["heat"]
        else:
            heat = np.zeros((h, w), dtype=np.float32)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cx < w and 0 <= cy < h:
                # Draw a Gaussian blob centred at (cx, cy)
                x_start = max(0, cx - radius)
                x_end = min(w, cx + radius + 1)
                y_start = max(0, cy - radius)
                y_end = min(h, cy + radius + 1)
                for iy in range(y_start, y_end):
                    for ix in range(x_start, x_end):
                        dist2 = (ix - cx) ** 2 + (iy - cy) ** 2
                        heat[iy, ix] += np.exp(-dist2 / (2 * (radius / 3) ** 2))

        if accumulate and _heat_state is not None:
            _heat_state["heat"] = heat

        # Normalise to [0, 255]
        max_val = heat.max()
        if max_val > 0:
            heat_norm = (heat / max_val * 255).astype(np.uint8)
        else:
            heat_norm = heat.astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_norm, colormap)
        result = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)
        return result

    def draw(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        """Convenience wrapper: draw everything in a result dict.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        result : dict
            The *result* dict from ``Vision.run()``.

        Returns
        -------
        np.ndarray
            Annotated copy of *frame*.
        """
        return self.draw_results(
            frame,
            detections=result.get("detections"),
            tracks=result.get("tracks"),
            poses=result.get("poses"),
            **kwargs,
        )

