"""
IoU-based multi-object tracker.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .utils import iou_cost_matrix, SCIPY_AVAILABLE
from visionframework.data.detection import Detection
from visionframework.data.track import Track

if SCIPY_AVAILABLE:
    from scipy.optimize import linear_sum_assignment


class IOUTracker:
    """Multi-object tracker using IoU-based association.

    Parameters
    ----------
    max_age : int
        Max frames a track survives without a match.
    min_hits : int
        Minimum age before a track is returned.
    iou_threshold : float
        IoU threshold for matching.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, **_kw):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 0

    def reset(self):
        self.tracks = []
        self.next_id = 0

    def _associate(self, detections, tracks):
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        cost = iou_cost_matrix([t.bbox for t in tracks], [d.bbox for d in detections])

        if SCIPY_AVAILABLE:
            tidx, didx = linear_sum_assignment(cost)
            matches, u_dets, u_tracks = [], set(range(len(detections))), set(range(len(tracks)))
            for t, d in zip(tidx, didx):
                if cost[t, d] < (1.0 - self.iou_threshold):
                    matches.append((t, d))
                    u_dets.discard(d)
                    u_tracks.discard(t)
            return matches, list(u_dets), list(u_tracks)
        else:
            pairs = []
            for i in range(len(tracks)):
                for j in range(len(detections)):
                    c = cost[i, j]
                    if c < (1.0 - self.iou_threshold):
                        pairs.append((c, i, j))
            pairs.sort()
            matches, u_dets, u_tracks = [], set(range(len(detections))), set(range(len(tracks)))
            for _, i, j in pairs:
                if i in u_tracks and j in u_dets:
                    matches.append((i, j))
                    u_tracks.remove(i)
                    u_dets.remove(j)
            return matches, list(u_dets), list(u_tracks)

    @staticmethod
    def _validate_detections(detections) -> List[Detection]:
        if detections is None:
            return []
        if not isinstance(detections, (list, tuple)):
            detections = list(detections)
        return detections

    def update(self, detections, image=None) -> List[Track]:
        """Update tracker with new detections."""
        detections = self._validate_detections(detections)

        for track in self.tracks:
            track.predict()

        matches, u_dets, u_tracks = self._associate(detections, self.tracks)

        for tidx, didx in matches:
            det = detections[didx]
            self.tracks[tidx].update(det.bbox, det.confidence)

        for didx in u_dets:
            det = detections[didx]
            self.tracks.append(Track(
                track_id=self.next_id, bbox=det.bbox,
                confidence=det.confidence, class_id=det.class_id,
                class_name=det.class_name,
            ))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.age >= self.min_hits]
