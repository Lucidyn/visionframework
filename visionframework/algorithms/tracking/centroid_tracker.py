"""
Centroid-based multi-object tracker.

Associates detections to tracks by Euclidean distance between box centers
(Hungarian / greedy via :func:`linear_assignment`).
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .utils import linear_assignment
from visionframework.data.detection import Detection
from visionframework.data.track import Track


def _centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist_cost_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    m, n = len(tracks), len(detections)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float32)
    cost = np.zeros((m, n), dtype=np.float32)
    for i, t in enumerate(tracks):
        cx1, cy1 = _centroid(t.bbox)
        for j, d in enumerate(detections):
            cx2, cy2 = _centroid(d.bbox)
            cost[i, j] = float(np.hypot(cx1 - cx2, cy1 - cy2))
    return cost


class CentroidTracker:
    """Associate detections to tracks by centroid distance.

    Parameters
    ----------
    max_distance : float
        Maximum center distance (pixels) for a valid match (passed as assignment threshold).
    max_age : int
        Max frames a track survives without a match (cf. IOUTracker).
    min_hits : int
        Minimum ``track.age`` before the track is returned.
    """

    def __init__(
        self,
        max_distance: float = 64.0,
        max_age: int = 30,
        min_hits: int = 3,
        **_kw,
    ):
        self.max_distance = float(max_distance)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.tracks: List[Track] = []
        self.next_id = 0

    def reset(self):
        self.tracks = []
        self.next_id = 0

    @staticmethod
    def _validate_detections(detections):
        if detections is None:
            return []
        if not isinstance(detections, (list, tuple)):
            detections = list(detections)
        return detections

    def _associate(self, detections, tracks):
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        cost = _dist_cost_matrix(tracks, detections)
        matches, u_dets, u_tracks = linear_assignment(
            cost_matrix=cost, thresh=self.max_distance
        )
        return (
            [(int(t), int(d)) for t, d in matches],
            [int(i) for i in u_dets],
            [int(i) for i in u_tracks],
        )

    def update(self, detections, image=None) -> List[Track]:
        detections = self._validate_detections(detections)

        for track in self.tracks:
            track.predict()

        matches, u_dets, u_tracks = self._associate(detections, self.tracks)

        for tidx, didx in matches:
            det = detections[didx]
            self.tracks[tidx].update(det.bbox, det.confidence)

        for didx in u_dets:
            det = detections[didx]
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.age >= self.min_hits]
