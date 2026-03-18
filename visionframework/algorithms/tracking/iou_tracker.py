"""
IoU-based multi-object tracker.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .utils import iou_cost_matrix, linear_assignment
from visionframework.data.detection import Detection
from visionframework.data.track import Track


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
        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            cost_matrix=cost,
            thresh=(1.0 - self.iou_threshold),
        )
        # matches: (K, 2) array of (track_idx, det_idx)
        return (
            [(int(t), int(d)) for t, d in matches],
            [int(i) for i in unmatched_dets],
            [int(i) for i in unmatched_tracks],
        )

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
