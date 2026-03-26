"""
SORT-style tracker: Kalman filter on (cx, cy, area, aspect) + IoU association.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .utils import iou_cost_matrix, linear_assignment
from visionframework.data.detection import Detection
from visionframework.data.track import Track


def _bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1e-6)
    h = max(float(y2 - y1), 1e-6)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    s = w * h
    r = w / h
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)


def _x_to_bbox(x: np.ndarray) -> Tuple[float, float, float, float]:
    cx, cy, s, r = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w if w > 1e-6 else 1.0
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return (x1, y1, x2, y2)


class _Kalman7:
    """Constant-velocity model: state [cx, cy, s, r, vx, vy, vs]."""

    def __init__(self):
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[0:4, 0:4] *= 0.01
        self.Q[4:7, 4:7] *= 0.0001
        self.R = np.eye(4, dtype=np.float32) * 0.01

    def initiate(self, z: np.ndarray):
        self.x[:4] = z
        self.x[4:] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(7, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


class _SortTrack:
    """Internal track with Kalman state and hit / age bookkeeping."""

    __slots__ = (
        "track_id",
        "kf",
        "hits",
        "time_since_update",
        "class_id",
        "class_name",
        "confidence",
    )

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        track_id: int,
        confidence: float,
        class_id: int,
        class_name,
    ):
        self.track_id = track_id
        self.kf = _Kalman7()
        self.kf.initiate(_bbox_to_z(bbox))
        self.hits = 1
        self.time_since_update = 0
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, det: Detection):
        self.kf.update(_bbox_to_z(det.bbox))
        self.time_since_update = 0
        self.hits += 1
        self.confidence = det.confidence
        self.class_id = det.class_id
        self.class_name = det.class_name

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return _x_to_bbox(self.kf.x[:4])

    def to_track(self) -> Track:
        t = Track(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )
        t.age = self.hits
        t.time_since_update = self.time_since_update
        t.history = [self.bbox]
        return t


class SortTracker:
    """SORT: Kalman prediction + Hungarian assignment on IoU cost (1 - IoU).

    Parameters
    ----------
    max_age : int
        Remove a track after this many frames without a match.
    min_hits : int
        Minimum consecutive hits before the track is returned.
    iou_threshold : float
        Assign matches only if IoU >= this value (cost threshold = 1 - iou).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        **_kw,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.tracks: List[_SortTrack] = []
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

    def _associate(self, detections: List[Detection], tracks: List[_SortTrack]):
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        tb = [t.bbox for t in tracks]
        db = [d.bbox for d in detections]
        cost = iou_cost_matrix(tb, db)
        matches, u_dets, u_tracks = linear_assignment(
            cost_matrix=cost,
            thresh=(1.0 - self.iou_threshold),
        )
        return (
            [(int(t), int(d)) for t, d in matches],
            [int(i) for i in u_dets],
            [int(i) for i in u_tracks],
        )

    def update(self, detections, image=None) -> List[Track]:
        detections = self._validate_detections(detections)

        for tr in self.tracks:
            tr.predict()

        matches, u_dets, u_tracks = self._associate(detections, self.tracks)

        for tidx, didx in matches:
            self.tracks[tidx].update(detections[didx])

        for didx in u_dets:
            det = detections[didx]
            self.tracks.append(
                _SortTrack(
                    det.bbox,
                    self.next_id,
                    det.confidence,
                    det.class_id,
                    det.class_name,
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t.to_track() for t in self.tracks if t.hits >= self.min_hits]