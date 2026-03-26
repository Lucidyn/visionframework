"""
DeepSORT-style association: IoU cost blended with cosine distance on embeddings.

When ``Detection._embedding`` is missing (plain detection pipeline), behavior
falls back to IoU-only matching like :class:`IOUTracker`.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .utils import calculate_iou, linear_assignment
from visionframework.data.detection import Detection
from visionframework.data.track import Track


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


class _DeepTrack:
    """Wraps :class:`Track` with an optional appearance embedding (EMA)."""

    __slots__ = ("track", "embedding", "_ema_alpha")

    def __init__(
        self,
        track_id: int,
        det: Detection,
        ema_alpha: float,
    ):
        self.track = Track(
            track_id=track_id,
            bbox=det.bbox,
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
        )
        emb = getattr(det, "_embedding", None)
        self.embedding = _norm_vec(emb) if emb is not None else None
        self._ema_alpha = ema_alpha

    def predict(self):
        self.track.predict()

    def update(self, det: Detection):
        self.track.update(det.bbox, det.confidence)
        emb = getattr(det, "_embedding", None)
        if emb is not None:
            e = _norm_vec(emb)
            if self.embedding is None:
                self.embedding = e
            else:
                a = self._ema_alpha
                self.embedding = a * self.embedding + (1.0 - a) * e
                self.embedding = _norm_vec(self.embedding)


class DeepSortTracker:
    """IoU + optional appearance (cosine) association with EMA gallery per track.

    Parameters
    ----------
    max_age : int
        Max frames without a match before the track is removed.
    min_hits : int
        Minimum ``Track.age`` before the track is returned (same as IOUTracker).
    iou_threshold : float
        Minimum IoU for IoU-only pairs (no embeddings).
    lambda_emb : float
        Weight on embedding distance; ``1 - lambda_emb`` weights IoU cost.
    emb_thresh : float
        Gate on cosine distance when blending thresholds for combined cost.
    ema_alpha : float
        EMA factor for updating stored embeddings (higher = slower change).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        lambda_emb: float = 0.5,
        emb_thresh: float = 0.25,
        ema_alpha: float = 0.9,
        **_kw,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.lambda_emb = float(lambda_emb)
        self.emb_thresh = float(emb_thresh)
        self.ema_alpha = float(ema_alpha)
        self.tracks: List[_DeepTrack] = []
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

    def _cost_matrix(
        self, detections: List[Detection], tracks: List[_DeepTrack]
    ) -> np.ndarray:
        m, n = len(tracks), len(detections)
        cost = np.zeros((m, n), dtype=np.float32)
        le = self.lambda_emb
        li = 1.0 - le
        for i, tr in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = calculate_iou(tr.track.bbox, det.bbox)
                iou_c = 1.0 - iou
                de = getattr(det, "_embedding", None)
                if tr.embedding is not None and de is not None:
                    de_n = _norm_vec(de)
                    cos = float(np.dot(tr.embedding, de_n))
                    emb_c = 1.0 - cos
                    cost[i, j] = li * iou_c + le * emb_c
                else:
                    cost[i, j] = iou_c
        return cost

    def _assignment_thresh(self) -> float:
        le = self.lambda_emb
        li = 1.0 - le
        t_iou = 1.0 - self.iou_threshold
        t_emb = self.emb_thresh
        return float(li * t_iou + le * t_emb)

    def _associate(self, detections: List[Detection], tracks: List[_DeepTrack]):
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        cost = self._cost_matrix(detections, tracks)
        matches, u_dets, u_tracks = linear_assignment(
            cost_matrix=cost, thresh=self._assignment_thresh()
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
            self.tracks.append(_DeepTrack(self.next_id, det, self.ema_alpha))
            self.next_id += 1

        self.tracks = [
            t for t in self.tracks if t.track.time_since_update <= self.max_age
        ]

        return [t.track for t in self.tracks if t.track.age >= self.min_hits]
