"""
ByteTrack multi-object tracker.

ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from .utils import iou_cost_matrix, linear_assignment
from visionframework.data.track import STrack
from visionframework.data.detection import Detection


class ByteTracker:
    """ByteTrack multi-object tracker.

    Parameters
    ----------
    track_thresh : float
        Detection confidence threshold to separate high/low.
    track_buffer : int
        Max frames a lost track is kept before removal.
    match_thresh : float
        IoU matching threshold.
    frame_rate : int
        Source frame rate (informational).
    min_box_area : float
        Minimum bbox area to consider.
    """

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30,
                 min_box_area: float = 10.0, **_kw):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.min_box_area = min_box_area

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0
        self.next_id = 1

    def reset(self):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1

    def _iou_distance(self, tracks, detections) -> np.ndarray:
        return iou_cost_matrix(
            [t.bbox for t in tracks],
            [d.bbox for d in detections],
        )

    @staticmethod
    def _validate_detections(detections) -> List[Detection]:
        if detections is None:
            return []
        if not isinstance(detections, (list, tuple)):
            detections = list(detections)
        return detections

    def update(self, detections, image=None) -> List[STrack]:
        """Process detections and update tracks."""
        detections = self._validate_detections(detections)
        self.frame_id += 1

        if not detections:
            for track in self.tracked_tracks:
                if track.state != "Lost":
                    track.mark_lost()
            self.lost_tracks.extend([t for t in self.tracked_tracks if t.state == "Lost"])
            self.tracked_tracks = []
            self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id <= self.track_buffer]
            return []

        dets_high = [d for d in detections if d.confidence >= self.track_thresh]
        dets_low  = [d for d in detections if d.confidence < self.track_thresh]

        dets_high = [
            d for d in dets_high
            if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= self.min_box_area
        ]

        activated, refined, lost, removed = [], [], [], []

        unmatched_det_indices: np.ndarray = np.arange(len(dets_high))
        if len(self.tracked_tracks) > 0:
            cost = self._iou_distance(self.tracked_tracks, dets_high)
            matches, u_track, u_det = linear_assignment(cost, 1.0 - self.match_thresh)
            unmatched_det_indices = u_det

            for itrack, idet in matches:
                track = self.tracked_tracks[itrack]
                det = dets_high[idet]
                if track.state == "Tracked":
                    track.update(det, self.frame_id)
                    activated.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined.append(track)

            for itrack in u_track:
                track = self.tracked_tracks[itrack]
                if track.state != "Lost":
                    track.mark_lost()
                    lost.append(track)

        if len(self.lost_tracks) > 0:
            cost = self._iou_distance(self.lost_tracks, dets_low)
            low_matches, _, _ = linear_assignment(cost, 1.0 - self.match_thresh)
            for itrack, idet in low_matches:
                track = self.lost_tracks[itrack]
                det = dets_low[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                refined.append(track)

        for idet in unmatched_det_indices:
            det = dets_high[idet]
            track = STrack(
                track_id=self.next_id, bbox=det.bbox,
                score=det.confidence, class_id=det.class_id,
                class_name=det.class_name,
            )
            track.activate(self.frame_id)
            activated.append(track)
            self.next_id += 1

        self.tracked_tracks = [t for t in activated if t.state == "Tracked"]
        self.tracked_tracks.extend(refined)

        self.lost_tracks = [t for t in self.lost_tracks if t.state == "Lost"]
        self.lost_tracks.extend(lost)
        self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id <= self.track_buffer]

        newly_removed = [t for t in self.lost_tracks if self.frame_id - t.frame_id > self.track_buffer]
        for t in newly_removed:
            t.mark_removed()
        self.removed_tracks.extend(newly_removed)
        self.lost_tracks = [t for t in self.lost_tracks if t.state != "Removed"]

        return [t for t in self.tracked_tracks if t.is_activated]
