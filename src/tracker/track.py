"""
Single-object track with Kalman filter motion model and lifecycle management.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

# EMA weight for appearance feature blending.
# Lower = faster adaptation (better for small/variable objects like bags).
_FEATURE_EMA_ALPHA = 0.75


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED = auto()


class Track:
    """
    Represents a single tracked object across frames.

    State vector:
        [cx, cy, w, h, vx, vy, vs]
    where vs is the shared scale velocity applied to both w and h each step,
    preserving aspect ratio during free prediction.
    """

    _id_counter: int = 0

    def __init__(
        self,
        bbox: np.ndarray,
        feature: np.ndarray,
        label: str,
        min_confirmed_age: int = 3,
        max_missed_frames: int = 5,
    ):
        self.track_id = Track._id_counter
        Track._id_counter += 1

        self.label = label
        self.state = TrackState.TENTATIVE
        self.age = 1
        self.missed = 0
        self.feature = feature.copy()
        self.bbox = np.asarray(bbox, dtype=np.float32)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        self.history: List[Tuple[int, int]] = [(int(cx), int(cy))]

        self._min_confirmed_age = min_confirmed_age
        self._max_missed_frames = max_missed_frames

        self.kf = self._init_kalman(bbox)

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0

    def _init_kalman(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.x = np.array([cx, cy, w, h, 0, 0, 0], dtype=float).reshape(7, 1)

        # vs (index 6) applied to both w (row 2) and h (row 3) — aspect ratio preserved.
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # cx  += vx
            [0, 1, 0, 0, 0, 1, 0],  # cy  += vy
            [0, 0, 1, 0, 0, 0, 1],  # w   += vs
            [0, 0, 0, 1, 0, 0, 1],  # h   += vs
            [0, 0, 0, 0, 1, 0, 0],  # vx const
            [0, 0, 0, 0, 0, 1, 0],  # vy const
            [0, 0, 0, 0, 0, 0, 1],  # vs const
        ])

        kf.H = np.eye(4, 7)

        kf.P *= 10.0
        kf.R *= 1.0
        # Larger Q (was 0.01) so the filter trusts observations more and adapts
        # faster to real movement, especially for fast-moving people in crowds.
        kf.Q[4:, 4:] *= 0.5   # velocity/scale process noise
        kf.Q[:4, :4] *= 0.1   # position process noise

        return kf

    def predict(self):
        self.kf.predict()

    def update(self, bbox, feature, label):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        self.kf.update(np.array([cx, cy, w, h]))

        # Use Kalman-smoothed center (not raw detection) for a smoother trajectory.
        smooth_cx = float(self.kf.x[0])
        smooth_cy = float(self.kf.x[1])

        # EMA feature smoothing: blend old appearance with new detection crop.
        blended = _FEATURE_EMA_ALPHA * self.feature + (1.0 - _FEATURE_EMA_ALPHA) * feature
        norm = np.linalg.norm(blended) + 1e-6
        self.feature = (blended / norm).astype(np.float32)

        self.label = label
        self.bbox = np.asarray(bbox, dtype=np.float32)

        self.age += 1
        self.missed = 0

        self.history.append((int(smooth_cx), int(smooth_cy)))

        if self.age >= self._min_confirmed_age:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        self.missed += 1
        # Only extend trajectory with Kalman prediction for short occlusions
        # (missed <= 3). Beyond that the velocity estimate drifts and creates
        # ghost path segments in the wrong direction.
        if self.missed <= 3:
            pred_cx = float(self.kf.x[0])
            pred_cy = float(self.kf.x[1])
            self.history.append((int(pred_cx), int(pred_cy)))

        if self.missed > self._max_missed_frames:
            self.state = TrackState.DELETED

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELETED

    def get_bbox(self):
        cx, cy, w, h = self.kf.x[:4].flatten()
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def get_center(self):
        x1, y1, x2, y2 = self.get_bbox()
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
