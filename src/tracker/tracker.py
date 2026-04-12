"""
DeepSORT-style multi-object tracker with two-stage matching.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .track import Track

logger = logging.getLogger(__name__)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        a: Array of shape (M, 4) with boxes [x1, y1, x2, y2]
        b: Array of shape (N, 4) with boxes [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (M, N)
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a[:, None] + area_b[None, :] - inter_area + 1e-6
    return inter_area / union


class DeepSortTracker:
    """
    Multi-object tracker using:
    1. Appearance matching on confirmed tracks
    2. IoU matching on tentative and unmatched tracks
    """

    def __init__(
        self,
        appearance_threshold: float = 0.45,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 5,
        min_confirmed_age: int = 3,
    ):
        self.tracks: List[Track] = []
        self.appearance_threshold = appearance_threshold
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_confirmed_age = min_confirmed_age

    def update(
        self,
        boxes: np.ndarray,
        features: np.ndarray,
        labels: List[str],
    ) -> List[Track]:
        """
        Run one update step of the tracker.

        Args:
            boxes: Array of shape (N, 4)
            features: Array of shape (N, D)
            labels: List of class labels of length N

        Returns:
            List of active tracks
        """
        for track in self.tracks:
            track.predict()

        if len(boxes) == 0:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
            return self.tracks

        if not self.tracks:
            for box, feature, label in zip(boxes, features, labels):
                self.tracks.append(
                    Track(
                        bbox=box,
                        feature=feature,
                        label=label,
                        min_confirmed_age=self.min_confirmed_age,
                        max_missed_frames=self.max_missed_frames,
                    )
                )
            return self.tracks

        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [
            i for i, t in enumerate(self.tracks)
            if not t.is_confirmed() and not t.is_deleted()
        ]

        matches, unmatched_confirmed, unmatched_detections = self._appearance_match(
            confirmed_indices,
            features,
            boxes,
            labels,
        )

        iou_candidates = tentative_indices + unmatched_confirmed

        extra_matches, unmatched_tracks, unmatched_detections = self._iou_match(
            iou_candidates,
            boxes,
            unmatched_detections,
            labels,
        )

        matches.extend(extra_matches)

        matched_track_indices = {track_idx for track_idx, _ in matches}

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                bbox=boxes[det_idx],
                feature=features[det_idx],
                label=labels[det_idx],
            )

        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track.mark_missed()

        for det_idx in unmatched_detections:
            self.tracks.append(
                Track(
                    bbox=boxes[det_idx],
                    feature=features[det_idx],
                    label=labels[det_idx],
                    min_confirmed_age=self.min_confirmed_age,
                    max_missed_frames=self.max_missed_frames,
                )
            )

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        logger.debug(
            "Tracking update | active=%d | confirmed=%d | tentative=%d",
            len(self.tracks),
            sum(t.is_confirmed() for t in self.tracks),
            sum(not t.is_confirmed() for t in self.tracks),
        )

        return self.tracks

    def _appearance_match(
        self,
        track_indices: List[int],
        det_features: np.ndarray,
        det_boxes: np.ndarray,
        det_labels: List[str],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match confirmed tracks to detections using cosine distance with two gates:
        1. Class gate  — cross-class matches are forbidden (cost = inf).
        2. Spatial gate — if the predicted track box has zero IoU with the
           detection box, the pair is geometrically implausible and is also
           blocked. This prevents appearance-similar objects far apart in the
           frame from swapping IDs.
        """
        num_detections = len(det_features)

        if not track_indices or num_detections == 0:
            return [], list(track_indices), list(range(num_detections))

        track_features = np.array([self.tracks[i].feature for i in track_indices])
        cost_matrix = cdist(track_features, det_features, metric="cosine")

        # --- spatial center-distance gate ---
        # Small objects (bags, backpacks, suitcases) move less and are closer
        # together, so use a tighter radius to avoid cross-bag ID swaps.
        # Larger objects (people, vehicles) use a wider radius.
        _SMALL_CLASSES = {"handbag", "backpack", "suitcase"}
        _DIST_FACTOR_SMALL = 1.5
        _DIST_FACTOR_LARGE = 2.5

        track_boxes_arr = np.array([self.tracks[i].get_bbox() for i in track_indices])
        det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
        det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2

        # --- class gate + spatial gate ---
        for row, ti in enumerate(track_indices):
            track_label = self.tracks[ti].label
            tx1, ty1, tx2, ty2 = track_boxes_arr[row]
            t_cx = (tx1 + tx2) / 2
            t_cy = (ty1 + ty2) / 2
            max_dim = max(tx2 - tx1, ty2 - ty1, 1.0)
            factor = _DIST_FACTOR_SMALL if track_label in _SMALL_CLASSES else _DIST_FACTOR_LARGE
            for col in range(num_detections):
                dist = np.hypot(det_cx[col] - t_cx, det_cy[col] - t_cy)
                if det_labels[col] != track_label or dist > factor * max_dim:
                    cost_matrix[row, col] = np.inf

        # linear_sum_assignment cannot handle all-inf matrices; replace inf with
        # a large finite sentinel so the solver always succeeds.  The threshold
        # check below will reject any pair whose original cost was inf.
        solve_matrix = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
        row_indices, col_indices = linear_sum_assignment(solve_matrix)

        matches = []
        matched_rows = set()
        matched_cols = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.appearance_threshold:
                matches.append((track_indices[row], col))
                matched_rows.add(row)
                matched_cols.add(col)

        unmatched_tracks = [
            track_indices[row]
            for row in range(len(track_indices))
            if row not in matched_rows
        ]
        unmatched_detections = [
            col for col in range(num_detections)
            if col not in matched_cols
        ]

        return matches, unmatched_tracks, unmatched_detections

    def _iou_match(
        self,
        track_indices: List[int],
        det_boxes: np.ndarray,
        det_indices: List[int],
        det_labels: List[str],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks to remaining detections using IoU with class gate.
        Cross-class pairs are forbidden regardless of IoU score.
        """
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)

        track_boxes = np.array([self.tracks[i].get_bbox() for i in track_indices])
        candidate_boxes = det_boxes[det_indices]

        iou_matrix = _iou_matrix(track_boxes, candidate_boxes)
        cost_matrix = 1.0 - iou_matrix

        # Class gate: forbid cross-class matches
        for row, ti in enumerate(track_indices):
            track_label = self.tracks[ti].label
            for col, di in enumerate(det_indices):
                if det_labels[di] != track_label:
                    cost_matrix[row, col] = np.inf

        solve_matrix = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
        row_indices, col_indices = linear_sum_assignment(solve_matrix)

        matches = []
        matched_rows = set()
        matched_cols = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                matches.append((track_indices[row], det_indices[col]))
                matched_rows.add(row)
                matched_cols.add(col)

        unmatched_tracks = [
            track_indices[row]
            for row in range(len(track_indices))
            if row not in matched_rows
        ]
        unmatched_detections = [
            det_indices[col]
            for col in range(len(det_indices))
            if col not in matched_cols
        ]

        return matches, unmatched_tracks, unmatched_detections