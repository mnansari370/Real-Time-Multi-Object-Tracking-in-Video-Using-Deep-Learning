"""
Visualisation utilities for drawing tracked objects and on-frame stats.
"""

from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from src.tracker.track import Track


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """
    Return a deterministic BGR colour for a given track ID.
    """
    rng = np.random.default_rng(track_id + 42)
    r, g, b = rng.integers(80, 230, size=3)
    return int(b), int(g), int(r)


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    trajectory_length: int = 30,
    box_thickness: int = 2,
    font_scale: float = 0.55,
    show_trajectory: bool = True,
    confirmed_only: bool = True,
) -> np.ndarray:
    """
    Draw tracked bounding boxes, labels, and trajectories on a frame copy.
    """
    output = frame.copy()

    for track in tracks:
        if confirmed_only and not track.is_confirmed():
            continue

        color = get_track_color(track.track_id)
        x1, y1, x2, y2 = map(int, track.get_bbox())

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{track.label.title()} #{track.track_id}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )

        badge_top = max(0, y1 - text_h - baseline - 6)
        badge_bottom = max(0, y1)

        cv2.rectangle(
            output,
            (x1, badge_top),
            (x1 + text_w + 6, badge_bottom),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 3, badge_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if show_trajectory and len(track.history) > 1:
            trail = track.history[-trajectory_length:]
            # Smooth the trail with a 3-point moving average to reduce jitter
            if len(trail) >= 3:
                smoothed = []
                for i in range(len(trail)):
                    lo = max(0, i - 1)
                    hi = min(len(trail) - 1, i + 1)
                    sx = int(sum(trail[j][0] for j in range(lo, hi + 1)) / (hi - lo + 1))
                    sy = int(sum(trail[j][1] for j in range(lo, hi + 1)) / (hi - lo + 1))
                    smoothed.append((sx, sy))
            else:
                smoothed = list(trail)
            for i in range(1, len(smoothed)):
                alpha = i / len(smoothed)
                faded_color = tuple(int(v * alpha) for v in color)
                cv2.line(output, smoothed[i - 1], smoothed[i], faded_color, 2, cv2.LINE_AA)

    return output


def draw_overlay_stats(
    frame: np.ndarray,
    counts: Counter,
    fps: float,
    frame_id: int,
) -> np.ndarray:
    """
    Draw a semi-transparent stats panel on the frame.
    """
    lines = [f"Frame : {frame_id}", f"FPS   : {fps:.1f}", ""] + [
        f"  {label.title():<16}: {count}"
        for label, count in sorted(counts.items())
    ]

    panel_w = 220
    panel_h = len(lines) * 20 + 14

    output = frame.copy()
    overlay = output.copy()

    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    for i, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return output