"""
I/O helpers for writing tracking results to disk.
"""

import csv
import logging
from pathlib import Path
from typing import List

from src.tracker.track import Track

logger = logging.getLogger(__name__)


class TrackingCSVWriter:
    """
    Incrementally write confirmed tracking results to a CSV file.
    """

    FIELDS = ("frame_id", "track_id", "label", "x1", "y1", "x2", "y2")

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write_frame(self, frame_id: int, tracks: List[Track]) -> None:
        """
        Write one row per confirmed track for the current frame.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.get_bbox()
            self._writer.writerow({
                "frame_id": frame_id,
                "track_id": track.track_id,
                "label": track.label,
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            })

    def close(self) -> None:
        self._file.close()
        logger.info("Tracking CSV saved to %s", self.path)