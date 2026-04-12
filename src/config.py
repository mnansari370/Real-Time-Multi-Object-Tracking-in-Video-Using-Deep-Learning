"""
Central configuration for the Multi-Object Tracker.
All tunable hyperparameters are defined here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Detection
    yolo_model: str = "yolov8l.pt"      # large model: best accuracy in crowds
    confidence_threshold: float = 0.25  # lower = catch partial/small detections
    nms_iou_threshold: float = 0.75     # high NMS IoU = keep overlapping people separate
    max_detections: int = 300           # allow many objects per frame
    target_classes: List[str] = field(default_factory=lambda: [
        "person",
        "backpack",
        "handbag",
        "suitcase",
        "car",
        "truck",
        "motorcycle",
        "traffic light",
        "stop sign",
    ])

    # Feature extraction
    crop_size: Tuple[int, int] = (128, 128)
    feature_dim: int = 128

    # Tracking
    appearance_threshold: float = 0.40  # cosine distance gate
    iou_threshold: float = 0.25         # lenient for crowded scenes
    max_missed_frames: int = 15         # survive longer occlusions
    min_confirmed_age: int = 2          # show tracks after 2 hits (was 3)

    # Visualisation
    trajectory_length: int = 30
    box_thickness: int = 2
    font_scale: float = 0.55
    show_trajectory: bool = True
    confirmed_only: bool = True

    # Paths
    output_video: str = "outputs/output_tracked.mp4"
    heatmap_path: str = "outputs/motion_heatmap.png"
    csv_path: str = "outputs/tracking_results.csv"
    stats_path: str = "outputs/summary_report.txt"
