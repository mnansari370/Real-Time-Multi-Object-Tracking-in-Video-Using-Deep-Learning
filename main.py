"""
Command-line entry point for the Multi-Object Tracker.
"""

import argparse
import logging

from src.config import Config
from src.pipeline import run_tracking_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + DeepSORT-style Multi-Object Tracker"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output video path.",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory trail rendering.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of class names to track.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    cfg.yolo_model = args.model
    cfg.confidence_threshold = args.conf
    cfg.show_trajectory = not args.no_trajectory

    if args.classes:
        cfg.target_classes = args.classes

    if args.output:
        cfg.output_video = args.output

    run_tracking_pipeline(cfg, args.video)


if __name__ == "__main__":
    main()