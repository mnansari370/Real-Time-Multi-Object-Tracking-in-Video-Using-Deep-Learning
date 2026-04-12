"""
Main tracking pipeline.
"""

import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio
import numpy as np
from ultralytics import YOLO

from src.config import Config
from src.tracker import DeepSortTracker, FeatureExtractor, Track
from src.utils import (
    TrackingCSVWriter,
    draw_overlay_stats,
    draw_tracks,
    generate_all_analytics,
    save_heatmap,
    update_heatmap,
    write_summary,
)

logger = logging.getLogger(__name__)


def run_tracking_pipeline(cfg: Config, video_path: str) -> None:
    """
    Run the full multi-object tracking pipeline on a video file.
    """
    Track.reset_id_counter()
    Path("outputs").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Video opened | path=%s | resolution=%dx%d | fps=%.1f | frames=%d",
        video_path,
        width,
        height,
        fps,
        total_frames,
    )

    detector = YOLO(cfg.yolo_model)
    extractor = FeatureExtractor(cfg.crop_size, cfg.feature_dim)
    tracker = DeepSortTracker(
        appearance_threshold=cfg.appearance_threshold,
        iou_threshold=cfg.iou_threshold,
        max_missed_frames=cfg.max_missed_frames,
        min_confirmed_age=cfg.min_confirmed_age,
    )

    video_writer = imageio.get_writer(cfg.output_video, fps=fps)
    csv_writer = TrackingCSVWriter(cfg.csv_path)
    heatmap = np.zeros((height, width), dtype=np.uint8)

    target_class_set = set(cfg.target_classes)
    frame_id = 0
    unique_track_labels = {}
    start_time = time.time()

    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(
            frame,
            conf=cfg.confidence_threshold,
            iou=cfg.nms_iou_threshold,   # higher = keep more overlapping boxes
            max_det=cfg.max_detections,
            verbose=False,
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy()
            raw_classes = results.boxes.cls.cpu().numpy().astype(int)

            keep_indices = [
                idx
                for idx, class_id in enumerate(raw_classes)
                if detector.names[class_id] in target_class_set
            ]

            boxes = raw_boxes[keep_indices]
            labels = [detector.names[raw_classes[idx]] for idx in keep_indices]
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = []

        if len(boxes) > 0:
            features = extractor.extract(frame, boxes)
            tracks = tracker.update(boxes, features, labels)
        else:
            tracks = tracker.update(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, cfg.feature_dim), dtype=np.float32),
                [],
            )

        for track in tracks:
            if track.is_confirmed() and track.track_id not in unique_track_labels:
                unique_track_labels[track.track_id] = track.label

        counts = Counter(track.label for track in tracks if track.is_confirmed())
        elapsed = time.time() - frame_start
        display_fps = 1.0 / max(elapsed, 1e-6)

        annotated_frame = draw_tracks(
            frame,
            tracks,
            trajectory_length=cfg.trajectory_length,
            box_thickness=cfg.box_thickness,
            font_scale=cfg.font_scale,
            show_trajectory=cfg.show_trajectory,
            confirmed_only=cfg.confirmed_only,
        )
        annotated_frame = draw_overlay_stats(
            annotated_frame,
            counts,
            display_fps,
            frame_id,
        )

        update_heatmap(heatmap, tracks)
        csv_writer.write_frame(frame_id, tracks)
        video_writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        frame_id += 1
        if frame_id % 100 == 0:
            logger.info("Processed %d / %d frames", frame_id, total_frames)

    cap.release()
    video_writer.close()
    csv_writer.close()

    save_heatmap(heatmap, cfg.heatmap_path)

    unique_counts = Counter(unique_track_labels.values())
    duration_seconds = time.time() - start_time

    logger.info("Generating analytics and charts...")
    generate_all_analytics(
        csv_path=cfg.csv_path,
        frame_size=(width, height),
        output_dir="outputs",
        unique_counts=unique_counts,
        total_frames=frame_id,
        duration_seconds=duration_seconds,
    )

    logger.info("Tracking complete")
    logger.info("Output video          : %s", cfg.output_video)
    logger.info("Motion heatmap        : %s", cfg.heatmap_path)
    logger.info("CSV results           : %s", cfg.csv_path)
    logger.info("Summary report        : outputs/summary_report.txt")
    logger.info("Count over time       : outputs/count_over_time.png")
    logger.info("Track lifetimes       : outputs/track_lifetimes.png")
    logger.info("Trajectory map        : outputs/trajectory_map.png")
    logger.info("Unique counts bar     : outputs/unique_counts_bar.png")
    logger.info("Class distribution    : outputs/class_distribution_pie.png")
    logger.info("Per-class heatmaps    : outputs/heatmap_<class>.png")