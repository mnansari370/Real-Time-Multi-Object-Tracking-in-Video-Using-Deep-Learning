"""
Reporting, heatmap, and analytics utilities.
Produces visualisations suitable for a Computer Vision presentation.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.tracker.track import Track

logger = logging.getLogger(__name__)

# Consistent colour palette per class
_CLASS_COLORS = {
    "person":       "#2196F3",
    "handbag":      "#FF9800",
    "backpack":     "#4CAF50",
    "suitcase":     "#9C27B0",
    "car":          "#F44336",
    "truck":        "#795548",
    "motorcycle":   "#00BCD4",
    "traffic light":"#FFEB3B",
    "stop sign":    "#E91E63",
}
_DEFAULT_COLOR = "#607D8B"


# ---------------------------------------------------------------------------
# Live heatmap helpers (called per-frame during tracking)
# ---------------------------------------------------------------------------

def update_heatmap(
    heatmap: np.ndarray,
    tracks: List[Track],
    radius: int = 8,
) -> None:
    """Accumulate confirmed track centres on the motion density heatmap."""
    for track in tracks:
        if track.is_confirmed():
            cx, cy = track.get_center()
            if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                cv2.circle(heatmap, (cx, cy), radius, 255, -1)


def save_heatmap(heatmap: np.ndarray, path: str) -> None:
    """Blur and colourise the motion density heatmap, then save."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blurred = cv2.GaussianBlur(heatmap, (31, 31), 0)
    colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    cv2.imwrite(path, colored)
    logger.info("Motion heatmap saved to %s", path)


# ---------------------------------------------------------------------------
# Post-processing analytics (called once after the pipeline finishes)
# ---------------------------------------------------------------------------

def generate_all_analytics(
    csv_path: str,
    frame_size: Tuple[int, int],
    output_dir: str,
    unique_counts: Counter,
    total_frames: int,
    duration_seconds: float,
) -> None:
    """
    Generate all analytics outputs from the tracking CSV.
    Calls every individual plot/report function.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("CSV is empty — skipping analytics")
        return

    write_summary(
        str(out / "summary_report.txt"),
        unique_counts,
        total_frames,
        duration_seconds,
    )
    plot_count_over_time(df, str(out / "count_over_time.png"))
    plot_track_lifetimes(df, str(out / "track_lifetimes.png"))
    plot_trajectory_map(df, frame_size, str(out / "trajectory_map.png"))
    plot_unique_counts_bar(unique_counts, str(out / "unique_counts_bar.png"))
    save_per_class_heatmaps(df, frame_size, str(out / "heatmap"))
    plot_class_distribution_pie(unique_counts, str(out / "class_distribution_pie.png"))


def write_summary(
    path: str,
    unique_counts: Counter,
    total_frames: int,
    duration_seconds: float,
) -> None:
    """Write a plain-text session summary report."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    average_fps = total_frames / max(duration_seconds, 1e-6)
    total_unique = sum(unique_counts.values())

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 46 + "\n")
        f.write("   Multi-Object Tracking - Summary Report\n")
        f.write("=" * 46 + "\n\n")
        f.write(f"  Frames processed  : {total_frames}\n")
        f.write(f"  Duration          : {duration_seconds:.2f} s\n")
        f.write(f"  Average FPS       : {average_fps:.1f}\n")
        f.write(f"  Total unique IDs  : {total_unique}\n\n")
        f.write("  Unique tracks by class:\n")
        for label, count in sorted(unique_counts.items(), key=lambda x: -x[1]):
            f.write(f"    {label:<22}: {count}\n")
        f.write("\n" + "=" * 46 + "\n")

    logger.info("Summary report saved to %s", path)


def plot_count_over_time(df: pd.DataFrame, path: str) -> None:
    """
    Line chart: number of tracked objects per class across frames.
    Useful for showing crowd density changes over time.
    """
    classes = df["label"].unique()
    counts_by_frame = (
        df.groupby(["frame_id", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(df["frame_id"].max() + 1), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for cls in sorted(classes):
        if cls in counts_by_frame.columns:
            color = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)
            ax.plot(
                counts_by_frame.index,
                counts_by_frame[cls],
                label=cls.title(),
                color=color,
                linewidth=1.8,
                alpha=0.85,
            )

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Active Tracks", fontsize=12)
    ax.set_title("Tracked Object Count Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Count-over-time chart saved to %s", path)


def plot_track_lifetimes(df: pd.DataFrame, path: str) -> None:
    """
    Histogram: distribution of how long (in frames) each track survived.
    Longer tracks = more stable tracking.
    """
    lifetimes = (
        df.groupby(["track_id", "label"])["frame_id"]
        .apply(lambda x: x.max() - x.min() + 1)
        .reset_index()
    )
    lifetimes.columns = ["track_id", "label", "lifetime"]

    classes = lifetimes["label"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    for cls in sorted(classes):
        data = lifetimes[lifetimes["label"] == cls]["lifetime"]
        color = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)
        ax.hist(data, bins=20, alpha=0.65, label=cls.title(), color=color, edgecolor="white")

    ax.set_xlabel("Track Lifetime (frames)", fontsize=12)
    ax.set_ylabel("Number of Tracks", fontsize=12)
    ax.set_title("Track Lifetime Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Track lifetime histogram saved to %s", path)


def plot_trajectory_map(
    df: pd.DataFrame,
    frame_size: Tuple[int, int],
    path: str,
) -> None:
    """
    Overlay all track trajectories on a dark canvas.
    Each unique track gets its own colour. Shows all movement paths.
    """
    width, height = frame_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    rng = np.random.default_rng(42)
    track_ids = df["track_id"].unique()

    for tid in track_ids:
        tdf = df[df["track_id"] == tid].sort_values("frame_id")
        cx = ((tdf["x1"] + tdf["x2"]) / 2).astype(int).tolist()
        cy = ((tdf["y1"] + tdf["y2"]) / 2).astype(int).tolist()
        if len(cx) < 2:
            continue
        color = tuple(int(v) for v in rng.integers(80, 230, 3))
        pts = list(zip(cx, cy))
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            faded = tuple(int(v * alpha) for v in color)
            cv2.line(canvas, pts[i - 1], pts[i], faded, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 4, color, -1)

    # Convert BGR → RGB for matplotlib
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("All Track Trajectories", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory map saved to %s", path)


def plot_unique_counts_bar(unique_counts: Counter, path: str) -> None:
    """
    Horizontal bar chart of total unique objects detected per class.
    """
    if not unique_counts:
        return

    labels = [l.title() for l in sorted(unique_counts, key=lambda x: -unique_counts[x])]
    values = [unique_counts[l.lower()] for l in labels]
    colors = [_CLASS_COLORS.get(l.lower(), _DEFAULT_COLOR) for l in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Unique Tracks", fontsize=12)
    ax.set_title("Total Unique Objects Detected", fontsize=14, fontweight="bold")
    ax.set_xlim(right=max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Unique counts bar chart saved to %s", path)


def save_per_class_heatmaps(
    df: pd.DataFrame,
    frame_size: Tuple[int, int],
    path_prefix: str,
) -> None:
    """
    Save a separate density heatmap for each tracked class.
    Shows where each class of object spent the most time.
    """
    width, height = frame_size
    classes = df["label"].unique()

    for cls in classes:
        cdf = df[df["label"] == cls]
        heatmap = np.zeros((height, width), dtype=np.float32)

        for _, row in cdf.iterrows():
            cx = int((row["x1"] + row["x2"]) / 2)
            cy = int((row["y1"] + row["y2"]) / 2)
            if 0 <= cy < height and 0 <= cx < width:
                cv2.circle(heatmap, (cx, cy), 12, 255, -1)

        blurred = cv2.GaussianBlur(heatmap, (31, 31), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        out_path = f"{path_prefix}_{cls.replace(' ', '_')}.png"
        cv2.imwrite(out_path, colored)
        logger.info("Per-class heatmap saved to %s", out_path)


def plot_class_distribution_pie(unique_counts: Counter, path: str) -> None:
    """
    Pie chart of the proportion of unique objects per class.
    """
    if not unique_counts:
        return

    labels = [l.title() for l in unique_counts]
    sizes  = list(unique_counts.values())
    colors = [_CLASS_COLORS.get(l.lower(), _DEFAULT_COLOR) for l in unique_counts]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title("Object Class Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Class distribution pie chart saved to %s", path)
