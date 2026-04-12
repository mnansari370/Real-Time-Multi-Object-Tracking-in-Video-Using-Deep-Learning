"""
Appearance feature extractor for Re-Identification (ReID).

Uses a pretrained MobileNetV2 backbone (torchvision) followed by a
128-dim projection. Features are L2-normalised for stable cosine distance.
Runs on GPU automatically if available (same device as YOLO/PyTorch).
"""

import logging
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts compact appearance embeddings from bounding-box crops.

    Architecture:
        MobileNetV2 (ImageNet weights, no classifier)
        -> AdaptiveAvgPool2d  (global average pooling)
        -> Dense(feature_dim, ReLU)
        -> L2 normalisation

    Args:
        crop_size: Target (width, height) for resizing crops.
        feature_dim: Output embedding size.
    """

    # ImageNet normalisation constants
    _MEAN = torch.tensor([0.485, 0.456, 0.406])
    _STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, crop_size: Tuple[int, int] = (128, 128), feature_dim: int = 128):
        self.crop_size = crop_size   # (W, H)
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

        logger.info(
            "FeatureExtractor initialised | device=%s | crop_size=%s | feature_dim=%d",
            self.device,
            self.crop_size,
            self.feature_dim,
        )

    def _build_model(self) -> None:
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Drop the original classifier; keep only the feature layers
        backbone.classifier = nn.Identity()
        # MobileNetV2 last conv outputs (1280,) after global avg pool
        self.model = nn.Sequential(
            backbone,
            nn.Linear(1280, self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.model.eval()
        self.model.to(self.device)

        # Move normalisation tensors to the same device
        self._mean = self._MEAN.to(self.device).view(1, 3, 1, 1)
        self._std  = self._STD.to(self.device).view(1, 3, 1, 1)

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Extract appearance features for all detection boxes in a frame.

        Args:
            frame: Input image in BGR format (H, W, 3).
            boxes: Detection boxes of shape (N, 4) in [x1, y1, x2, y2] format.

        Returns:
            Array of shape (N, feature_dim) with L2-normalised embeddings.
        """
        if len(boxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            # Crops smaller than 10×10 px produce garbage features — use zeros
            # so the EMA smoothing in Track.update() gradually dilutes the noise.
            if crop.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
                crop = np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, self.crop_size)

            # BGR → RGB, HWC → CHW, normalise to [0, 1]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crops.append(crop)

        # Stack into batch tensor and move to device
        batch = torch.stack(crops).to(self.device)       # (N, 3, H, W)
        batch = (batch - self._mean) / self._std         # ImageNet normalise

        with torch.no_grad():
            features = self.model(batch)                 # (N, feature_dim)

        # L2 normalise
        norms = features.norm(dim=1, keepdim=True).clamp(min=1e-6)
        features = (features / norms).cpu().numpy().astype(np.float32)

        return features
