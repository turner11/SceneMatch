from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""
    video_id: str
    frame_index: int
    timestamp: float
    keypoints: tuple[cv2.KeyPoint, ...]
    features: np.array


@dataclass
class FrameMatch:
    """Represents a match between two frames."""
    frame: FrameMetadata
    frame_reference: FrameMetadata
    distance_score: float
    notes: str = ''