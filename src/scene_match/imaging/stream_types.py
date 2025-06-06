from dataclasses import dataclass

import numpy as np


@dataclass
class FrameCapture:
    success: bool
    image: np.array
    fps: float
    stream_fps: float
    n: int  # image number
    error_message: str = ''
    frames_dropped: int = 0
    timestamp: float = 0  # seconds


@dataclass
class StreamParams:
    start_frame: int = -1
    sample_interval: int = 1
    visualize: bool = False


@dataclass
class IndexParams:
    n_features: int = 500
