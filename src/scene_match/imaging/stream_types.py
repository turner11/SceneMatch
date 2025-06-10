from dataclasses import dataclass

import numpy as np


@dataclass
class FrameCapture:
    success: bool  # was the frame captured successfully?
    image: np.array  # the captured image
    fps: float  # processing fps
    stream_fps: float  # stream fps
    n: int  # the frame number in the stream
    error_message: str = ''  # error message if the frame was not captured successfully
    frames_dropped: int = 0  # the number of frames dropped due to processing time being longer than stream fps
    timestamp: float = 0  # timestamp of the frame capture, in seconds since the beginning of the stream


@dataclass
class StreamParams:
    start_frame: int = -1  # start frame for the stream, -1 means start from the beginning
    sample_interval: int = 1  # sample every n-th frame, 1 means every frame
    allow_drop_frames = False  # if True, will drop frames if processing is slower than the stream fps


@dataclass
class DrawParams:
    visualize: bool = False  # should visualize the image?
    show_matches: bool = True  # should show matches on the image
    show_keypoints: bool = True  # should show keypoints on the image
    n_features: int = 10  # number of features to draw on the image


@dataclass
class IndexParams:
    n_features: int = 500  # number of features to index
