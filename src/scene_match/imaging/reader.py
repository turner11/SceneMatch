from pathlib import Path
import cv2
import numpy as np
from .fps import FPS
from .stream_types import FrameCapture

empty_image = np.ndarray((0, 0, 3))


def get_stream(source, start_frame=0, fps_window=4, sample_interval=1, drop_frames=False):
    if source != 0 and not source:
        # 0 is a valid source for webcam1
        raise ValueError('Please provide a valid source for the stream.')

    if isinstance(source, Path):
        source = str(source)

    # noinspection PyTypeChecker
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        error_message = f'Could not open video source for {source}'
        raise ValueError(error_message)

    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = start_frame if (start_frame and start_frame >= 0) else 0
    i_frame = start_frame if start_frame > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    fps_counter = FPS(capacity=fps_window).start()
    try:
        while True:

            if i_frame % sample_interval == 0:
                ret, frame = cap.read()
            else:
                frame = None
                ret = cap.grab()

            if not ret:
                break

            if frame is None:
                i_frame += 1
                continue

            is_read_success, image = cap.read()
            processing_time = fps_counter.time_since_last()

            fps_counter.update()
            fps = fps_counter.fps()

            time_behind: float = max(processing_time.total_seconds(), 0)
            if time_behind and drop_frames:
                frames_to_drop = int(time_behind * stream_fps)
                _ = [cap.grab() for _ in range(frames_to_drop)]
                # cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + frames_to_drop)
                i_frame += frames_to_drop
            else:
                frames_to_drop = 0

            error_message = ''
            if not is_read_success:
                image = empty_image
                error_message = 'Could not read image from video source'

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            i_frame += 1
            payload = FrameCapture(image=image, fps=fps, stream_fps=stream_fps, n=i_frame, success=is_read_success,
                                   error_message=error_message, frames_dropped=frames_to_drop, timestamp=timestamp)

            yield payload
            if not payload.success:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
