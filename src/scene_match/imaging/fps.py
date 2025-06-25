import datetime


# Based on imutils:
# https://github.com/PyImageSearch/imutils/blob/9f740a53bcc2ed7eba2558afed8b4c17fd8a1d4c/imutils/video/fps.py


class FPS:
    def __init__(self, capacity=1000):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._total_count = 0
        self.capacity = capacity
        self.time_stamps = []

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        self.time_stamps[:] = [self._start]
        self._total_count = 0
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
        self.time_stamps.append(self._end)

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self.time_stamps.append(datetime.datetime.now())
        self.time_stamps = self.time_stamps[-self.capacity:]
        self._total_count += 1

    def elapsed(self):
        # return the total number of seconds between the start and end interval
        if not self.time_stamps:
            return 0

        end = self.time_stamps[-1]
        start = self.time_stamps[0]
        return (end - start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        elapsed = self.elapsed()
        if not elapsed:
            return 0
        return len(self.time_stamps) / self.elapsed()

    def fps_single_frame_frames(self):
        return self.fps_for_frames(1)

    def fps_for_frames(self, n_frames):
        # compute the (approximate) frames per second for the last n_frames
        if not self.time_stamps:
            return 0

        frames = self.time_stamps[-(n_frames + 1):]
        start, end = frames[0], frames[-1]
        elapsed = (end - start).total_seconds()
        return len(frames) / elapsed if elapsed else 0

    def time_since_last(self, time_stamp=None):
        if not self.time_stamps:
            return 0

        time_stamp = time_stamp or datetime.datetime.now()
        last_time = self.time_stamps[-1]
        diff = time_stamp - last_time
        return diff

    def total_fps(self):
        # compute the (approximate) frames per second regardless of capacity
        end = self._end or datetime.datetime.now()
        elapsed = (end - self._start).total_seconds()
        return self._total_count / elapsed
