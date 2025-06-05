"""
Frame Matcher module for finding similar frames between two drone videos.
"""
from pathlib import Path

import cv2
import numpy as np
import faiss
from dataclasses import dataclass
import logging
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""
    video_id: str
    frame_index: int
    timestamp: float
    keypoints: tuple[cv2.KeyPoint, ...]


@dataclass
class FrameMatch:
    """Represents a match between two frames."""
    frame: FrameMetadata
    frame_reference: FrameMetadata
    distance_score: float


class FrameMatcher:
    def __init__(self, video_path: str | Path, sample_interval: int = 10, visualize: bool = False,
                 video_id="reference", nfeatures=200):
        """
        Initialize the Frame Matcher.

        Args:
            sample_interval: Number of frames to skip when sampling from video A
            visualize: Whether to show visualization during processing
        """
        self.sample_interval = sample_interval
        self.index: faiss.IndexFlatL2 | None = None

        self.frame_metadata = []
        self.visualize = visualize
        self.visualizer = Visualizer() if visualize else None
        self.video_path = Path(video_path)
        self.video_id = video_id
        # noinspection PyUnresolvedReferences
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        logger.info(f"Initialized FrameMatcher with sample_interval={sample_interval}, visualize={visualize}")

    def extract_features(self, frame: np.ndarray) -> tuple[np.ndarray, list[cv2.KeyPoint]]:
        """
        Extract features from a frame using SIFT.

        Args:
            frame: Input frame as numpy array

        Returns:
            Tuple of (feature vector, keypoints)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = self.sift

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None:
            logger.warning("No SIFT descriptors found in frame")
            return np.zeros((1, 128), dtype=np.float32), []

        # If we have multiple descriptors, take the mean
        return np.mean(descriptors, axis=0).reshape(1, -1).astype(np.float32), keypoints

    def build_index(self) -> None:
        """
        Process a video and build the FAISS index.
        """
        video_path, video_id = self.video_path, self.video_id
        logger.info(f"Building index for video {video_id} from {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        index, sample_interval = -1, self.sample_interval
        features_list = []

        try:
            while True:
                index += 1
                if index % sample_interval == 0:
                    ret, frame = cap.read()
                else:
                    frame = None
                    ret = cap.grab()

                if not ret:
                    break

                if frame is None:
                    continue

                features, keypoints = self.extract_features(frame)
                features_list.append(features)

                # Store metadata
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds

                self.frame_metadata.append(FrameMetadata(
                    video_id=video_id,
                    frame_index=index,
                    timestamp=timestamp,
                    keypoints=tuple(keypoints),
                ))

                # Show visualization if enabled
                if self.visualize:
                    self.visualizer.show_frame(frame, tuple(keypoints))



        except KeyboardInterrupt:
            logger.info("Indexing stopped by user")
        finally:
            cap.release()
            if self.visualize:
                self.visualizer.close()

        # Build FAISS index
        if features_list:
            features_array = np.vstack(features_list)
            dimension = features_array.shape[1]

            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(dimension)
            # noinspection PyArgumentList
            self.index.add(features_array, )

            logger.info(f"Successfully built index with {len(features_list)} frames from video {video_id}")
        else:
            error_msg = "No frames were processed from the video"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def find_matches(self, video_path: str | Path, video_id: str, k: int = 1) -> list[FrameMatch]:
        """
        Find matching frames in the indexed video for frames from a new video.

        Args:
            video_path: Path to the video file to match against
            video_id: Unique identifier for the video
            k: Number of nearest neighbors to return

        Returns:
            List of FrameMatch objects containing the matches
        """
        if self.index is None:
            error_msg = "Index not built. Call build_index before searching for matches."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Finding matches for video {video_id} from {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        reference_path = str(self.video_path)
        cap_ref = cv2.VideoCapture(reference_path)
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        matches = []
        frame_index = -1

        try:
            while True:
                frame_index += 1
                if frame_index % self.sample_interval == 0:
                    ret, frame = cap.read()
                else:
                    frame = None
                    ret = cap.grab()
                if not ret:
                    break

                if frame is None:
                    continue

                features, keypoints = self.extract_features(frame)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Search for nearest neighbors
                distances, indices = self.index.search(features, k)
                matches_and_distances = list(zip(indices[0], distances[0]))

                # Create match objects
                for i in range(k):
                    match_index, distance = matches_and_distances[i]
                    if match_index != -1:  # Valid match found

                        frame_meta = FrameMetadata(
                            video_id=video_id,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            keypoints=tuple(keypoints),
                        )
                        reference_match: FrameMetadata = self.frame_metadata[match_index]

                        match = FrameMatch(
                            frame=frame_meta,
                            frame_reference=reference_match,
                            distance_score=float(distance)
                        )
                        matches.append(match)
                        logger.debug(f"Found match: Frame {frame_index} -> Frame {match.frame_reference.frame_index} "
                                     f"(score: {match.distance_score:.4f})")

                        # Show visualization if enabled
                        if self.visualize and i == 0:
                            # Get the matched frame from the indexed video
                            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, match.frame_reference.frame_index)
                            ret_ref, frame_ref = cap_ref.read()
                            if ret_ref:
                                # Create matches for visualization
                                vis_matches = tuple()
                                self.visualizer.show_matches(frame, frame_ref, frame_meta.keypoints,
                                                             reference_match.keypoints, vis_matches)



        except KeyboardInterrupt:
            logger.info("Matching stopped by user")
        finally:
            cap.release()
            cap_ref.release()
            if self.visualize:
                self.visualizer.close()

        logger.info(f"Found {len(matches)} matches for video {video_id}")
        return matches
