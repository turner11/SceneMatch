"""
Frame Matcher module for finding similar frames between two drone videos.
"""
import itertools
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import faiss
import logging

from .match_types import FrameMetadata, FrameMatch
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class FrameMatcher:
    def __init__(self, video_path: str | Path, sample_interval: int = 10, visualize: bool = False,
                 video_id="reference", n_features=500):
        """
        Initialize the Frame Matcher.

        Args:
            sample_interval: Number of frames to skip when sampling from video A
            visualize: Whether to show visualization during processing
        """
        self.sample_interval = sample_interval
        self.index: faiss.IndexFlatL2 | None = None

        self.visualize = visualize
        self.visualizer = Visualizer() if visualize else None
        self.video_path = Path(video_path)
        self.video_id = video_id
        self.n_features = n_features
        self.frame_metadata_by_frame_index = {}
        self.frame_data_by_frame_index = []
        # noinspection PyUnresolvedReferences
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        logger.info(f"Initialized FrameMatcher with sample_interval={sample_interval}, visualize={visualize}")

    def extract_features(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[cv2.KeyPoint, ...]]:
        """
        Extract features from a frame using SIFT.

        Args:
            frame: Input frame as numpy-array

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
            return np.zeros((1, 128), dtype=np.float32), tuple()

        # If we have multiple descriptors, take the mean
        return descriptors.astype(np.float32), keypoints

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

        frame_index, sample_interval = -1, self.sample_interval
        frame_metadata_by_frame_index = {}
        self.frame_metadata_by_frame_index.clear()
        try:
            while True:
                frame_index += 1
                if frame_index % sample_interval == 0:
                    ret, frame = cap.read()
                else:
                    frame = None
                    ret = cap.grab()

                if not ret:
                    break

                if frame is None:
                    continue

                features, keypoints = self.extract_features(frame)
                # Store metadata
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds

                frame_metadata_by_frame_index[frame_index] = FrameMetadata(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    keypoints=tuple(keypoints),
                    features=features,
                )

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
        if frame_metadata_by_frame_index:
            self.frame_metadata_by_frame_index.update(frame_metadata_by_frame_index)

            tpls = sorted(frame_metadata_by_frame_index.items(), key=lambda t: t[0])
            frames_meta: list[FrameMetadata] = [t[1] for t in tpls]
            descriptor_to_frame_index = list(
                itertools.chain.from_iterable([m.frame_index] * len(m.features) for m in frames_meta))

            all_descriptors = [m.features for m in frames_meta]
            features_array = np.vstack(all_descriptors).astype(np.float32)
            dimension = features_array.shape[1]

            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(dimension)
            # noinspection PyArgumentList
            self.index.add(features_array, )
            self.frame_data_by_frame_index[:] = descriptor_to_frame_index

            logger.info(f"Successfully built index with {len(all_descriptors)} frames from video {video_id}")
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
                # noinspection PyArgumentList
                distances, indices = self.index.search(features, k)
                desc_to_idx = self.frame_data_by_frame_index

                # Create match objects
                distance_threshold = float('inf')
                image_votes = defaultdict(list)
                for dists, desc_idxs in zip(distances, indices):
                    for dist, desc_idx in zip(dists, desc_idxs):
                        if desc_idx != -1 and dist < distance_threshold:
                            match_index = desc_to_idx[desc_idx]
                            image_votes[match_index].append(dist)

                # Majority vote
                tpl = max(image_votes.items(), key=lambda t: len(t[1]))
                match_index, distances = tpl

                total_vectors = len(distances)
                votes = len(distances)

                if match_index != -1:  # Valid match found
                    frame_meta = FrameMetadata(
                        video_id=video_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        keypoints=tuple(keypoints),
                        features=features,
                    )
                    reference_match: FrameMetadata = self.frame_metadata_by_frame_index[match_index]

                    match = FrameMatch(
                        frame=frame_meta,
                        frame_reference=reference_match,
                        distance_score=float(statistics.median(distances)),
                        notes=f'{votes} votes / {total_vectors} vectors ({votes * 100 / float(total_vectors)}%)',
                    )
                    matches.append(match)
                    logger.debug(f"Found match: Frame {frame_index} -> Frame {match.frame_reference.frame_index} "
                                 f"(score: {match.distance_score:.4f})")

                    # Show visualization if enabled
                    if self.visualize:
                        # Get the matched frame from the indexed video
                        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, match.frame_reference.frame_index)
                        ret_ref, frame_ref = cap_ref.read()
                        if ret_ref:
                            # Create matches for visualization
                            self.visualizer.show_matches(frame, frame_ref, match)



        except KeyboardInterrupt:
            logger.info("Matching stopped by user")
        finally:
            cap.release()
            cap_ref.release()
            if self.visualize:
                self.visualizer.close()

        logger.info(f"Found {len(matches)} matches for video {video_id}")
        return matches
