"""
Frame Matcher module for finding similar frames between two drone videos.
"""
import itertools
import statistics
from collections import defaultdict
from pathlib import Path
import orjson

import cv2
import numpy as np
import faiss
import logging

from .match_types import FrameMetadata, FrameMatch

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = 'index.faiss'
META_DATA_FILE_NAME = 'metadata.json'


class FrameMatcher:
    def __init__(self, video_source: str | Path, n_features=500):
        """
        Initialize the Frame Matcher.

        Args:
            n_features: Number of SIFT features to extract from each frame
        """

        self.video_source = Path(video_source).resolve().absolute()
        self.index: faiss.IndexFlatL2 | None = None
        self.n_features = n_features
        self.frame_metadata_by_frame_index = {}
        self.frame_data_by_frame_index = []
        # noinspection PyUnresolvedReferences
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        logger.info(f"Initialized FrameMatcher with n_features={n_features}")

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

    def save_frame_features(self, image, frame_index, timestamp=-1):
        features, keypoints = self.extract_features(image)

        frame_data = FrameMetadata(
            frame_index=frame_index,
            timestamp=timestamp,
            keypoints=tuple(keypoints),
            features=features,
        )

        self.frame_metadata_by_frame_index[frame_index] = frame_data

        return frame_data

    def match(self, frame_meta: FrameMetadata, k: int = 1) -> FrameMatch:
        """
        Find matching frames in the indexed video for frames from a new video.

        Args:
            frame_meta: FrameMetadata object containing the frame to match
            k: Number of nearest neighbors to return

        Returns:
            List of FrameMatch objects containing the matches
        """
        if self.index is None:
            error_msg = "Index not built. Call build_index before searching for matches."
            logger.error(error_msg)
            raise ValueError(error_msg)

        features, keypoints = frame_meta.features, frame_meta.keypoints

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
        votes = len(distances)

        match = None
        if match_index != -1:  # Valid match found
            reference_match: FrameMetadata = self.frame_metadata_by_frame_index[match_index]

            match = FrameMatch(
                frame=frame_meta,
                frame_reference=reference_match,
                distance_score=float(statistics.median(distances)),
                features_percentage=votes / float(self.n_features),
                notes=f'{votes} features matched)',
            )

            # logger.debug(f"Found match: Frame {frame_meta.frame_index} -> Frame {reference_match.frame_index} "
            #              f"(score: {match.distance_score:.4f})")

        return match

    def get_detailed_matches(self, frame_meta1: FrameMetadata, frame_meta2: FrameMetadata, max_matches=50):
        """
        Get detailed descriptor matches between two frames.
        """
        features1, keypoints1 = frame_meta1.features, frame_meta1.keypoints
        features2, keypoints2 = frame_meta2.features, frame_meta2.keypoints

        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(features1, features2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        return keypoints1, keypoints2, matches[:max_matches]

    def build_index(self):
        metadata_by_frame_index = self.frame_metadata_by_frame_index
        if not metadata_by_frame_index:
            raise ValueError("No frame metadata available to build index.")

        tpls = sorted(metadata_by_frame_index.items(), key=lambda t: t[0])
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

        logger.info(f"Built index with {len(all_descriptors)} features from {len(metadata_by_frame_index)} frames")

        return self.index

    def serialize(self, path: str | Path):
        """
        Serialize the frame matcher state to a directory.
        
        Args:
            path: directory path to save the serialized data to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save frame metadata
        metadata_path = (path / 'metadata.json').resolve().absolute()
        metadata_dict = {
            'video_source': str(self.video_source),
            'n_features': self.n_features,
            'frame_metadata': {str(k): v.to_dict() for k, v in self.frame_metadata_by_frame_index.items()},
            'frame_data_by_frame_index': self.frame_data_by_frame_index
        }
        logger.info(f"Dumping index metadata to '{metadata_path}'")
        with open(metadata_path, 'wb') as f:
            f.write(orjson.dumps(metadata_dict))

        # Save FAISS index if it exists
        if self.index is not None:
            logger.info(f"Dumping Feiss index")
            index_path = path / INDEX_FILE_NAME
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

        logger.info(f"Saved frame matcher state to {path} ({len(self.frame_metadata_by_frame_index)} frames)")
        return path.resolve().absolute()

    @classmethod
    def deserialize(cls, path: str | Path) -> 'FrameMatcher':
        """
        Deserialize a frame matcher from a directory.
        
        Args:
            path: Path to the serialized data directory
            
        Returns:
            FrameMatcher: The deserialized frame matcher
        """
        path = Path(path)

        # Load metadata
        metadata_path = (path / META_DATA_FILE_NAME).resolve().absolute()
        with open(metadata_path, 'rb') as f:
            metadata_dict = orjson.loads(f.read())

        # Create matcher instance
        matcher = FrameMatcher(video_source=metadata_dict['video_source'], n_features=metadata_dict['n_features'])

        # Load frame metadata
        matcher.frame_metadata_by_frame_index = {
            int(k): FrameMetadata.from_dict(v)
            for k, v in metadata_dict['frame_metadata'].items()
        }
        matcher.frame_data_by_frame_index = metadata_dict['frame_data_by_frame_index']

        # Load FAISS index if it exists
        index_path = path / INDEX_FILE_NAME
        if index_path.exists():
            matcher.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {index_path}")

        logger.info(f"Loaded frame matcher state from {path}")
        return matcher
