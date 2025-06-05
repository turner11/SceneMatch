"""
Tests for the frame matcher module.
"""
import tempfile
from pathlib import Path

import pytest
import numpy as np
import cv2

from scene_match.lib import frame_matcher as fm


def create_test_video(filename: str, num_frames: int = 30) -> None:
    """Create a test video file."""
    w, h = 1800, 1020
    # Create a video writer
    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))

    rect_width = 100
    rect_height = 100

    # Create frames with a moving pattern
    for i in range(num_frames):
        # Create a frame with a moving rectangle
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 10) % (w - rect_width)  # Moving rectangle
        x2 = x + rect_width
        y = x
        y2 = y + rect_height
        cv2.rectangle(frame, (x, y), (x2 + rect_width, y2), (0, 255, 0), -1)
        out.write(frame)

    out.release()


@pytest.fixture(scope='session')
def temp_dir():
    sub_path = "scene_match_test_"
    with tempfile.TemporaryDirectory(prefix=sub_path) as temp_path:
        yield Path(temp_path)


@pytest.fixture(scope="session")
def test_video(temp_dir):
    """Create a test video A."""
    video_path = temp_dir / "test_video.mp4"
    create_test_video(str(video_path))
    return str(video_path)


@pytest.fixture(scope="session")
def test_video_reference(temp_dir):
    """Create a test video B with similar patterns."""
    video_path = temp_dir / "test_video_reference.mp4"
    create_test_video(str(video_path))
    return str(video_path)


@pytest.fixture
def matcher_no_index(test_video_reference):
    matcher = fm.FrameMatcher(video_path=test_video_reference, sample_interval=5,
                              visualize=False, video_id="test reference")
    return matcher


@pytest.fixture(scope='module')
def matcher(test_video_reference):
    matcher = fm.FrameMatcher(video_path=test_video_reference, sample_interval=5,
                              visualize=False, video_id="test reference")
    matcher.build_index()
    return matcher


def test_frame_matcher_initialization(matcher_no_index):
    """Test FrameMatcher initialization."""
    assert matcher_no_index.index is None
    assert len(matcher_no_index.frame_metadata_by_frame_index) == 0


def test_build_index(matcher):
    """Test building the index from a video."""
    matcher.build_index()
    assert matcher.index is not None
    assert len(matcher.frame_metadata_by_frame_index) > 0


def test_find_matches(matcher, test_video):
    """Test finding matches between videos."""
    matches = matcher.find_matches(test_video, "test video")

    assert len(matches) > 0
    assert all(isinstance(match, fm.FrameMatch) for match in matches)
    assert all(isinstance(match.frame, fm.FrameMetadata) for match in matches)
    assert all(isinstance(match.frame_reference, fm.FrameMetadata) for match in matches)


def test_extract_features(matcher):
    """Test feature extraction from a frame."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)

    features, keypoints = matcher.extract_features(frame)
    assert features.shape[1] == 128  # SIFT descriptor size
    assert features.dtype == np.float32


def test_match_with_self(matcher):
    """Test matching a video with itself."""
    matches = matcher.find_matches(matcher.video_path, matcher.video_id)

    assert len(matches) > 0
    for match in matches:
        assert match.frame.video_id == matcher.video_id, "Frame video ID should match matcher video ID"
        assert match.frame_reference.video_id == matcher.video_id, "frame video ID should match matcher video ID"
        assert match.distance_score == 0.0, "Distance score should be zero when matching with itself"
        assert match.frame.frame_index == match.frame_reference.frame_index, \
            "Frame index should match reference frame index when matching with itself"
