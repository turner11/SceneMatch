"""
Visualizer module for displaying video frames and detected features.
"""
import cv2
import numpy as np
import logging

from scene_match.lib.match_types import FrameMatch, FrameMetadata

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, window_name: str = "SceneMatch Visualizer"):
        """
        Initialize the visualizer.
        
        Args:
            window_name: Name of the visualization window
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        logger.info(f"Initialized visualizer with window: {window_name}")

    @staticmethod
    def draw_keypoints(frame: np.ndarray, keypoints: tuple[cv2.KeyPoint, ...],
                       color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw SIFT keypoints on the frame.
        
        Args:
            frame: Input frame
            keypoints: List of SIFT keypoints
            color: Color for the keypoints (BGR format)
            
        Returns:
            Frame with keypoints drawn
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()

        # Draw keypoints
        for kp in keypoints:
            x, y = map(int, kp.pt)
            size = int(kp.size)
            # Draw circle for keypoint
            cv2.circle(vis_frame, (x, y), size, color, 1)
            # Draw orientation line
            angle = kp.angle * np.pi / 180.0
            end_x = int(x + size * np.cos(angle))
            end_y = int(y + size * np.sin(angle))
            cv2.line(vis_frame, (x, y), (end_x, end_y), color, 1)

        return vis_frame

    def show_frame(self, image: np.ndarray, keypoints: tuple[cv2.KeyPoint, ...] = None, show_keypoints=True) -> None:
        """
        Display a frame with optional keypoints.
        
        Args:
            image: Frame to display
            keypoints: Optional list of keypoints to draw
            show_keypoints: Whether to draw keypoints on the frame
        """
        if show_keypoints and keypoints is not None:
            image = self.draw_keypoints(image, keypoints)

        cv2.imshow(self.window_name, image)

    def show_frame_matches(self, image: np.ndarray, image_reference: np.ndarray, matches: FrameMatch,
                           show_keypoints=False,
                           show_match_lines=True,
                           features_to_draw=10) -> None:
        """
        Display two frames side by side with matching keypoints.
        
        Args:
            image: First frame to display
            image_reference: Second frame to display
            matches: List of FrameMatch objects containing keypoints and matches
            show_keypoints: Whether to draw keypoints on the frames
            show_match_lines: Whether to draw lines between matching keypoints
        """

        frame_data: FrameMetadata = matches.frame
        frame_ref_data: FrameMetadata = matches.frame_reference

        if show_match_lines:
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            # Draw lines between matching keypoints
            features_matches = bf.match(frame_data.features, frame_ref_data.features)
            features_matches = sorted(features_matches, key=lambda x: x.distance)
        else:
            features_matches = []

        # features_matches = features_matches[:features_to_draw]

        # flags docs: https://docs.opencv.org/4.x/d4/d5d/group__features2d__draw.html#ga2c2ede79cd5141534ae70a3fd9f324c8
        flags = cv2.DrawMatchesFlags_DEFAULT
        if show_keypoints:
            flags = flags | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        else:
            flags = flags | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

        img_matches = cv2.drawMatches(
            image, frame_data.keypoints, image_reference, frame_ref_data.keypoints, features_matches,
            outImg=None,
            # matchColor=(0, 255, 0), # Green color for matches
            singlePointColor=(0, 0, 255),  # Red color for no match
            flags=flags,
        )



        cv2.imshow(self.window_name, img_matches)

    @staticmethod
    def close() -> None:
        """Close all visualization windows."""
        cv2.destroyAllWindows()

        logger.info("Closed visualization windows")
