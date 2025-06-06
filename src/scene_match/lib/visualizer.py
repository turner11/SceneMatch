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

    def show_frame(self, frame: np.ndarray, keypoints: tuple[cv2.KeyPoint, ...] = None, wait_time: int = 1) -> None:
        """
        Display a frame with optional keypoints.
        
        Args:
            frame: Frame to display
            keypoints: Optional list of keypoints to draw
            wait_time: Time to wait for key press (ms)
        """
        if keypoints is not None:
            frame = self.draw_keypoints(frame, keypoints)

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(wait_time)

        # Handle window close
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Visualization stopped by user")

    def show_matches(self, frame: np.ndarray, frame_reference: np.ndarray, matches: FrameMatch,
                     draw_descriptors=False, wait_time: int = 1) -> None:
        """
        Display two frames side by side with matching keypoints.
        
        Args:
            frame: First frame to display
            frame_reference: Second frame to display
            matches: List of FrameMatch objects containing keypoints and matches
            draw_descriptors: Whether to draw keypoints on the frames
            wait_time: Time to wait for key press (ms)
        """

        frame_data: FrameMetadata = matches.frame
        frame_ref_data: FrameMetadata = matches.frame_reference
        # Draw keypoints on both frames
        if draw_descriptors:
            img = self.draw_keypoints(frame, frame_data.keypoints, (0, 255, 0))
            img_ref = self.draw_keypoints(frame_reference, frame_ref_data.keypoints, (255, 0, 0))
        else:
            img, img_ref = frame.copy(), frame_reference.copy()

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # Draw lines between matching keypoints
        features_matches = bf.match(frame_data.features, frame_ref_data.features)
        features_matches = sorted(features_matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(
            img, frame_data.keypoints, img_ref, frame_ref_data.keypoints, features_matches[:10], None,
            matchColor=(0, 255, 0),  # Green color for matches
            singlePointColor=(255, 0, 0),  # Blue color for keypoints
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # Don't draw unmatched keypoints
        )

        cv2.imshow(self.window_name, img_matches)
        key = cv2.waitKey(wait_time)

        # Handle window close
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Visualization stopped by user")

    @staticmethod
    def close() -> None:
        """Close all visualization windows."""
        cv2.destroyAllWindows()

        logger.info("Closed visualization windows")
