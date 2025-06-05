"""
Visualizer module for displaying video frames and detected features.
"""
import cv2
import numpy as np
import logging

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

    def show_frame(self, frame: np.ndarray, keypoints: tuple[cv2.KeyPoint,...] = None, wait_time: int = 1) -> None:
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



    def show_matches(self, frame1: np.ndarray, frame2: np.ndarray,
                     keypoints: tuple[cv2.KeyPoint, ...], keypoints_ref: tuple[cv2.KeyPoint, ...],
                     matches: tuple[cv2.DMatch], wait_time: int = 1) -> None:
        """
        Display two frames side by side with matching keypoints.
        
        Args:
            frame1: First frame
            frame2: Second frame
            keypoints: Keypoints in first frame
            keypoints_ref: Keypoints in second frame
            matches: List of matches between keypoints
            wait_time: Time to wait for key press (ms)
        """
        # Draw keypoints on both frames
        vis_a = self.draw_keypoints(frame1, keypoints, (0, 255, 0))
        vis_b = self.draw_keypoints(frame2, keypoints_ref, (255, 0, 0))

        # Create side-by-side visualization
        h1, w1 = vis_a.shape[:2]
        h2, w2 = vis_b.shape[:2]

        # Create a canvas for both images
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = vis_a
        vis[:h2, w1:w1 + w2] = vis_b

        # Draw lines between matching keypoints
        matches = [m for m in matches if m.queryIdx and m.trainIdx]
        for match in matches:
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints_ref[match.trainIdx].pt))
            pt2 = (pt2[0] + w1, pt2[1])  # Adjust x coordinate for second image

            # Draw line between matches
            cv2.line(vis, pt1, pt2, (0, 0, 255), 1)

        cv2.imshow(self.window_name, vis)
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
