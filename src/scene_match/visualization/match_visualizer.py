import sys
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog)

from scene_match.scene_lib.frame_matcher import FrameMatch, FrameMatcher
# import numpy as np
# from ..imaging.reader import get_stream
# from ..imaging.stream_types import StreamParams


class MatchVisualizer(QMainWindow):
    def __init__(self, matcher: FrameMatcher, video_path: str | Path):
        super().__init__()
        self.matcher = matcher
        self.video_path = Path(video_path).resolve().absolute()


        self.video_path_ref = Path(matcher.video_source).resolve().absolute()

        non_existing_paths = [p for p in [self.video_path, self.video_path_ref] if not Path(p).exists()]
        if non_existing_paths:
            missing = ', '.join(str(p) for p in non_existing_paths)
            raise FileNotFoundError(f"Video file not found: {missing}")

        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1  # frames per second
        self.current_match = None

        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.ref_cap =   cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.init_ui()
        self.update_frames()

    def init_ui(self):
        self.setWindowTitle('Match Visualizer')
        self.setGeometry(100, 100, 1600, 900)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create image display area
        display_layout = QHBoxLayout()
        
        # Current frame display
        self.current_label = QLabel()
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.current_label)

        # Match frame display
        self.match_label = QLabel()
        self.match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.match_label)

        layout.addLayout(display_layout)

        # Create controls
        controls_layout = QHBoxLayout()

        # Playback controls
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        self.prev_button = QPushButton('Previous')
        self.prev_button.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_button)

        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_button)

        # Speed control
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.change_speed)
        controls_layout.addWidget(QLabel('Speed:'))
        controls_layout.addWidget(self.speed_slider)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.valueChanged.connect(self.set_frame)
        controls_layout.addWidget(QLabel('Frame:'))
        controls_layout.addWidget(self.frame_slider)

        # Match info label
        self.match_info_label = QLabel()
        self.match_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.match_info_label)

        layout.addLayout(controls_layout)

        # Setup playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)

    def update_frames(self):
        # Get current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, current_frame = self.cap.read()
        if not ret:
            return

        # Convert current frame to RGB
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        current_qimg = QImage(current_frame.data, current_frame.shape[1], current_frame.shape[0],
                            current_frame.strides[0], QImage.Format.Format_RGB888)
        self.current_label.setPixmap(QPixmap.fromImage(current_qimg).scaled(
            self.current_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Get match for current frame
        frame_meta = self.matcher.save_frame_features(current_frame, self.current_frame)
        self.current_match = self.matcher.match(frame_meta)

        if self.current_match:
            # Get reference frame
            ref_frame = self.current_match.frame_reference
            ref_frame_index = ref_frame.frame_index
            
            # Load reference video if needed
            if not hasattr(self, 'ref_cap') or self.ref_cap is None:
                self.ref_cap = cv2.VideoCapture(str(self.matcher.reference_path))
            
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_index)
            ret, match_frame = self.ref_cap.read()
            
            if ret:
                match_frame = cv2.cvtColor(match_frame, cv2.COLOR_BGR2RGB)
                match_qimg = QImage(match_frame.data, match_frame.shape[1], match_frame.shape[0],
                                  match_frame.strides[0], QImage.Format.Format_RGB888)
                self.match_label.setPixmap(QPixmap.fromImage(match_qimg).scaled(
                    self.match_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                
                # Update match info
                info_text = (f"Match: Frame {self.current_frame} → Frame {ref_frame_index}\n"
                           f"Score: {self.current_match.distance_score:.2f}\n"
                           f"Features: {self.current_match.features_percentage:.1%}")
                self.match_info_label.setText(info_text)
        else:
            self.match_label.clear()
            self.match_info_label.setText("No match found")

        # Update frame slider
        self.frame_slider.setValue(self.current_frame)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText('Pause' if self.is_playing else 'Play')
        if self.is_playing:
            self.playback_timer.start(1000 // self.playback_speed)
        else:
            self.playback_timer.stop()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frames()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frames()

    def set_frame(self, index: int):
        self.current_frame = index
        self.update_frames()

    def change_speed(self, speed: int):
        self.playback_speed = speed
        if self.is_playing:
            self.playback_timer.setInterval(1000 // speed)

    def closeEvent(self, event):
        self.cap.release()
        if hasattr(self, 'ref_cap') and self.ref_cap is not None:
            self.ref_cap.release()
        super().closeEvent(event)


def visualize_matches(matcher: FrameMatcher, video_path: str | Path):
    """
    Launch the match visualizer application.
    
    Args:
        matcher: FrameMatcher instance with built index
        video_path: Path to the video file to analyze
    """
    app = QApplication(sys.argv)
    visualizer = MatchVisualizer(matcher, video_path)
    visualizer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize frame matches')
    parser.add_argument('video', help='Path to video file to analyze')
    parser.add_argument('matcher', help='Path to matcher file (serialized FrameMatcher)')
    
    args = parser.parse_args()

    matcher = FrameMatcher.deserialize(args.matcher)
    visualize_matches(matcher, args.video)