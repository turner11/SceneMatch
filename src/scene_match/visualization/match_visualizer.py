import sys
from pathlib import Path


import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog)

from ..scene_lib.frame_matcher import FrameMatch, FrameMatcher
# import numpy as np
# from ..imaging.reader import get_stream
# from ..imaging.stream_types import StreamParams


class MatchVisualizer(QMainWindow):

    def __init__(self, matcher: FrameMatcher, source_path: str | Path, reference_path: str | Path):
        super().__init__()
        self.matcher = matcher
        self.source_path = Path(source_path)
        self.reference_path = Path(reference_path)
        self.current_index = 0
        self.is_playing = False
        self.playback_speed = 1  # frames per second

        # Initialize video captures
        self.source_cap = cv2.VideoCapture(str(self.source_path))
        self.reference_cap = cv2.VideoCapture(str(self.reference_path))

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
        
        # Source frame display
        self.source_label = QLabel()
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.source_label)

        # Reference frame display
        self.reference_label = QLabel()
        self.reference_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        display_layout.addWidget(self.reference_label)

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
        self.frame_slider.setMaximum(len(self.matches) - 1)
        self.frame_slider.valueChanged.connect(self.set_frame)
        controls_layout.addWidget(QLabel('Frame:'))
        controls_layout.addWidget(self.frame_slider)

        layout.addLayout(controls_layout)

        # Setup playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)

    def update_frames(self):
        if not self.matches:
            return

        match = self.matches[self.current_index]
        
        # Get source frame
        self.source_cap.set(cv2.CAP_PROP_POS_FRAMES, match.frame.frame_index)
        ret, source_frame = self.source_cap.read()
        if ret:
            source_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
            source_qimg = QImage(source_frame.data, source_frame.shape[1], source_frame.shape[0],
                               source_frame.strides[0], QImage.Format.Format_RGB888)
            self.source_label.setPixmap(QPixmap.fromImage(source_qimg).scaled(
                self.source_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Get reference frame
        self.reference_cap.set(cv2.CAP_PROP_POS_FRAMES, match.frame_reference.frame_index)
        ret, reference_frame = self.reference_cap.read()
        if ret:
            reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)
            reference_qimg = QImage(reference_frame.data, reference_frame.shape[1], reference_frame.shape[0],
                                  reference_frame.strides[0], QImage.Format.Format_RGB888)
            self.reference_label.setPixmap(QPixmap.fromImage(reference_qimg).scaled(
                self.reference_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Update frame slider
        self.frame_slider.setValue(self.current_index)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText('Pause' if self.is_playing else 'Play')
        if self.is_playing:
            self.playback_timer.start(1000 // self.playback_speed)
        else:
            self.playback_timer.stop()

    def next_frame(self):
        if self.current_index < len(self.matches) - 1:
            self.current_index += 1
            self.update_frames()

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_frames()

    def set_frame(self, index: int):
        self.current_index = index
        self.update_frames()

    def change_speed(self, speed: int):
        self.playback_speed = speed
        if self.is_playing:
            self.playback_timer.setInterval(1000 // speed)

    def closeEvent(self, event):
        self.source_cap.release()
        self.reference_cap.release()
        super().closeEvent(event)


def visualize_matches(matcher: FrameMatcher, source_path: str | Path, reference_path: str | Path):
    """
    Launch the match visualizer application.
    
    Args:
        matches: List of FrameMatch objects containing the matching results
        source_path: Path to the source video file
        reference_path: Path to the reference video file
    """
    app = QApplication(sys.argv)
    visualizer = MatchVisualizer(matcher, source_path, reference_path)
    visualizer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize frame matches')
    parser.add_argument('source', help='Path to source video file')
    parser.add_argument('reference', help='Path to reference video file')
    parser.add_argument('matcher', help='Path to matcher file (serialized FrameMatcher)')
    
    args = parser.parse_args()

    matcher = FrameMatcher.deserialize(args.matcher)

    
    visualize_matches(matcher, args.source, args.reference)