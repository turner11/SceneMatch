import sys
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QSizePolicy, QCheckBox,
                             QStackedLayout)

from scene_match.scene_lib.frame_matcher import FrameMatch, FrameMatcher


# import numpy as np
# from ..imaging.reader import get_stream
# from ..imaging.stream_types import StreamParams


class MatchVisualizer(QMainWindow):

    @property
    def show_descriptors(self) -> bool:
        return self.visualize_descriptors_checkbox.isChecked()

    def __init__(self, matcher: FrameMatcher, video_path: str | Path):
        super().__init__()
        self.matcher = matcher
        self.video_path = Path(video_path).resolve().absolute()
        self.jump_size = 1

        self.video_path_ref = Path(matcher.video_source).resolve().absolute()

        non_existing_paths = [p for p in [self.video_path, self.video_path_ref] if not Path(p).exists()]
        if non_existing_paths:
            missing = ', '.join(str(p) for p in non_existing_paths)
            raise FileNotFoundError(f"Video file not found: {missing}")

        self.current_frame = 0
        self.is_playing = False
        self.current_match = None

        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.ref_cap = cv2.VideoCapture(str(self.video_path))
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

        # Frame slider (above controls)
        slider_layout = QHBoxLayout()
        slider_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.setSpacing(0)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.valueChanged.connect(self.set_frame)
        # Set min width to total button width (5*48 + 1*64 = 304), max width to 50% of window
        self.frame_slider.setMinimumWidth(304)
        self.frame_slider.setMaximumWidth(int(self.width() * 0.5))
        self.frame_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider_layout.addWidget(self.frame_slider)
        layout.addLayout(slider_layout)

        # Create View Stack for different visualizations
        self.view_stack = QStackedLayout()

        # Page 1: Normal two-video view
        normal_view_widget = QWidget()
        display_layout = QHBoxLayout(normal_view_widget)
        display_layout.setSpacing(10)

        # Comparing video layout
        comparing_video_layout = QVBoxLayout()
        comparing_header = QLabel("<h3>Comparing Video</h3>")
        comparing_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_label = QLabel()
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        comparing_video_layout.addWidget(comparing_header)
        comparing_video_layout.addWidget(self.current_label)
        display_layout.addLayout(comparing_video_layout)

        # Reference video layout
        reference_video_layout = QVBoxLayout()
        reference_header = QLabel("<h3>Reference Video (Index)</h3>")
        reference_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_label = QLabel()
        self.match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        reference_video_layout.addWidget(reference_header)
        reference_video_layout.addWidget(self.match_label)
        display_layout.addLayout(reference_video_layout)
        self.view_stack.addWidget(normal_view_widget)

        # Page 2: Descriptor view
        descriptor_view_widget = QWidget()
        descriptor_layout = QHBoxLayout(descriptor_view_widget)
        self.descriptor_label = QLabel()
        self.descriptor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.descriptor_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        descriptor_layout.addWidget(self.descriptor_label)
        self.view_stack.addWidget(descriptor_view_widget)

        # Add the view stack to the main layout
        stack_container = QWidget()
        stack_container.setLayout(self.view_stack)
        layout.insertWidget(0, stack_container)

        # Controls (below slider)
        controls_layout = QHBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.setSpacing(20)

        self.prev_button = QPushButton('⏮️')
        self.prev_button.setToolTip('Previous Frame')
        self.prev_button.setFixedSize(48, 48)
        self.prev_button.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_button)

        self.slower_button = QPushButton('⏪')
        self.slower_button.setToolTip('Slower')
        self.slower_button.setFixedSize(48, 48)
        self.slower_button.clicked.connect(self.slower_speed)
        controls_layout.addWidget(self.slower_button)

        self.play_button = QPushButton('▶️')
        self.play_button.setToolTip('Play/Pause')
        self.play_button.setFixedSize(64, 64)
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        self.faster_button = QPushButton('⏩')
        self.faster_button.setToolTip('Faster')
        self.faster_button.setFixedSize(48, 48)
        self.faster_button.clicked.connect(self.faster_speed)
        controls_layout.addWidget(self.faster_button)

        self.next_button = QPushButton('⏭️')
        self.next_button.setToolTip('Next Frame')
        self.next_button.setFixedSize(48, 48)
        self.next_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_button)

        self.speed_label = QLabel(f'Skip: {self.jump_size} frames')
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.speed_label)

        layout.addLayout(controls_layout)

        # Match info label (below controls, centered)
        self.match_info_label = QLabel()
        self.match_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.match_info_label)

        # Visualization options
        self.visualize_descriptors_checkbox = QCheckBox("Show Descriptors")
        self.visualize_descriptors_checkbox.stateChanged.connect(self.update_frames)
        layout.addWidget(self.visualize_descriptors_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        # Setup playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)

        # Update slider max width on resize
        self.resizeEvent = self._on_resize

    def update_frames(self):
        # Get current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, current_frame_bgr = self.cap.read()
        if not ret:
            return

        # Convert current frame to RGB
        current_frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)

        # Get match for current frame
        frame_meta = self.matcher.save_frame_features(current_frame_bgr, self.current_frame)
        self.current_match = self.matcher.match(frame_meta)

        if self.current_match:
            # Get reference frame
            ref_frame_meta = self.current_match.frame_reference
            ref_frame_index = ref_frame_meta.frame_index

            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_index)
            ret, ref_frame_bgr = self.ref_cap.read()

            if ret:
                ref_frame_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)

                if self.show_descriptors:
                    self.view_stack.setCurrentIndex(1)
                    # Draw matches
                    kp1, kp2, matches = self.matcher.get_detailed_matches(frame_meta, ref_frame_meta)
                    match_img = cv2.drawMatches(
                        current_frame_bgr, kp1, ref_frame_bgr, kp2, matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

                    # Display combined image in descriptor_label
                    h, w, ch = match_img_rgb.shape
                    q_img = QImage(match_img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    self.descriptor_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                        self.descriptor_label.width(), self.descriptor_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    ))
                else:
                    self.view_stack.setCurrentIndex(0)
                    # Display current frame
                    current_qimg = QImage(current_frame_rgb.data, current_frame_rgb.shape[1],
                                          current_frame_rgb.shape[0],
                                          current_frame_rgb.strides[0], QImage.Format.Format_RGB888)
                    self.current_label.setPixmap(QPixmap.fromImage(current_qimg).scaled(
                        self.current_label.width(), self.current_label.height(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))

                    # Display matched frame
                    match_qimg = QImage(ref_frame_rgb.data, ref_frame_rgb.shape[1], ref_frame_rgb.shape[0],
                                        ref_frame_rgb.strides[0], QImage.Format.Format_RGB888)
                    self.match_label.setPixmap(QPixmap.fromImage(match_qimg).scaled(
                        self.match_label.width(), self.match_label.height(), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))

                # Update match info
                info_text = (f"Match: Frame {self.current_frame} → Frame {ref_frame_index}\n"
                             f"Score: {self.current_match.distance_score:.2f}\n"
                             f"Features: {self.current_match.features_percentage:.1%}")
                self.match_info_label.setText(info_text)
        else:
            # No match found
            self.match_info_label.setText("No match found")
            if self.show_descriptors:
                self.view_stack.setCurrentIndex(1)
                self.descriptor_label.clear()
                self.descriptor_label.setText("No match to visualize")
            else:
                self.view_stack.setCurrentIndex(0)
                self.match_label.clear()
                current_qimg = QImage(current_frame_rgb.data, current_frame_rgb.shape[1],
                                      current_frame_rgb.shape[0],
                                      current_frame_rgb.strides[0], QImage.Format.Format_RGB888)
                self.current_label.setPixmap(QPixmap.fromImage(current_qimg).scaled(
                    self.current_label.width(), self.current_label.height(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

        # Update frame slider
        self.frame_slider.setValue(self.current_frame)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText('⏸️' if self.is_playing else '▶️')
        if self.is_playing:
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.playback_timer.start(interval)
        else:
            self.playback_timer.stop()

    def faster_speed(self):
        self.jump_size += 10
        self.speed_label.setText(f'Skip: {self.jump_size} frames')

    def slower_speed(self):
        if self.jump_size > 1:
            self.jump_size = max(1, self.jump_size - 10)
        self.speed_label.setText(f'Skip: {self.jump_size} frames')

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += self.jump_size
            self.current_frame = min(self.current_frame, self.total_frames - 1)
            self.update_frames()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= self.jump_size
            self.current_frame = max(0, self.current_frame)
            self.update_frames()

    def set_frame(self, index: int):
        self.current_frame = index
        self.update_frames()

    def closeEvent(self, event):
        self.cap.release()
        if hasattr(self, 'ref_cap') and self.ref_cap is not None:
            self.ref_cap.release()
        super().closeEvent(event)

    def _on_resize(self, event):
        # Update slider max width to 50% of window width
        self.frame_slider.setMaximumWidth(int(self.width() * 0.5))
        super().resizeEvent(event)


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
