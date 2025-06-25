import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QAction, QActionGroup
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QSizePolicy, QCheckBox,
                             QStackedLayout, QComboBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox)

from scene_match.scene_lib.frame_matcher import FrameMatch, FrameMatcher

# import numpy as np
# from ..imaging.reader import get_stream
# from ..imaging.stream_types import StreamParams


DARK_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #2E2E2E;
        color: #E0E0E0;
        font-family: "Segoe UI", "Frutiger", "Frutiger Linotype", "Dejavu Sans", "Helvetica Neue", "Arial", sans-serif;
    }
    QGraphicsView {
        border: 1px solid #454545;
        background-color: #1E1E1E;
    }
    QLabel {
        background-color: transparent;
        border: none;
    }
    QLabel[level="h3"] {
        color: #CCCCCC;
        font-size: 16px;
        font-weight: bold;
    }
    QPushButton {
        background-color: #4A4A4A;
        color: #E0E0E0;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px;
        font-size: 18px;
    }
    QPushButton:hover {
        background-color: #5A5A5A;
        border-color: #666666;
    }
    QPushButton:pressed {
        background-color: #3A3A3A;
    }
    QPushButton#PlayButton {
        font-size: 28px;
        padding-bottom: 4px; /* Center the icon vertically */
    }
    QSlider::groove:horizontal {
        border: 1px solid #454545;
        height: 4px;
        background: #3E3E3E;
        margin: 2px 0;
    }
    QSlider::handle:horizontal {
        background: #8A8A8A;
        border: 1px solid #999999;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    QComboBox {
        selection-background-color: #6A6A6A;
        background-color: #4A4A4A;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: #555555;
        border-left-style: solid;
    }
    QComboBox QAbstractItemView {
        background-color: #4A4A4A;
        border: 1px solid #555555;
        selection-background-color: #6A6A6A;
        color: #E0E0E0;
    }
"""


class FrameProcessorWorker(QObject):
    frame_processed = pyqtSignal(int, str, object, object, str)  # frame_idx, mode, img1, img2, info_text
    ready = pyqtSignal()

    def __init__(self, matcher, video_path, video_path_ref):
        super().__init__()
        self.matcher = matcher
        self.video_path = video_path
        self.video_path_ref = video_path_ref
        self.cap = None
        self.ref_cap = None

    @pyqtSlot()
    def setup_captures(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.ref_cap = cv2.VideoCapture(str(self.video_path_ref))
        self.ready.emit()

    @pyqtSlot(int, str)
    def process_frame(self, frame_index, mode):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, current_frame_bgr = self.cap.read()
        if not ret:
            return

        frame_meta = self.matcher.save_frame_features(current_frame_bgr, frame_index)
        current_match = self.matcher.match(frame_meta)

        img1_data, img2_data, info_text = None, None, "No match found"

        if current_match:
            ref_frame_meta = current_match.frame_reference
            ref_frame_index = ref_frame_meta.frame_index
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_index)
            ret_ref, ref_frame_bgr = self.ref_cap.read()

            info_text = (f"Match: Frame {frame_index} → Frame {ref_frame_index}\n"
                         f"Score: {current_match.distance_score:.2f}\n"
                         f"Features: {current_match.features_percentage:.1%}")

            if ret_ref:
                if mode == "Show Matches":
                    kp1, kp2, matches = self.matcher.get_detailed_matches(frame_meta, ref_frame_meta)
                    img1_data = cv2.drawMatches(
                        current_frame_bgr, kp1, ref_frame_bgr, kp2, matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                elif mode == "Show Unmatched Descriptors":
                    kp1, kp2, matches = self.matcher.get_detailed_matches(frame_meta, ref_frame_meta)
                    matched_kp1_indices = {m.queryIdx for m in matches}
                    unmatched_kp1 = [kp for i, kp in enumerate(kp1) if i not in matched_kp1_indices]
                    img1_data = cv2.drawKeypoints(current_frame_bgr, unmatched_kp1, None, color=(0, 0, 255),
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    img2_data = ref_frame_bgr
                else:  # Images Only
                    img1_data = current_frame_bgr
                    img2_data = ref_frame_bgr
        else:  # No match
            if mode == "Show Unmatched Descriptors":
                kp1, _, _ = self.matcher.get_detailed_matches(frame_meta, frame_meta)
                img1_data = cv2.drawKeypoints(current_frame_bgr, kp1, None, color=(0, 255, 0),
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:  # Images Only or Show Matches (with no match)
                img1_data = current_frame_bgr

        self.frame_processed.emit(frame_index, mode, img1_data, img2_data, info_text)


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._pixmap_item = None
        self._has_been_fitted = False

    def setImage(self, pixmap: QPixmap):
        if pixmap.isNull():
            if self._pixmap_item:
                self._scene.removeItem(self._pixmap_item)
                self._pixmap_item = None
                self._scene.setSceneRect(self._scene.itemsBoundingRect())
            return

        if self._pixmap_item:
            self._pixmap_item.setPixmap(pixmap)
        else:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
            self._has_been_fitted = False  # Ready for an initial fit

        self._scene.setSceneRect(self._pixmap_item.boundingRect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._has_been_fitted and self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._has_been_fitted = True

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mouseDoubleClickEvent(self, event):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)


class MatchVisualizer(QMainWindow):
    request_frame = pyqtSignal(int, str)

    def __init__(self, matcher: FrameMatcher, video_path: str | Path):
        super().__init__()
        self.matcher = matcher
        self.video_path = Path(video_path).resolve().absolute()
        self.jump_size = 10

        self.video_path_ref = Path(matcher.video_source).resolve().absolute()

        non_existing_paths = [p for p in [self.video_path, self.video_path_ref] if not Path(p).exists()]
        if non_existing_paths:
            missing = ', '.join(str(p) for p in non_existing_paths)
            raise FileNotFoundError(f"Video file not found: {missing}")

        self.current_frame = 0
        self.is_playing = False

        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.init_ui()
        self.init_menu_bar()
        self.setup_worker()

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        open_videos_action = QAction("Specify Video Inputs...", self)
        open_videos_action.triggered.connect(self._show_video_input_info)
        file_menu.addAction(open_videos_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")
        theme_menu = settings_menu.addMenu("Theme")

        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        dark_action = QAction("Dark", self)
        dark_action.setCheckable(True)
        dark_action.setChecked(True)  # Default to dark
        dark_action.triggered.connect(lambda: self._set_theme("dark"))
        theme_menu.addAction(dark_action)
        theme_group.addAction(dark_action)

        light_action = QAction("Light", self)
        light_action.setCheckable(True)
        light_action.triggered.connect(lambda: self._set_theme("light"))
        theme_menu.addAction(light_action)
        theme_group.addAction(light_action)

    def _set_theme(self, theme_name):
        app = QApplication.instance()
        if theme_name == "dark":
            app.setStyleSheet(DARK_STYLESHEET)
        else:
            app.setStyleSheet("")  # Use default platform style

    def _show_video_input_info(self):
        QMessageBox.information(
            self,
            "Specify Video Inputs",
            "This feature is under development.\\n\\n"
            "For now, please specify the video inputs via the command line when launching the application.",
            QMessageBox.StandardButton.Ok
        )

    def setup_worker(self):
        self.thread = QThread()
        self.worker = FrameProcessorWorker(self.matcher, self.video_path, self.video_path_ref)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.setup_captures)
        self.worker.frame_processed.connect(self.update_ui_from_worker)
        self.worker.ready.connect(self.update_frames)
        self.request_frame.connect(self.worker.process_frame)

        self.thread.start()

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
        display_layout.setSpacing(0)
        display_layout.setContentsMargins(0, 0, 0, 0)

        # Comparing video layout
        comparing_video_layout = QVBoxLayout()
        comparing_header = QLabel("Comparing Video")
        comparing_header.setProperty("level", "h3")
        comparing_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_label = ZoomableGraphicsView()
        comparing_video_layout.addWidget(comparing_header)
        comparing_video_layout.addWidget(self.current_label)
        display_layout.addLayout(comparing_video_layout)

        # Reference video layout
        reference_video_layout = QVBoxLayout()
        reference_header = QLabel("Reference Video (Index)")
        reference_header.setProperty("level", "h3")
        reference_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_label = ZoomableGraphicsView()
        reference_video_layout.addWidget(reference_header)
        reference_video_layout.addWidget(self.match_label)
        display_layout.addLayout(reference_video_layout)
        self.view_stack.addWidget(normal_view_widget)

        # Page 2: Descriptor view
        descriptor_view_widget = QWidget()
        descriptor_layout = QHBoxLayout(descriptor_view_widget)
        self.descriptor_label = ZoomableGraphicsView()
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
        self.play_button.setObjectName("PlayButton")
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
        self.next_button.clicked.connect(self.next_frame_manual)
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
        vis_options_layout = QHBoxLayout()
        vis_options_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_options_layout.addWidget(QLabel("View Mode:"))
        self.visualization_mode_combo = QComboBox()
        self.visualization_mode_combo.addItems([
            "Images Only",
            "Show Matches",
            "Show Unmatched Descriptors"
        ])
        self.visualization_mode_combo.currentIndexChanged.connect(self.update_frames)
        vis_options_layout.addWidget(self.visualization_mode_combo)
        layout.addLayout(vis_options_layout)

        # Setup playback timer
        self.playback_timer = QTimer()
        self.playback_timer.setSingleShot(True)
        self.playback_timer.timeout.connect(self.next_frame_playback)

        # Update slider max width on resize
        self.resizeEvent = self._on_resize

    def update_frames(self):
        self.request_frame.emit(self.current_frame, self.visualization_mode_combo.currentText())

    @pyqtSlot(int, str, object, object, str)
    def update_ui_from_worker(self, frame_index, mode, img1_data, img2_data, info_text):
        self.match_info_label.setText(info_text)

        def to_pixmap(bgr_image):
            if bgr_image is None: return QPixmap()
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(q_img)

        if mode == "Show Matches":
            self.view_stack.setCurrentIndex(1)
            self.descriptor_label.setImage(to_pixmap(img1_data))
        else:  # Images Only and Show Unmatched
            self.view_stack.setCurrentIndex(0)
            self.current_label.setImage(to_pixmap(img1_data))
            self.match_label.setImage(to_pixmap(img2_data))

        # Update frame slider
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)

        # Continue playback if active
        if self.is_playing and self.current_frame < self.total_frames - 1:
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.playback_timer.start(interval)
        elif self.is_playing:  # Reached end of video
            self.stop_playback()

    def stop_playback(self):
        self.is_playing = False
        self.play_button.setText('▶️')
        self.playback_timer.stop()

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText('⏸️' if self.is_playing else '▶️')
        if self.is_playing:
            self.next_frame_playback()  # Kick off the playback loop
        else:
            self.playback_timer.stop()

    def faster_speed(self):
        self.jump_size += 10
        self.speed_label.setText(f'Skip: {self.jump_size} frames')

    def slower_speed(self):
        if self.jump_size > 1:
            self.jump_size = max(1, self.jump_size - 10)
        self.speed_label.setText(f'Skip: {self.jump_size} frames')

    def next_frame_playback(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += self.jump_size
            self.current_frame = min(self.current_frame, self.total_frames - 1)
            self.update_frames()

    def next_frame_manual(self):
        self.stop_playback()
        self.next_frame_playback()

    def prev_frame(self):
        self.stop_playback()
        if self.current_frame > 0:
            self.current_frame -= self.jump_size
            self.current_frame = max(0, self.current_frame)
            self.update_frames()

    def set_frame(self, index: int):
        if self.is_playing:
            self.stop_playback()
        self.current_frame = index
        self.update_frames()

    def _on_resize(self, event):
        # Update slider max width to 50% of window width
        self.frame_slider.setMaximumWidth(int(self.width() * 0.5))
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


def visualize_matches(matcher: FrameMatcher, video_path: str | Path):
    """
    Launch the match visualizer application.

    Args:
        matcher: FrameMatcher instance with built index
        video_path: Path to the video file to analyze
    """
    app = QApplication.instance()  # Should already exist
    if app and not app.styleSheet():
        app.setStyleSheet(DARK_STYLESHEET)

    visualizer = MatchVisualizer(matcher, video_path)
    visualizer.show()


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Visualize frame matches')
    parser.add_argument('video', help='Path to video file to analyze')
    parser.add_argument('matcher', help='Path to matcher file (serialized FrameMatcher)')

    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET) # Set dark theme by default
    matcher = FrameMatcher.deserialize(args.matcher)
    visualize_matches(matcher, args.video)
    sys.exit(app.exec())
