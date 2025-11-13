"""
PySide6 GUI application for OV9281 chessboard calibration recording.

Features:
- Real-time video feed with chessboard corner detection
- 2D visualization of detected corners coverage area
- Record corners to msgpack format
- GPIO sync support (pin 17)
- Adjustable FPS (30-40 range)
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
from collections import deque

try:
    import msgpack as mp
    import msgpack_numpy as mpn
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QFileDialog, QFrame,
    QGroupBox, QGridLayout, QProgressBar, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect, QPoint
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PySide6.QtGui import QBrush

try:
    from picamera2 import Picamera2
    import libcamera
    PICAMERA2_AVAILABLE = True
except (ImportError, ValueError) as e:
    # ValueError can occur from numpy version mismatch
    PICAMERA2_AVAILABLE = False
    PICAMERA2_ERROR = str(e)

try:
    import gpiod
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class CameraThread(QThread):
    """Separate thread for camera capture to keep UI responsive."""

    frame_ready = Signal(np.ndarray, bool)  # frame, has_corners
    corners_detected = Signal(np.ndarray)   # corners array
    fps_info = Signal(float)                # current FPS
    error = Signal(str)                     # error message

    def __init__(self, target_fps=30, use_gpio=False):
        super().__init__()
        self.target_fps = target_fps
        self.use_gpio = use_gpio and GPIO_AVAILABLE
        self.running = False
        self.frame_count = 0
        self.last_fps_time = 0

        # Camera parameters
        self.FRAME_SIZE = (1280, 800)
        self.CHESSBOARD_SIZE = (6, 4)

        # For GPIO sync
        self.sync_line = None
        if self.use_gpio and GPIO_AVAILABLE:
            try:
                chip = gpiod.Chip("gpiochip4")
                self.sync_line = chip.get_line(17)
                self.sync_line.request(consumer="ChessboardRecorder", type=gpiod.LINE_REQ_DIR_IN)
            except Exception as e:
                self.use_gpio = False

    def run(self):
        """Main camera capture loop."""
        if not PICAMERA2_AVAILABLE:
            self.error.emit("picamera2 not available. Check environment.")
            return

        try:
            # Initialize camera
            picam2 = Picamera2()
            main_config = {
                "format": "YUV420",  # OV9281 native format
                "size": self.FRAME_SIZE
            }
            controls = {"FrameRate": self.target_fps}

            config = picam2.create_video_configuration(
                main_config,
                controls=controls,
                transform=libcamera.Transform(vflip=1)  # Flip for correct orientation
            )
            picam2.configure(config)
            picam2.start()

            self.running = True
            import time
            frame_time = 1.0 / self.target_fps
            last_frame_time = time.time()

            while self.running:
                current_time = time.time()

                # Capture frame
                frame = picam2.capture_array()
                if frame is None:
                    continue

                # Convert YUV420 to grayscale (Y channel is monochrome)
                h, w = self.FRAME_SIZE[1], self.FRAME_SIZE[0]
                if frame.shape[0] == h * 3 // 2:  # YUV420 format
                    frame = frame[:h]  # Extract Y channel (grayscale)

                # Flip frame
                frame = cv2.flip(frame, 1)

                # Detect chessboard corners
                ret, corners = cv2.findChessboardCorners(frame, self.CHESSBOARD_SIZE)

                if ret:
                    # Refine corners
                    corners = cv2.cornerSubPix(
                        frame,
                        corners,
                        (5, 5),
                        (-1, -1),
                        criteria=(
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001
                        ),
                    )
                    self.corners_detected.emit(corners)

                # Emit frame
                self.frame_ready.emit(frame, ret)
                self.frame_count += 1

                # Calculate FPS
                if current_time - self.last_fps_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_fps_time)
                    self.fps_info.emit(fps)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # Frame rate control
                elapsed = time.time() - current_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            picam2.stop()
            picam2.close()

        except Exception as e:
            self.error.emit(f"Camera error: {str(e)}")

    def stop(self):
        """Stop camera thread."""
        self.running = False
        self.wait()

    def set_fps(self, fps):
        """Update target FPS (takes effect on next frame)."""
        self.target_fps = fps


class CornersVisualization(QGraphicsView):
    """2D visualization of detected chessboard corners with coverage area."""

    def __init__(self, frame_size=(1280, 800), chessboard_size=(6, 4)):
        super().__init__()
        self.frame_size = frame_size
        self.chessboard_size = chessboard_size
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")

        # Scale factor to fit in view
        self.scale_x = 300 / frame_size[0]
        self.scale_y = 200 / frame_size[1]

        self.corners = None
        self.coverage_points = []  # Track all detected corners for coverage
        self.recording = False
        self.draw_background()

    def draw_background(self):
        """Draw a light grid representing the camera sensor."""
        self.scene.clear()

        # Draw frame border
        frame_rect = self.scene.addRect(
            0, 0,
            self.frame_size[0] * self.scale_x,
            self.frame_size[1] * self.scale_y
        )
        frame_rect.setPen(QPen(QColor(100, 100, 100), 1))

        # Draw grid
        grid_spacing_x = 128  # Every 128 pixels in sensor coords
        grid_spacing_y = 80

        for x in range(0, self.frame_size[0], grid_spacing_x):
            line = self.scene.addLine(
                x * self.scale_x, 0,
                x * self.scale_x, self.frame_size[1] * self.scale_y
            )
            line.setPen(QPen(QColor(60, 60, 60), 0.5))

        for y in range(0, self.frame_size[1], grid_spacing_y):
            line = self.scene.addLine(
                0, y * self.scale_y,
                self.frame_size[0] * self.scale_x, y * self.scale_y
            )
            line.setPen(QPen(QColor(60, 60, 60), 0.5))

    def update_corners(self, corners):
        """Update visualization with detected corners."""
        self.corners = corners
        self.draw_background()

        # Collect coverage data if recording
        if self.recording and corners is not None and len(corners) > 0:
            for corner in corners:
                self.coverage_points.append(corner[0])

        # Draw coverage area if recording
        if self.recording and len(self.coverage_points) > 3:
            self.draw_coverage_area()

        if corners is not None and len(corners) > 0:
            # Draw corners as points
            for idx, corner in enumerate(corners):
                x, y = corner[0]
                x_scaled = x * self.scale_x
                y_scaled = y * self.scale_y

                # Draw corner circle
                radius = 3
                ellipse = self.scene.addEllipse(
                    x_scaled - radius, y_scaled - radius,
                    radius * 2, radius * 2
                )
                ellipse.setBrush(QBrush(QColor(0, 255, 0)))
                ellipse.setPen(QPen(QColor(0, 200, 0), 1))

                # Draw corner number text (every 2 corners to avoid clutter)
                if idx % 2 == 0:
                    text = self.scene.addText(str(idx))
                    text.setPos(x_scaled + 5, y_scaled - 10)
                    text.setDefaultTextColor(QColor(200, 200, 200))
                    font = text.font()
                    font.setPointSize(6)
                    text.setFont(font)

            # Draw chessboard grid if all corners detected
            if len(corners) == self.chessboard_size[0] * self.chessboard_size[1]:
                # Draw lines connecting corners in grid pattern
                cols, rows = self.chessboard_size
                for i in range(rows):
                    for j in range(cols - 1):
                        idx1 = i * cols + j
                        idx2 = i * cols + j + 1
                        x1, y1 = corners[idx1][0]
                        x2, y2 = corners[idx2][0]

                        line = self.scene.addLine(
                            x1 * self.scale_x, y1 * self.scale_y,
                            x2 * self.scale_x, y2 * self.scale_y
                        )
                        line.setPen(QPen(QColor(0, 150, 255), 0.5))

                for j in range(cols):
                    for i in range(rows - 1):
                        idx1 = i * cols + j
                        idx2 = (i + 1) * cols + j
                        x1, y1 = corners[idx1][0]
                        x2, y2 = corners[idx2][0]

                        line = self.scene.addLine(
                            x1 * self.scale_x, y1 * self.scale_y,
                            x2 * self.scale_x, y2 * self.scale_y
                        )
                        line.setPen(QPen(QColor(0, 150, 255), 0.5))

    def draw_coverage_area(self):
        """Draw the cumulative coverage area as a filled convex hull."""
        if len(self.coverage_points) < 3:
            return

        try:
            from scipy.spatial import ConvexHull
            import numpy as np

            # Convert to numpy array
            points = np.array(self.coverage_points)

            # Get convex hull
            if len(points) >= 3:
                hull = ConvexHull(points)
                vertices = hull.vertices

                # Build polygon path
                polygon_points = []
                for idx in vertices:
                    x, y = points[idx]
                    x_scaled = x * self.scale_x
                    y_scaled = y * self.scale_y
                    polygon_points.append((x_scaled, y_scaled))

                if len(polygon_points) >= 3:
                    # Close the polygon
                    polygon_points.append(polygon_points[0])

                    # Draw filled polygon
                    from PySide6.QtCore import QPointF
                    qpoints = [QPointF(x, y) for x, y in polygon_points]

                    # Create polygon item
                    from PySide6.QtGui import QPolygonF
                    polygon = QPolygonF(qpoints)
                    poly_item = self.scene.addPolygon(polygon)

                    # Style: semi-transparent fill with orange border
                    poly_item.setBrush(QBrush(QColor(255, 165, 0, 80)))  # Orange with alpha
                    poly_item.setPen(QPen(QColor(255, 140, 0), 2))

        except (ImportError, Exception):
            # Fallback: draw bounding box if scipy not available
            self.draw_coverage_bbox()

    def draw_coverage_bbox(self):
        """Draw bounding box of coverage area (fallback if scipy unavailable)."""
        if len(self.coverage_points) < 1:
            return

        points = np.array(self.coverage_points)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # Scale coordinates
        x1_scaled = min_x * self.scale_x
        y1_scaled = min_y * self.scale_y
        x2_scaled = max_x * self.scale_x
        y2_scaled = max_y * self.scale_y

        # Draw filled rectangle
        width = x2_scaled - x1_scaled
        height = y2_scaled - y1_scaled

        rect = self.scene.addRect(x1_scaled, y1_scaled, width, height)
        rect.setBrush(QBrush(QColor(255, 165, 0, 80)))  # Orange with alpha
        rect.setPen(QPen(QColor(255, 140, 0), 2))

    def set_recording(self, recording):
        """Set recording state and clear coverage if stopping."""
        if not recording and self.recording:
            # Recording stopped, keep the final coverage visualization
            pass
        elif recording and not self.recording:
            # Recording started, clear coverage
            self.coverage_points = []

        self.recording = recording

    def clear_coverage(self):
        """Clear coverage data."""
        self.coverage_points = []


class VideoDisplay(QLabel):
    """Display video frames with optional overlay."""

    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setMinimumSize(640, 480)
        self.frame = None
        self.recording = False
        self.corners_count = 0

    def display_frame(self, frame):
        """Display a frame from numpy array."""
        if frame is None or frame.size == 0:
            return

        self.frame = frame.copy()

        # Convert to RGB if grayscale
        if len(frame.shape) == 2:
            # Grayscale - convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add recording indicator
        if self.recording:
            cv2.putText(
                frame_rgb,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
            # Red circle indicator
            cv2.circle(frame_rgb, (30, 70), 10, (0, 0, 255), -1)

        # Add corners count
        cv2.putText(
            frame_rgb,
            f"Corners: {self.corners_count}",
            (20, frame_rgb.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Convert to QImage and display
        h, w, ch = frame_rgb.shape
        bytes_per_line = 3 * w

        # Ensure C-contiguous layout
        frame_rgb = np.ascontiguousarray(frame_rgb)

        # Create QImage from bytes
        qt_image = QImage(
            frame_rgb.tobytes(),
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        # Convert to pixmap
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fill available space while maintaining aspect ratio
        available_width = self.width()
        available_height = self.height()

        # Use actual available space or defaults
        if available_width <= 1:
            available_width = 640
        if available_height <= 1:
            available_height = 480

        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled = pixmap.scaledToHeight(available_height, Qt.SmoothTransformation)

        # If it's still wider than available, scale down by width instead
        if scaled.width() > available_width:
            scaled = pixmap.scaledToWidth(available_width, Qt.SmoothTransformation)

        self.setPixmap(scaled)
        self.update()  # Force repaint

    def set_recording(self, recording):
        """Update recording state."""
        self.recording = recording

    def set_corners_count(self, count):
        """Update corners count display."""
        self.corners_count = count


class ChessboardRecorderApp(QMainWindow):
    """Main application window for chessboard recording."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chessboard Calibration Recorder")
        self.setGeometry(100, 100, 1400, 900)

        # Recording state
        self.recording = False
        self.save_path = None
        self.msgpack_file = None
        self.corner_history = deque(maxlen=100)  # Keep last 100 detections

        # Camera thread
        self.camera_thread = None

        # Setup UI
        self.setup_ui()
        self.init_camera()

    def setup_ui(self):
        """Setup user interface layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        # Top: Video display
        video_layout = QHBoxLayout()
        self.video_display = VideoDisplay()
        video_layout.addWidget(self.video_display)

        layout.addLayout(video_layout, stretch=2)

        # Middle: Visualization and info
        viz_layout = QHBoxLayout()

        # Left: 2D corner visualization
        self.viz_view = CornersVisualization()
        viz_layout.addWidget(self.viz_view, stretch=1)

        # Right: Info panel
        info_panel = QGroupBox("Recording Info")
        info_layout = QGridLayout()

        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Courier", 10))
        info_layout.addWidget(QLabel("Status:"), 0, 0)
        info_layout.addWidget(self.status_label, 0, 1)

        # FPS display
        self.fps_label = QLabel("FPS: --")
        info_layout.addWidget(QLabel("FPS:"), 1, 0)
        info_layout.addWidget(self.fps_label, 1, 1)

        # Frames recorded
        self.frame_count_label = QLabel("Frames: 0")
        info_layout.addWidget(QLabel("Frames:"), 2, 0)
        info_layout.addWidget(self.frame_count_label, 2, 1)

        # Corners detected
        self.corners_label = QLabel("Corners: 0/24")
        info_layout.addWidget(QLabel("Corners:"), 3, 0)
        info_layout.addWidget(self.corners_label, 3, 1)

        # Add stretch to push content to top
        info_layout.setRowStretch(4, 1)

        info_panel.setLayout(info_layout)
        viz_layout.addWidget(info_panel, stretch=0)

        layout.addLayout(viz_layout, stretch=1)

        # Bottom: Controls
        control_layout = QHBoxLayout()

        # Path selection
        path_layout = QVBoxLayout()
        path_label = QLabel("Save Location:")
        path_label.setFont(QFont("Courier", 9))
        path_layout.addWidget(path_label)

        self.path_display = QLabel("Not selected")
        self.path_display.setStyleSheet("background-color: #2a2a2a; padding: 5px; border: 1px solid #444;")
        self.path_display.setFont(QFont("Courier", 8))
        path_layout.addWidget(self.path_display)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_save_path)
        path_layout.addWidget(browse_btn)

        control_layout.addLayout(path_layout, stretch=1)

        # FPS control
        fps_layout = QVBoxLayout()
        fps_label = QLabel("Target FPS (30-40):")
        fps_layout.addWidget(fps_label)

        fps_spinbox = QSpinBox()
        fps_spinbox.setMinimum(30)
        fps_spinbox.setMaximum(40)
        fps_spinbox.setValue(30)
        fps_spinbox.valueChanged.connect(self.update_fps)
        fps_layout.addWidget(fps_spinbox)

        control_layout.addLayout(fps_layout)

        # Record control
        record_layout = QVBoxLayout()
        record_layout.addWidget(QLabel("Recording:"))

        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Recording")
        self.start_btn.setStyleSheet("background-color: #2d5a2d;")
        self.start_btn.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.setStyleSheet("background-color: #5a2d2d;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        button_layout.addWidget(self.stop_btn)

        record_layout.addLayout(button_layout)
        control_layout.addLayout(record_layout)

        layout.addLayout(control_layout, stretch=0)

        main_widget.setLayout(layout)

    def init_camera(self):
        """Initialize camera thread."""
        if PICAMERA2_AVAILABLE:
            self.camera_thread = CameraThread(target_fps=30, use_gpio=False)
            status = "Camera ready (RPi mode)"
        else:
            # Fallback to demo mode for testing without hardware
            from demo_mode import DemoCameraThread
            self.camera_thread = DemoCameraThread(target_fps=30)
            status = "Demo mode (no camera available)"

        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.corners_detected.connect(self.on_corners_detected)
        self.camera_thread.fps_info.connect(self.on_fps_update)
        if hasattr(self.camera_thread, 'error'):
            self.camera_thread.error.connect(self.on_camera_error)

        self.camera_thread.start()
        self.update_status(status)

    def on_frame_ready(self, frame, has_corners):
        """Handle new frame from camera."""
        self.video_display.display_frame(frame)
        self.video_display.set_recording(self.recording)

        # Record frame if recording
        if self.recording and self.msgpack_file and MSGPACK_AVAILABLE:
            # Record the detected corners if available
            if self.corner_history:
                corners = self.corner_history[-1]
                try:
                    packed = mp.packb(corners, default=mpn.encode)
                    self.msgpack_file.write(packed)
                    self.frame_count_label.setText(
                        f"Frames: {self.msgpack_file.tell() // 100}"
                    )
                except Exception as e:
                    self.update_status(f"Error recording: {e}")

    def on_corners_detected(self, corners):
        """Handle detected chessboard corners."""
        self.corner_history.append(corners)
        self.viz_view.update_corners(corners)
        self.video_display.set_corners_count(len(corners))
        self.corners_label.setText(f"Corners: {len(corners)}/24")

    def on_fps_update(self, fps):
        """Handle FPS update."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_camera_error(self, error_msg):
        """Handle camera errors."""
        self.update_status(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Camera Error", error_msg)

    def browse_save_path(self):
        """Open file dialog to select save location."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder for recordings"
        )
        if path:
            self.save_path = path
            self.path_display.setText(path)
            self.update_status(f"Save path: {path}")

    def update_fps(self, fps):
        """Update target FPS."""
        if self.camera_thread:
            self.camera_thread.set_fps(fps)
            self.update_status(f"FPS set to {fps}")

    def start_recording(self):
        """Start recording chessboard corners."""
        if not self.save_path:
            QMessageBox.warning(self, "Warning", "Please select a save location first.")
            return

        if not MSGPACK_AVAILABLE:
            QMessageBox.warning(
                self,
                "Warning",
                "msgpack not installed. Can record visualization only, not corners data.\n"
                "Install msgpack to save corner data: pip install msgpack msgpack-numpy"
            )

        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chessboard_{timestamp}.msgpack"
            filepath = os.path.join(self.save_path, filename)

            # Create directory if needed
            os.makedirs(self.save_path, exist_ok=True)

            # Open file for writing (only if msgpack available)
            if MSGPACK_AVAILABLE:
                self.msgpack_file = open(filepath, "wb")
            else:
                self.msgpack_file = None

            self.recording = True
            self.corner_history.clear()
            self.viz_view.set_recording(True)  # Start tracking coverage

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            self.update_status(f"Recording to: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording: {e}")
            self.update_status(f"Error: {e}")

    def stop_recording(self):
        """Stop recording chessboard corners."""
        if self.msgpack_file:
            self.msgpack_file.close()
            self.msgpack_file = None

        self.recording = False
        self.viz_view.set_recording(False)  # Stop tracking coverage
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        frames_recorded = len(self.corner_history)
        coverage_points = len(self.viz_view.coverage_points)
        self.update_status(
            f"Recording stopped. {frames_recorded} frames, {coverage_points} coverage points"
        )
        self.frame_count_label.setText(f"Frames: {frames_recorded}")

    def update_status(self, message):
        """Update status label."""
        self.status_label.setText(message)

    def closeEvent(self, event):
        """Handle application close."""
        if self.recording:
            self.stop_recording()

        if self.camera_thread:
            self.camera_thread.stop()

        event.accept()


def main():
    """Run the application."""
    app = QApplication(sys.argv)

    # Check dependencies
    if not PICAMERA2_AVAILABLE:
        import warnings
        warnings.warn(
            "picamera2 not available. Application will run in demo mode."
        )

    window = ChessboardRecorderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
