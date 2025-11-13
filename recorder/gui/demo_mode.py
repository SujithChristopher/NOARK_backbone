"""
Demo/fallback mode for testing GUI without picamera2.

Generates synthetic frames and corners for development/testing.
"""

import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal
import time


class DemoCameraThread(QThread):
    """Simulated camera thread for testing GUI without RPi hardware."""

    frame_ready = Signal(np.ndarray, bool)
    corners_detected = Signal(np.ndarray)
    fps_info = Signal(float)
    error = Signal(str)

    def __init__(self, target_fps=30):
        super().__init__()
        self.target_fps = target_fps
        self.running = False
        self.frame_count = 0
        self.last_fps_time = 0

        self.FRAME_SIZE = (1280, 800)
        self.CHESSBOARD_SIZE = (6, 4)

    def run(self):
        """Generate synthetic frames and corners."""
        self.running = True
        import time

        frame_time = 1.0 / self.target_fps
        last_frame_time = time.time()
        frame_num = 0

        while self.running:
            current_time = time.time()

            # Generate synthetic frame
            frame = self.generate_frame(frame_num)

            # Simulate corner detection (gradually move corners around)
            corners = self.generate_corners(frame_num)
            ret = True

            if ret:
                self.corners_detected.emit(corners)

            self.frame_ready.emit(frame, ret)
            self.frame_count += 1
            frame_num += 1

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

    def generate_frame(self, frame_num):
        """Generate a synthetic grayscale frame."""
        frame = np.ones((800, 1280), dtype=np.uint8) * 50

        # Add some noise
        noise = np.random.normal(0, 10, (800, 1280)).astype(np.uint8)
        frame = cv2.add(frame, noise)

        # Add a moving gradient
        gradient = np.linspace(50, 100, 1280, dtype=np.uint8)
        frame[:] = np.tile(gradient, (800, 1))

        # Add some circles for visual interest
        angle = (frame_num * 2) % 360
        x = int(640 + 200 * np.cos(np.radians(angle)))
        y = int(400 + 150 * np.sin(np.radians(angle)))
        cv2.circle(frame, (x, y), 50, 200, -1)

        return frame

    def generate_corners(self, frame_num):
        """Generate synthetic chessboard corners with movement."""
        cols, rows = self.CHESSBOARD_SIZE

        # Base grid
        start_x = 200
        start_y = 150
        spacing_x = 150
        spacing_y = 150

        # Add some movement
        offset_x = 20 * np.sin(frame_num * 0.05)
        offset_y = 15 * np.cos(frame_num * 0.05)

        corners = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x + offset_x
                y = start_y + row * spacing_y + offset_y

                # Add small random jitter
                x += np.random.normal(0, 0.5)
                y += np.random.normal(0, 0.5)

                corners.append([[x, y]])

        return np.array(corners, dtype=np.float32)

    def stop(self):
        """Stop demo thread."""
        self.running = False
        self.wait()

    def set_fps(self, fps):
        """Update target FPS."""
        self.target_fps = fps
