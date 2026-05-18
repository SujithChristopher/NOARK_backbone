"""
charuco_apriltag_tracker.py

ChArUco board = world origin.
AprilTag poses are printed in ChArUco (world) frame.
"""

import os, sys, argparse
import cv2
import numpy as np
import toml
from picamera2 import Picamera2
import libcamera
import keyboard

WIDTH  = 1280
HEIGHT = 800

APRIL_MARKER_LENGTH = 0.05   # metres — match your physical tag


class CharucoAprilTagTracker:

    def __init__(self, config_path,
                 squares_x=4, squares_y=3,
                 square_length=0.037, marker_length=0.027,
                 april_marker_length=APRIL_MARKER_LENGTH):

        self.squares_x        = squares_x
        self.squares_y        = squares_y
        self.square_length    = square_length
        self.marker_length    = marker_length
        self.april_marker_len = april_marker_length

        self._load_config(config_path)
        self._init_camera()
        self._init_undistortion_maps()
        self._init_charuco_detector()
        self._init_april_detector()

        self.charuco_R    = None   # (3,3) — set once ChArUco is locked
        self.charuco_tvec = None   # (3,)

    # ── config ─────────────────────────────────────────────────────────────────

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        cfg = toml.load(config_path)
        self.camera_matrix = np.array(cfg["calibration"]["camera_matrix"])
        self.dist_coeffs   = np.array(cfg["calibration"]["dist_coeffs"])
        self.resolution    = tuple(cfg["camera"]["resolution"])
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = self.resolution

    # ── camera ─────────────────────────────────────────────────────────────────

    def _init_camera(self):
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": (WIDTH, HEIGHT)},
            controls={"FrameRate": 100, "ExposureTime": 9000},
            transform=libcamera.Transform(vflip=1))
        self.picam2.configure(cfg)
        self.picam2.start()
        print("Camera started.")

    # ── undistortion ───────────────────────────────────────────────────────────

    def _init_undistortion_maps(self):
        self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, self.dist_coeffs, self.resolution, np.eye(3), balance=1.0)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, np.eye(3),
            self.new_camera_matrix, self.resolution, cv2.CV_16SC2)

    # ── detectors ──────────────────────────────────────────────────────────────

    def _init_charuco_detector(self):
        """Mirrors your working ChArUco code exactly."""
        self.charuco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length, self.marker_length, self.charuco_dict)
        self.charuco_aruco_detector = cv2.aruco.ArucoDetector(
            self.charuco_dict, cv2.aruco.DetectorParameters())
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)

    def _init_april_detector(self):
        """Separate detector using a completely different dictionary."""
        april_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        april_params = cv2.aruco.DetectorParameters()
        # same defaults as your original april tag code
        april_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.april_detector = cv2.aruco.ArucoDetector(april_dict, april_params)

        half = self.april_marker_len / 2.0
        self.april_obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("Running — press 'q' to quit.\n")

        while True:
            # ── capture & undistort (exactly as your ChArUco code) ─────────────
            frame = self.picam2.capture_array()
            gray  = cv2.flip(frame[:HEIGHT, :WIDTH], 1)
            undist = cv2.remap(gray, self.map1, self.map2, cv2.INTER_LINEAR)

            display = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)

            # ── 1. ChArUco detection (mirrors your working code) ───────────────
            charuco_ids_aruco, charuco_corners_aruco, _ = \
                self.charuco_aruco_detector.detectMarkers(undist)

            board_locked = False
            if charuco_ids_aruco is not None and len(charuco_ids_aruco) > 0:
                charuco_corners, charuco_ids, _, _ = \
                    self.charuco_detector.detectBoard(undist)

                if (charuco_corners is not None and
                        charuco_ids is not None and
                        len(charuco_ids) >= 4):

                    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.charuco_board,
                        self.new_camera_matrix, np.zeros((4, 1)), None, None)

                    if ok:
                        board_locked      = True
                        self.charuco_R, _ = cv2.Rodrigues(rvec)
                        self.charuco_tvec = tvec.flatten()

            # ── 2. AprilTag detection — completely independent call ─────────────
            april_corners, april_ids, _ = self.april_detector.detectMarkers(undist)

            if april_ids is not None and len(april_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, april_corners, april_ids)

                if board_locked:
                    for corner, tag_id in zip(april_corners, april_ids.flatten()):
                        ok, rvec_a, tvec_a = cv2.solvePnP(
                            self.april_obj_pts, corner,
                            self.new_camera_matrix, None,  # already undistorted
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)

                        if ok:
                            t_tag   = tvec_a.flatten()
                            t_world = self.charuco_R.T @ (t_tag - self.charuco_tvec)
                            print(
                                f"Tag {tag_id:3d} | "
                                f"X={t_world[0]:+.4f}  "
                                f"Y={t_world[1]:+.4f}  "
                                f"Z={t_world[2]:+.4f}  m"
                            )
                else:
                    print("[waiting for ChArUco board...]")

            # ── display ────────────────────────────────────────────────────────
            cv2.putText(display,
                        "ChArUco: LOCKED" if board_locked else "ChArUco: searching...",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if board_locked else (0, 0, 200), 2)
            cv2.imshow("Tracker", cv2.resize(display, (480, 320)))
            cv2.waitKey(1)

            if keyboard.is_pressed("q"):
                print("Exiting...")
                break

        self.picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--squares_x",  type=int,   default=4)
    parser.add_argument("--squares_y",  type=int,   default=3)
    parser.add_argument("--square_len", type=float, default=0.037)
    parser.add_argument("--marker_len", type=float, default=0.027)
    parser.add_argument("--april_len",  type=float, default=0.05)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"Error: config not found → {config_path}")
        sys.exit(1)

    tracker = CharucoAprilTagTracker(
        config_path,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_len,
        marker_length=args.marker_len,
        april_marker_length=args.april_len,
    )
    tracker.run()
