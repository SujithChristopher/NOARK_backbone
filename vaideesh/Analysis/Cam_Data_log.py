import csv
import gpiod
import numpy as np
import cv2
from cv2 import aruco
import platform
import toml
import os
from datetime import datetime
import time
import argparse 
import sys
import keyboard
class Config:
    FRAME_SIZE = (1280, 800)
    MARKER_LENGTH = 0.049
    DEFAULT_IDS = [12, 14, 20]

class MainClass:
    def __init__(self, cam_calib_path, table_calib_path):
        # 1. Load Calibration Data (Camera Intrinsics + Table Reference Frame)
        calib_data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(calib_data["calibration"]["camera_matrix"]).reshape(3, 3)
        self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"]).flatten()[:4].reshape(4, 1)

        # 2. Pre-compute Undistortion Maps 
        # Using balance=1.0 to retain full FOV for the 160-degree lens
        self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, 
            self.distortion_coeff, 
            Config.FRAME_SIZE, 
            np.eye(3), 
            balance=1.0
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, 
            self.distortion_coeff, 
            np.eye(3), 
            self.new_camera_matrix, 
            Config.FRAME_SIZE, 
            cv2.CV_16SC2
        )
        self.detector = self._init_detector()
        self.video_frame = None
        self.trigger_cam = False

          # Sync Pin Setup
        sync_pin = 17
        chip = gpiod.Chip("gpiochip0")
        self.sync_line = chip.get_line(sync_pin)
        self.sync_line.request(consumer="SyncPin", type=gpiod.LINE_REQ_DIR_IN)

        
        self.csv_path = "camera_data.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            "timestamp", "sync_pin",
            "tvec_12_x", "tvec_12_y", "tvec_12_z",
            "tvec_14_x", "tvec_14_y", "tvec_14_z",
            "tvec_20_x", "tvec_20_y", "tvec_20_z",
            "rvec_12_x", "rvec_12_y", "rvec_12_z",
            "rvec_14_x", "rvec_14_y", "rvec_14_z",
            "rvec_20_x", "rvec_20_y", "rvec_20_z"   
        ])
      

        if platform.system() == "Linux":
            self._init_rpi_camera()
        self._frame_count = 0
        self._fps_timer = time.time()
    def _init_detector(self):
        aruco_params = aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        return aruco.ArucoDetector(aruco_dict, aruco_params)

    def _init_rpi_camera(self):
        from picamera2 import Picamera2
        import libcamera
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": Config.FRAME_SIZE},
            controls={"FrameRate": 100, "ExposureTime": 3000},
            transform=libcamera.Transform(vflip=1)
        )
        self.picam2.configure(config)
        self.picam2.start()

    def estimate_pose(self, corners):
        """Standard PnP using IPPE_SQUARE for better stability."""
        marker_points = np.array([
            [-Config.MARKER_LENGTH / 2,  Config.MARKER_LENGTH / 2, 0],
            [ Config.MARKER_LENGTH / 2,  Config.MARKER_LENGTH / 2, 0],
            [ Config.MARKER_LENGTH / 2, -Config.MARKER_LENGTH / 2, 0],
            [-Config.MARKER_LENGTH / 2, -Config.MARKER_LENGTH / 2, 0],
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for corner in corners:
            # We use None for distCoeffs because the image is already undistorted via remap
            success, rvec, tvec = cv2.solvePnP(
                marker_points, 
                corner, 
                self.new_camera_matrix, 
                None, 
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
        return np.array(rvecs), np.array(tvecs)
    def _close_files(self):
        print("CSV saved and closed.")
    
    def _write_frame(self, timestamp):
        sync = self.sync_line.get_value()

        row = [timestamp, sync]

      # Extract Translation Vectors (tvecs)
        for m in [self.marker_12, self.marker_14, self.marker_20]:
            if m["tvec"] is not None:
                row.extend([round(float(v), 6) for v in m["tvec"]])
            else:
                row.extend([float("nan")] * 3)
        # Extract Rotation Vectors (rvecs)
        for m in [self.marker_12, self.marker_14, self.marker_20]:
            if m["rvec"] is not None:
                row.extend([round(float(v), 6) for v in m["rvec"]])
            else:
                row.extend([float("nan")] * 3)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    def process_frame(self):
        if platform.system() == "Linux":
            # Direct Y-channel extraction (Grayscale) for speed
            raw_data = self.picam2.capture_array()
            gray = raw_data[:Config.FRAME_SIZE[1], :Config.FRAME_SIZE[0]]
            gray = cv2.flip(gray, 1)

        # Efficiency: Use pre-computed maps to rectify fisheye distortion
        undistorted = cv2.remap(gray, self.map1, self.map2, cv2.INTER_LINEAR)
        corners, ids, _ = self.detector.detectMarkers(undistorted)
        self.video_frame = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
        # --- Initialize per-marker storage (None if marker not detected) ---
        self.marker_12 = {"id": 12, "tvec": None, "rvec": None}
        self.marker_14 = {"id": 14, "tvec": None, "rvec": None}
        self.marker_20 = {"id": 20, "tvec": None, "rvec": None}

        marker_map = {
            12: self.marker_12,
            14: self.marker_14,
            20: self.marker_20,
        }

        if ids is not None:
           
            # aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in marker_map:
                    marker_map[marker_id]["tvec"] = tvecs[i]
                    marker_map[marker_id]["rvec"] = rvecs[i]
            # print(f"Marker 12: tvec={self.marker_12['tvec']}, rvec={self.marker_12['rvec']}")

        if self.trigger_cam:    
            self._write_frame(timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                
        # cv_show = cv2.resize(self.video_frame, (480, 320))
        # cv2.imshow("Camera", cv_show)
        # cv2.pollKey()
      #FPS Calculation (every 10 frames)
        # self._frame_count += 1
        # if self._frame_count % 10 == 0:
        #     elapsed = time.time() - self._fps_timer
        #     fps = 10 / elapsed
        #     self._fps_timer = time.time()
        #     print(f"FPS: {fps:.2f}")
        if keyboard.is_pressed('q'):
           cv2.destroyAllWindows()
           return False
        return True
    
    def run(self):
        while self.process_frame():
            pass
        self.picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description="Realtime Fisheye Tracker with Marker Offsets")
    
    # 2. Add arguments for the TOML paths
    parser.add_argument(
        "-c", "--camera", 
        help="Path to camera calibration TOML", 
        default="/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
    )
    parser.add_argument(
        "-t", "--table", 
        help="Path to table pose TOML", 
        default="/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"
    )

    args = parser.parse_args()

    # 3. Resolve absolute paths to prevent "File Not Found" errors
    cam_path = os.path.abspath(args.camera)
    table_path = os.path.abspath(args.table)

    # 4. Verify files exist before starting
    if not os.path.exists(cam_path) or not os.path.exists(table_path):
        print(f"Error: One of the configuration files was not found.\nCam: {cam_path}\nTable: {table_path}")
        sys.exit(1)

    # 5. Initialize and Run
    main = MainClass(cam_calib_path=cam_path, table_calib_path=table_path)
    main.run()