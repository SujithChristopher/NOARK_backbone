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
# import keyboard
class Config:
    FRAME_SIZE = (1280, 800)
    MARKER_LENGTH = 0.05
    DEFAULT_IDS = [12, 14, 20]
    MARKER_OFFSETS = {
        12: np.array([0, 0, -0.055]),
        #12: np.array([0, 0, 0.0]),
        14: np.array([-0.126, 0, -0.054]),
        20: np.array([0.126, 0, -0.054]),
    }

class MainClass:
    def __init__(self, cam_calib_path, table_calib_path):
        # 1. Load Calibration Data (Camera Intrinsics + Table Reference Frame)
        calib_data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(calib_data["calibration"]["camera_matrix"]).reshape(3, 3)
        self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"]).flatten()[:4].reshape(4, 1)

        table_calib_data = toml.load(table_calib_path)
        self.table_rotation_matrix = np.array(table_calib_data["rotation_matrix"]).reshape(3, 3)
        self.table_translation_vector = np.array(table_calib_data["tvec"]).reshape(3, 1)

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
        self.noark_in_table_frame = None
        self.video_frame = None

        if platform.system() == "Linux":
            self._init_rpi_camera()

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

    def _get_centroid(self, ids, rvecs, tvecs):
        """Calculates the handle of the object by applying marker-specific offsets."""
        ids = np.array(ids).flatten()
        rvecs = np.array(rvecs).reshape(-1, 3)
        tvecs = np.array(tvecs).reshape(-1, 3)
        
        transformed_points = []
        for i, m_id in enumerate(ids):
            if m_id in Config.MARKER_OFFSETS:
                R_mat, _ = cv2.Rodrigues(rvecs[i])
                offset = Config.MARKER_OFFSETS[m_id].reshape(3, 1)
                # Apply Rotation to Offset + Translation
                pos_cam = R_mat @ offset + tvecs[i].reshape(3, 1)
                transformed_points.append(pos_cam.flatten())

        if not transformed_points:
            return None
        
        # Returns (3,1) to match table_translation_vector shape
        return np.mean(transformed_points, axis=0).reshape(3, 1)

    def _get_local_coordinates(self, ids, rvecs, tvecs):
        centroid_cam = self._get_centroid(ids, rvecs, tvecs)
        if centroid_cam is None: return None
        # P_table = R_table.T * (P_camera - T_table)
        relative_pos = centroid_cam - self.table_translation_vector
        p_table = self.table_rotation_matrix.T @ relative_pos
        return p_table.flatten()
    def process_frame(self):
        if platform.system() == "Linux":
            # Direct Y-channel extraction (Grayscale) for speed
            raw_data = self.picam2.capture_array()
            gray = raw_data[:Config.FRAME_SIZE[1], :Config.FRAME_SIZE[0]]
            gray = cv2.flip(gray, 1)
        else:
            ret, frame = self.camera.read()
            if not ret: return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Efficiency: Use pre-computed maps to rectify fisheye distortion
        undistorted = cv2.remap(gray, self.map1, self.map2, cv2.INTER_LINEAR)
        
        corners, ids, _ = self.detector.detectMarkers(undistorted)
        self.video_frame = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
        # --- Initialize per-marker storage (None if marker not detected) ---
        if ids is not None:
            aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)

    
            # print(self.marker_12)
            self.noark_in_table_frame = self._get_local_coordinates(ids, rvecs, tvecs)
            
            if self.noark_in_table_frame is not None:
                # print(f"Table Frame (cm): {np.round(self.noark_in_table_frame * 100, 3)}")
                # Optional: draw axes for the first marker
                cv2.drawFrameAxes(
                    self.video_frame, 
                    self.new_camera_matrix, 
                    None, 
                    rvecs[0], 
                    tvecs[0], 
                    0.05)
            # --- PRINT RAW TVECS AT THE END OF PROCESSING ---
            # print("\n--- Raw Camera Frame Detections ---")
            # for i, marker_id in enumerate(ids.flatten()):
            #     # raw_tvec is the [x, y, z] in meters from the camera lens center
            #     raw_tvec_cm = tvecs[i] * 100 
            #     raw_rvec_cm = rvecs[i] * 100
            #     print(f"Marker ID {marker_id}: {np.round(raw_tvec_cm, 2)} cm")
            #     print(f"Marker ID {marker_id}: {np.round(raw_rvec_cm, 2)} cm")
            
            if self.noark_in_table_frame is not None:
                print(f"NOARK in Table Frame: {np.round(self.noark_in_table_frame * 100, 2)} cm")
        cv_show = cv2.resize(self.video_frame, (480, 320))
        cv2.imshow("Optimized Tracker", cv_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
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
        default="/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose_picam.toml"
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