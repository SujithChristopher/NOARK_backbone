import numpy as np
from cv2 import aruco
import cv2
import math
from picamera2 import Picamera2
import libcamera
import threading

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 1
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)

frame_size = (1280, 800)

def estimate_pose_single_markers(
    corners, marker_size, camera_matrix, distortion_coefficients
):
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    for corner in corners:
        _, r, t = cv2.solvePnP(marker_points, corner, camera_matrix, distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
        if r is not None and t is not None:
            r = np.array(r).reshape(1, 3).tolist()
            t = np.array(t).reshape(1, 3).tolist()
            rvecs.append(r) 
            tvecs.append(t)
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)


class RpiCamera:
    def __init__(self, cameraMatrix = np.eye(3), distCoeffs = np.zeros((5, 1)), debug=False):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        
        self.picam2 = Picamera2()
        WIDTH = frame_size[0]
        HEIGHT = frame_size[1]
        main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
        _c = {
            "FrameRate": 100,
            # 'ExposureTime':500
        }
        config = self.picam2.create_video_configuration(
            main, controls=_c, transform=libcamera.Transform(vflip=1)
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.default_ids = [12, 88, 89, 14, 20]
        
        self.FIRST_FRAME = True
        self.debug = debug
        
    def run_detection(self):
        gray = self.video_frame.copy()
        corners, ids, rejected = detector.detectMarkers(gray)
        rvecs, tvecs = estimate_pose_single_markers(
            corners, markerLength, self.cameraMatrix, self.distCoeffs
        )
        self.rvecs = rvecs
        self.tvecs = tvecs
        if self.debug:
            print(self.tvecs)
        
    def start_thread(self): threading.Thread(target=self.run_camera).start()
    
    def get_pose(self): return [self.rvecs, self.tvecs]
    
    def run_camera(self):
        WIDTH = frame_size[0]
        HEIGHT = frame_size[1]
        while True:
            self.video_frame = self.picam2.capture_array()
            self.video_frame = self.video_frame[:HEIGHT, :WIDTH]
            self.run_detection()
        
if __name__ == "__main__":
    RpiCamera(debug=True).start_thread()