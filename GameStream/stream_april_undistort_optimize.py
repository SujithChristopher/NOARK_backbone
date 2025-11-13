import numpy as np
import cv2
from cv2 import aruco
import socket
import toml
import os
import keyboard
from numba import njit

try:
    import libcamera
    from picamera2 import Picamera2

    WEBCAM = False
except ImportError:
    WEBCAM = True

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 2
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

# Define board once globally
markerLength = 0.05
markerSeparation = 0.01
board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeparation,
    dictionary=ARUCO_DICT,
)

frame_size = (1200, 800)

# Load fisheye params once globally
_fish_params = toml.load("/home/sujith/Documents/programs/undistort_best.toml")
_fish_matrix = np.array(_fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
_fish_dist = np.array(_fish_params["calibration"]["dist_coeffs"])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, frame_size, cv2.CV_16SC2
)


class ExponentialMovingAverageFilter3D:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema_x = self.ema_y = self.ema_z = None

    def update(self, ema):
        if self.ema_x is None:
            self.ema_x, self.ema_y, self.ema_z = ema
        else:
            self.ema_x = self.alpha * ema[0] + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * ema[1] + (1 - self.alpha) * self.ema_y
            self.ema_z = self.alpha * ema[2] + (1 - self.alpha) * self.ema_z
        return np.array([self.ema_x, self.ema_y, self.ema_z])


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
    rvecs, tvecs = [], []
    for corner in corners:
        _, r, t = cv2.solvePnP(
            marker_points,
            corner,
            camera_matrix,
            distortion_coefficients,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if r is not None and t is not None:
            rvecs.append(r.flatten())
            tvecs.append(t.flatten())
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)


class MainClass:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.UDP_STREAM = udp_stream
        self.camera_matrix, self.distortion_coeff = self.load_camera_calibration(
            cam_calib_path
        )
        self.filter = ExponentialMovingAverageFilter3D(alpha=0.4)
        self.setup_camera()
        self.first_frame = True

        if self.UDP_STREAM:
            self.setup_udp()

    def setup_camera(self):
        if not WEBCAM:
            self.picam2 = Picamera2()
            WIDTH, HEIGHT = frame_size
            main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
            config = self.picam2.create_video_configuration(
                main,
                controls={"FrameRate": 100, "ExposureTime": 5000},
                transform=libcamera.Transform(vflip=1),
            )
            self.picam2.configure(config)
            self.picam2.start()
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

    def load_camera_calibration(self, path):
        data = toml.load(path)
        camera_matrix = np.array(data["calibration"]["camera_matrix"]).reshape(3, 3)
        distortion_coeff = np.array(data["calibration"]["dist_coeffs"])
        return camera_matrix, distortion_coeff

    def setup_udp(self):
        udp_ip = "localhost"
        udp_port = 8000
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((udp_ip, udp_port))
        self.udp_socket.settimeout(5)
        print(f"UDP Bound to {udp_ip}:{udp_port}")

    def process_frame(self, frame):
        corners, ids, rejected = detector.detectMarkers(frame)
        # print(corners)
        corners, ids, rejected, _ = detector.refineDetectedMarkers(
            image=frame,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected,
        )
        return corners, ids

    def camera_thread(self):
        while True:
            frame = self.get_frame()
            corners, ids = self.process_frame(frame)

            if ids is not None and len(ids) > 0:
                rvec, tvec = estimate_pose_single_markers(
                    corners, markerLength, self.camera_matrix, self.distortion_coeff
                )
                self.draw_axes(frame, rvec, tvec)
                self.filter_and_send_position(ids, tvec)

            # cv2.imshow("frame", frame)

            if keyboard.is_pressed("q"):
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def get_frame(self):
        if not WEBCAM:
            frame = self.picam2.capture_array()
            frame = frame[: frame_size[1], : frame_size[0]]
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            _, frame = self.camera.read()
        return cv2.flip(frame, 1)

    def draw_axes(self, frame, rvec, tvec):
        for r, t in zip(rvec, tvec):
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.distortion_coeff, r, t, 0.05
            )

    def filter_and_send_position(self, ids, tvec):
        filtered_pos = self.filter.update(np.median(tvec, axis=0))
        # Perform UDP send if needed
        if self.UDP_STREAM:
            message = ",".join(map(str, filtered_pos))
            self.udp_socket.sendto(message.encode("utf-8"), self.addr)

    def run(self):
        self.camera_thread()


if __name__ == "__main__":
    UDP_STREAM = False
    CAMERA_CALIBRATION_FILE = (
        "/home/sujith/Documents/programs/calib_undistort_aruco.toml"
    )
    main = MainClass(cam_calib_path=CAMERA_CALIBRATION_FILE, udp_stream=UDP_STREAM)
    main.run()
