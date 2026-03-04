import numpy as np
import cv2
from cv2 import aruco
import platform
import socket
import toml
import os
from GameStream.filters import ExponentialMovingAverageFilter3D
import struct
import csv
from datetime import datetime
# from rpy_helper import rtime

class Config:
    FRAME_SIZE = (1280, 800)
    MARKER_LENGTH = 0.05
    MARKER_SEPARATION = 0.01
    UDP_IP = "localhost"
    UDP_PORT = 8000
    DEFAULT_IDS = [4, 8, 12, 14, 20]
    ALPHA = 0.4  # Exponential moving average filter smoothing factor
    MARKER_OFFSETS = {
        4: np.array([-0.054, 0.031, -0.069]),
        8: np.array([0.00, 0.1025, -0.069]),
        12: np.array([0.00, 0.01, -0.069]),
        14: np.array([0.00, 0.031, -0.1075]),
        20: np.array([0.054, 0.031, -0.069]),
    }
    MARKER_OFFSETS = {
        4: np.array([0.00, 0.1, -0.069]),
        8: np.array([0.00, 0.01, -0.069]),
        12: np.array([0.00, 0.0, -0.1075]),
        14: np.array([-0.09, 0.0, -0.069]),
        20: np.array([0.1, 0.0, -0.069]),
    }


class MainClass:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.udp_stream = udp_stream
        self.filter = ExponentialMovingAverageFilter3D(alpha=Config.ALPHA)
        self.default_ids = Config.DEFAULT_IDS
        self.frame_size = Config.FRAME_SIZE
        self.marker_length = Config.MARKER_LENGTH
        self.marker_separation = Config.MARKER_SEPARATION

        # Load calibration parameters
        calib_data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(
            calib_data["calibration"]["camera_matrix"]
        ).reshape(3, 3)
        self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"])

        self.detector = self._init_detector()
        self.board = self._init_board()

        self.picam2, self.map1, self.map2 = None, None, None
        self.video_frame = None
        self.tvec_dist = np.zeros(3)
        self.first_frame = True
        self.save_path = None
        self.csv_writer = None
        self.record = False

        self.received_message = ""
        
        self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')

        if platform.system() == "Linux":
            self._init_rpi_camera()
        else:
            self._init_camera()

        if self.udp_stream:
            self._init_udp_socket()

    def _init_detector(self):
        aruco_params = aruco.DetectorParameters()
        aruco_params.useAruco3Detection = True
        aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        return aruco.ArucoDetector(aruco_dict, aruco_params)

    def _init_board(self):
        return aruco.GridBoard(
            size=(1, 1),
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.detector.getDictionary(),
        )

    def _init_rpi_camera(self):
        from picamera2 import Picamera2
        import libcamera

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": self.frame_size},
            controls={"FrameRate": 100, "ExposureTime": 5000},
            transform=libcamera.Transform(vflip=1),
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Load fisheye calibration
        fish_params = toml.load("/home/sujith/Documents/rpi_python/undistort_best.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(
            3, 3
        )
        fish_dist = np.array(fish_params["calibration"]["dist_coeffs"])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            fish_matrix,
            fish_dist,
            np.eye(3),
            fish_matrix,
            self.frame_size,
            cv2.CV_16SC2,
        )

    def _init_camera(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def _init_udp_socket(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((Config.UDP_IP, Config.UDP_PORT))
        self.udp_socket.setblocking(False)  # This line - make it non-blocking
        print("UDP socket initialized:", self.udp_socket.getsockname())

    def estimate_pose(self, corners):
        marker_points = np.array(
            [
                [-self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, self.marker_length / 2, 0],
                [self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0],
            ],
            dtype=np.float32,
        )

        rvecs, tvecs = [], []
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                corner,
                self.camera_matrix,
                self.distortion_coeff,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
        return np.array(rvecs), np.array(tvecs)

    def _draw_axes(self, rvecs, tvecs):
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(
                self.video_frame,
                self.camera_matrix,
                self.distortion_coeff,
                rvec,
                tvec,
                0.05,
            )

    def _get_centroid(self, ids, rvecs, tvecs):
        ids = np.array(ids).flatten()
        tvecs = np.array(tvecs).reshape(len(ids), 3)
        rvecs = np.array(rvecs).reshape(len(ids), 3)

        _transformed = np.full((len(ids), 3), np.nan)
        for index, _id in enumerate(ids):
            match np.array(_id):
                case 4:
                    _transformed[index] = (
                        cv2.Rodrigues(rvecs[index])[0]
                        @ Config.MARKER_OFFSETS[4].reshape(3, 1)
                        + tvecs[index].reshape(3, 1)
                    ).T[0]
                case 8:
                    _transformed[index] = (
                        cv2.Rodrigues(rvecs[index])[0]
                        @ Config.MARKER_OFFSETS[8].reshape(3, 1)
                        + tvecs[index].reshape(3, 1)
                    ).T[0]
                case 12:
                    _transformed[index] = (
                        cv2.Rodrigues(rvecs[index])[0]
                        @ Config.MARKER_OFFSETS[12].reshape(3, 1)
                        + tvecs[index].reshape(3, 1)
                    ).T[0]
                case 14:
                    _transformed[index] = (
                        cv2.Rodrigues(rvecs[index])[0]
                        @ Config.MARKER_OFFSETS[14].reshape(3, 1)
                        + tvecs[index].reshape(3, 1)
                    ).T[0]
                case 20:
                    _transformed[index] = (
                        cv2.Rodrigues(rvecs[index])[0]
                        @ Config.MARKER_OFFSETS[20].reshape(3, 1)
                        + tvecs[index].reshape(3, 1)
                    ).T[0]

        return np.nanmean(_transformed, axis=0).flatten()

    def _get_local_coordinates(self, first_id, first_rvecs, first_tvecs, centroid):
        
        first_id = np.array(first_id).flatten()
        first_tvecs = np.array(first_tvecs).reshape(len(first_id), 3)
        first_rvecs = np.array(first_rvecs).reshape(len(first_id), 3)

        _id = first_id[0]
        _r = cv2.Rodrigues(first_rvecs[0])[0]
        _t = first_tvecs[0]

        _local_camera_t = _r @ Config.MARKER_OFFSETS[_id].reshape(3, 1) + _t.reshape(
            3, 1
        )

        # _local_camera_t = _r.T @ _t.reshape(3,1)

        _local_camera_t = _local_camera_t.T[0]
        # _local_camera_t = first_tvecs[0].flatten()
        _local_coordinates = _r.T @ (_local_camera_t - centroid).reshape(3, 1)
        return _local_coordinates.T[0]

    def _send_coordinates(self, _message, _transformed):
        match _message:
            case "STOP":
                _msg = -99.0
            case "START":
                _msg = 2.0
            case "RESET":
                _msg = 5.0

        _transformed = np.append(_msg, _transformed).flatten()
        _data_bytes = struct.pack("f" * len(_transformed), *_transformed)
        self.udp_socket.sendto(_data_bytes, self.addr)


    def process_frame(self):
    # Capture frame
        ret = None
        if platform.system() == "Linux":
            self.video_frame = self.picam2.capture_array()
            self.video_frame = cv2.remap(
                 self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
            )
            self.video_frame = cv2.flip(self.video_frame, 1)
        else:
            ret, self.video_frame = self.camera.read()

        if self.video_frame is None or ret is False:
             return

         # Move UDP communication here - BEFORE ArUco detection
        if self.udp_stream:
             try:
                self.received_message, self.addr = self.udp_socket.recvfrom(30)
             except socket.error:
                pass

        corners, ids, _ = self.detector.detectMarkers(self.video_frame)
        if ids is not None:
            self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)
            if self.first_frame:
               self.first_id = ids
               self.first_rvec, self.first_tvec = rvecs, tvecs
               self.first_frame = False
            self._draw_axes(rvecs, tvecs)
            _centroid = self._get_centroid(ids, rvecs, tvecs)
            _local_coordinates = self._get_local_coordinates(
                self.first_id, self.first_rvec, self.first_tvec, _centroid
            )

            _local_coordinates = self.filter.update(_local_coordinates)

        # Handle received messages here
            if hasattr(self, 'received_message') and self.received_message:
                if self.received_message == b"STOP":
                    self._send_coordinates("STOP", _local_coordinates)
                elif self.received_message.startswith(b"USER:"):
                    self._hid = self.received_message.decode("utf-8").split(":")[1]
                    if self.save_path is None:
                        self._select_hospitalid()
                    self._send_coordinates("START", _local_coordinates)
                    self.record = True
                elif self.received_message.startswith(b"CHANGE:"):
                    self.save_path = None
                    self._hid = self.received_message.decode("utf-8").split(":")[1]
                    self._select_hospitalid()
                    self._send_coordinates("START", _local_coordinates)
                    self.record = True
                elif self.received_message == b"RESET":
                    self._send_coordinates("RESET", _local_coordinates)
                else:
                    self._send_coordinates("START", _local_coordinates)

                if self.record:
                    self.csv_writer.writerow([[datetime.now().strftime("%d/%m/%Y %H:%M:%S"),*_local_coordinates]])

        self.video_frame = cv2.resize(self.video_frame, (350, 200))
        cv2.imshow("frame", self.video_frame)

    def _select_hospitalid(self):
        # TODO: Implement hospital id selection
        if self.save_path is None:
            self.save_path = os.path.join(
                os.path.expanduser("~/Documents/NOARK/data"), self._hid, self._curr_session
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            self.csv_writer = csv.writer(
                open(
                    os.path.join(
                        self.save_path,
                        datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_data.csv",
                    ),
                    "w",
                )
            )
            self.csv_writer.writerow(["Time", "X", "Y", "Z"])

    # def run(self):
    #     while True:
    #         self.process_frame()
    #         if self.received_message == b"STOP":
    #             break
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     cv2.destroyAllWindows()

    def run(self):
        import time
        last_heartbeat = time.time()
    
        while True:
            try:
               self.process_frame()
            
            # If we successfully received a message, update heartbeat
               if hasattr(self, 'received_message') and self.received_message:
                 last_heartbeat = time.time()
            
            # If no communication for 3 seconds, Godot probably closed
               if time.time() - last_heartbeat > 3.0:
                   print("Lost connection to Godot, exiting...")
                   break
                
            except:
            # Any network error means Godot is gone
               print("Network error, Godot closed...")
               break
            
            if self.received_message == b"STOP":
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        cv2.destroyAllWindows()


if __name__ == "__main__":
    
    if platform.system() == "Linux":
        CAMERA_CALIB_PATH = "/home/sujith/Documents/rpi_python/old_calibration/calib_mono_faith3D.toml"
    else:
        CAMERA_CALIB_PATH = (
            r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"
        )
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH, udp_stream=True)
    main.run()
