import numpy as np
import cv2
from cv2 import aruco
import platform
import toml
import os
from datetime import datetime
import matplotlib.pyplot as plt


class Config:

    FRAME_SIZE = (1200, 800)
    MARKER_LENGTH = 0.049
    MARKER_SEPARATION = 0.01
    DEFAULT_IDS = [12, 14, 20]
   
    MARKER_OFFSETS = {
        12: np.array([0, 0, -0.057]),
        14: np.array([-0.126, 0, -0.054]),
        # 20: np.array([0, 0, 0])
        20: np.array([0.126, 0, -0.054]),
        # 20: np.array([0.054, 0, -0.126])
    }

    # MARKER_OFFSETS = {
    #     4: np.array([0.00, 0.1, -0.069]),
    #     8: np.array([0.00, 0.01, -0.069]),
    #     12: np.array([0.00, 0.0, -0.1075]),
    #     14: np.array([-0.09, 0.0, -0.069]),
    #     20: np.array([0.1, 0.0, -0.069]),
    # }

class MainClass:
 
 def __init__(self,cam_calib_path,table_calib_path):

    self.default_ids = Config.DEFAULT_IDS
    self.frame_size = Config.FRAME_SIZE
    self.marker_length = Config.MARKER_LENGTH
    self.marker_separation = Config.MARKER_SEPARATION


    # Load calibration parameters 
    calib_data = toml.load(cam_calib_path)
    self.camera_matrix = np.array(
        calib_data["calibration"]["camera_matrix"]
    ).reshape(3, 3)
    dist = np.array(calib_data["calibration"]["dist_coeffs"]).flatten()
    self.distortion_coeff = dist[:4].reshape(4, 1)
    # For Table frame extrinsics
    table_calib_data = toml.load(table_calib_path)
    self.table_rotation_matrix = np.array(
        table_calib_data["Calibration"]["Rotation_matrix"]).reshape(3, 3)
    self.table_translation_vector = np.array(
        table_calib_data["Calibration"]["Translation_vector_1"])   


    self.detector = self._init_detector()
    self.board = self._init_board()

    self.final_pos_cm = None
    self.true_value = np.array([0.0, 0.0, 0.0])  # in meters
    self.noark_in_table_frame = None
    self.picam2, self.map1, self.map2 = None, None, None
    self.video_frame = None
    self.tvec_dist = np.zeros(3)
    self.first_frame = True
    self.save_path = None
    self.csv_writer = None
    self.record = False

    self.received_message = ""    
    self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')
    self.plot_initialized = False

    self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        self.camera_matrix,
        self.distortion_coeff, 
        self.frame_size, 
        np.eye(3), 
        balance=1
    )
    # print("camera_matrix:\n", self.camera_matrix)
    # print("distortion_coeff:\n", self.distortion_coeff)
    # print("new_camera_matrix:\n", self.new_camera_matrix)
    # print("dist_coeff shape:", self.distortion_coeff.shape)  # must be (4,1)
    if platform.system() == "Linux":
        self._init_rpi_camera()

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
        fish_params = toml.load("/home/sujith/Documents/NOARK_backbone/calibration_toml/old/calib_mono_faith3D.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
        fish_dist = np.array(fish_params["calibration"]["dist_coeffs"]).flatten()[:4].reshape(4, 1)
        # self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
        #     fish_matrix,
        #     fish_dist,
        #     np.eye(3),
        #     fish_matrix,
        #     self.frame_size,
        #     cv2.CV_16SC2,
        # )
        

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
                self.new_camera_matrix,
                np.zeros(4),  # Use zero distortion for pose estimation
                flags=cv2.SOLVEPNP_ITERATIVE, 
                # flags=cv2.SOLVEPNP_IPPE_SQUARE
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
            self.distortion_coeff,  # Use zero distortion for drawing axes
            rvec,
            tvec,
            0.049,
        )

 def _get_centroid(self,ids,rvecs, tvecs):
    ids = np.array(ids).flatten()
    tvecs = np.array(tvecs).reshape(len(ids),3)
    rvecs = np.array(rvecs).reshape(len(ids),3)

    _transformed = np.full((len(ids),3), np.nan)
    for index, _id in enumerate(ids):
        match np.array(_id):
            case 12:
                _transformed[index] = (
                    cv2.Rodrigues(rvecs[index])[0]
                    @Config.MARKER_OFFSETS[12].reshape(3,1)
                    + tvecs[index].reshape(3,1)
                ).T[0]
                # print(f"Marker ID: {_id}, Transformed position: {_transformed[index] * 100} cm")
            case 14:
                _transformed[index] = (
                    cv2.Rodrigues(rvecs[index])[0]
                    @ Config.MARKER_OFFSETS[14].reshape(3, 1)
                    + tvecs[index].reshape(3, 1)
                ).T[0]
                # print(f"Marker ID: {_id}, Transformed position: {_transformed[index] * 100} cm")
            case 20:
                _transformed[index] = (
                    cv2.Rodrigues(rvecs[index])[0]
                    @ Config.MARKER_OFFSETS[20].reshape(3, 1)
                    + tvecs[index].reshape(3, 1)
                ).T[0]
                # print(f"Marker ID: {_id}, Transformed position: {_transformed[index] * 100} cm")
    # print(f"Transformed positions: {_transformed * 100} cm id {ids}")
    return np.nanmean(_transformed, axis=0).flatten()
 
    
 def _get_local_coordinates(self, ids, rvecs, tvecs):
        
        # print(f"camera frame", self._get_centroid(ids, rvecs, tvecs) * 100)

        # Marker calibration to set origin at table frame
        # pos_in_table_frame =  self.table_rotation_matrix.T @ self.table_translation_vector + self.table_translation_vector
        # measured_value = pos_in_table_frame
        # offset_value = measured_value - self.true_value  # Marker offset value


        # NOARK IN TABLE FRAME CALCULATION
        # _local_camera_t = self.table_rotation_matrix.T @ (self._get_centroid(ids, rvecs, tvecs) - self.table_translation_vector).reshape(3,1).T[0]
        # _local_camera_t_o = self.table_rotation_matrix.T @ self._get_centroid(ids, rvecs, tvecs) + self.table_translation_vector
        _local_camera_t_o = self.table_rotation_matrix.T @ (self._get_centroid(ids, rvecs, tvecs) - self.table_translation_vector)
        # _local_camera_t = _local_camera_t_o - offset_value   
        return (np.round(_local_camera_t_o, 3))

 def process_frame(self):
   
    ret = None
    if platform.system() == "Linux":
        self.video_frame = self.picam2.capture_array()[:800, :1200]
        self.video_frame = cv2.flip(self.video_frame, 1)
        # print(self.video_frame.size)
    else:
        ret, self.video_frame = self.camera.read()

    if self.video_frame is None or ret is False:
        return
     # Visualize undistortion on the full frame
    # test_undistorted = cv2.fisheye.undistortImage(
    # self.video_frame, self.camera_matrix, self.distortion_coeff, Knew=self.new_camera_matrix
    # )
    # cv2.imshow("undistorted", test_undistorted)
    # cv2.imshow("original", self.video_frame)

    corners, ids, rejected = self.detector.detectMarkers(self.video_frame)

    if ids is not None:
        self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)

        undistorted_corners = [
            cv2.fisheye.undistortPoints(
                corner,
                self.camera_matrix,
                self.distortion_coeff,
                None,
                P=self.new_camera_matrix
            )
            for corner in corners
        ]

        rvecs, tvecs = self.estimate_pose(undistorted_corners)
        print(f"Camera frame (undistorted): {self._get_centroid(ids, rvecs, tvecs) * 100} cm")
        self.noark_in_table_frame = self._get_local_coordinates(ids, rvecs, tvecs)
        print(f"Local Coordinates: {self.noark_in_table_frame * 100} cm")
        self._draw_axes(rvecs, tvecs)
        
    
    self.video_frame = cv2.resize(self.video_frame, (350, 200))
    cv2.imshow("frame", self.video_frame)
    cv2.waitKey(1)
    return True
 def run(self):
    while True:
        self.process_frame()
        # self.update_plot()


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if platform.system() == "Linux":
        CAMERA_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/calibration_toml/old/calib_mono_faith3D.toml"
        TABLE_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/vaideesh/table_frame_data.toml"
    
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH,table_calib_path=TABLE_CALIB_PATH)
    main.run()
