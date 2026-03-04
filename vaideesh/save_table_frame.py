import numpy as np
import cv2
from cv2 import aruco
import platform
import toml
import os
from datetime import datetime
import matplotlib.pyplot as plt


class Config:

    FRAME_SIZE = (1280, 800)
    MARKER_LENGTH = 0.07
    MARKER_SEPARATION = 0.01
    DEFAULT_IDS = [1,2,3]

class MainClass:
 
 def __init__(self,cam_calib_path):

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

    self.final_pos_cm = None
    self.table_frame_pos = None
    self.table_frame_rot = None

    self.picam2, self.map1, self.map2 = None, None, None
    self.video_frame = None
    self.tvec_dist = np.zeros(3)
    self.first_frame = True
    self.save_path = None
    self.csv_writer = None
    self.record = False
    self.v1_v2 = 0.2515
    self.v3_v2 = 0.25

    self.received_message = ""    
    self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')
    self.plot_initialized = False

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
        fish_params = toml.load("/home/sujith/Documents/NOARK_backbone/calibration_toml/old/calib_mono_faith2D.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
        fish_dist = np.array(fish_params["calibration"]["dist_coeffs"])

        print(fish_matrix, fish_dist, type(fish_matrix))
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            fish_matrix,
            fish_dist,
            np.eye(3),
            fish_matrix,
            self.frame_size,
            cv2.CV_16SC2,
        )
        

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
            0.07,
        )
 def process_frame(self):
    ret = None
    if platform.system() == "Linux":
        self.video_frame = self.picam2.capture_array()[:800, :1280]
        self.video_frame = cv2.flip(self.video_frame, 1)
        self.video_frame = cv2.remap(
            self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
        )

    if self.video_frame is None or ret is False:
        return

    corners, ids, rejected = self.detector.detectMarkers(self.video_frame)

    if ids is not None and len(ids) > 0:
        rvecs, tvecs = self.estimate_pose(corners)
        ids = np.array(ids).flatten()
        tvecs = np.array(tvecs).reshape(len(ids), 3)
        rvecs = np.array(rvecs).reshape(len(ids), 3)

        # Store tvecs by marker ID
        tvec_dict = {marker_id: tvecs[i] for i, marker_id in enumerate(ids)}
        print(f"tvec_dict: {tvec_dict}")

        # Only compute rotation matrix if all 3 markers are detected
        if 1 in tvec_dict and 2 in tvec_dict and 3 in tvec_dict:
            org = tvec_dict[2]   # origin
            v1  = tvec_dict[1] - org # x direction
            v2  = tvec_dict[3] - org  # z direction

             # Distance calculations
            dist_1_to_2 = np.linalg.norm(tvec_dict[1] - tvec_dict[2]) * 100  # in cm
            dist_3_to_2 = np.linalg.norm(tvec_dict[3] - tvec_dict[2]) * 100  # in cm
            print(f"Distance marker 1 to 2: {dist_1_to_2:.2f} cm")
            print(f"Distance marker 3 to 2: {dist_3_to_2:.2f} cm") 

            # Gram-Schmidt orthogonalization
            vxnorm = v1 / np.linalg.norm(v1)                         # x axis

            vzcap = v2 - np.dot(v2, vxnorm) * vxnorm                                                
            vznorm = vzcap / np.linalg.norm(vzcap)
            
            vynorm = np.cross(vznorm, vxnorm)                       # y axis
            R = np.hstack((vxnorm, vynorm, vznorm))                     # 3x3 rotation matrix

            print(f"Origin (marker 2): {org * 100} cm")
            print(f"Rotation matrix:\n{R}")
        # if self.v1_v2 == 0.2515 and self.v3_v2 == 0.25:
            extrinsic_data = {
                "Calibration": {
                    "Rotation_matrix": R.tolist(),
                    "aruco": self.marker_length,
                    "Translation_vector_1": tvec_dict[1].tolist(),
                    "Translation_vector_2": tvec_dict[2].tolist(),
                    "Translation_vector_3": tvec_dict[3].tolist(),

                    "Rotation_vector_1": rvecs[ids == 1][0].tolist(),
                    "Rotation_vector_2": rvecs[ids == 2][0].tolist(),
                    "Rotation_vector_3": rvecs[ids == 3][0].tolist()
                }
            }

            with open("vaideesh/table_frame_data.toml", "w") as toml_file:
                toml.dump(extrinsic_data, toml_file)

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
        CAMERA_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/calibration_toml/old/calib_mono_faith2D.toml"
    else:
        CAMERA_CALIB_PATH = r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"
    
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH)
    main.run()
