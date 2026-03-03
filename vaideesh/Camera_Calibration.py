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
    MARKER_LENGTH = 0.07
    MARKER_SEPARATION = 0.01
    DEFAULT_IDS = []
    MARKER_OFFSETS = {
        1: np.array([0.00,0.0,0.0])
        # 4: np.array([0.00, 0.1, -0.069]),
        # 8: np.array([0.00, 0.01, -0.069]),
        # 12: np.array([0.00, 0.0, -0.1075]),
        # 14: np.array([-0.09, 0.0, -0.069]),
        # 20: np.array([0.1, 0.0, -0.069]),
    }

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

    self.NOARK_dist = np.array([0.0, 0.0,0.12])  # in meters 
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

    self.received_message = ""    
    self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')
    self.plot_initialized = False

    if platform.system() == "Linux":
        self._init_rpi_camera()
    else:
        self._init_camera()

#  def _init_plot(self):

#     plt.ion()
#     self.fig, self.ax = plt.subplots()
#     self.x_data, self.y_data = [], []
#     self.line = self.ax.plot(self.x_data, self.y_data, 'b-')
#     self.ax.set_xlim(-100, 100)
#     self.ax.set_ylim(-50, 50)
#     self.ax.set_xlabel('X(cm)')
#     self.ax.set_ylabel('Y(cm)')
#     self.ax.set_title('Table Frame')
#     self.plot_initialized = True


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
            controls={"FrameRate": 60, "ExposureTime": 5000},
            transform=libcamera.Transform(vflip=1),

        )
        self.picam2.configure(config)
        self.picam2.start()

        # Load fisheye calibration
        fish_params = toml.load("/home/sujith/Documents/rpi_python/undistort_best.toml")
        fish_matrix = np.array(fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
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
        self.camera.set(cv2.CAP_PROP_FPS, 60)

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

#  def update_plot(self):
#     if not self.plot_initialized:
#         self._init_plot()

#     self.x_data.append(self.tvec_dist[0])  
#     self.y_data.append(self.tvec_dist[1])   

#     max_points = 100
#     if len(self.x_data) > max_points:
#         self.x_data.pop(0)
#         self.y_data.pop(0)
#     self.line.set_data(self.x_data, self.y_data)
#     self.ax.relim()
#     self.ax.autoscale_view()
#     self.fig.canvas.draw()
#     self.fig.canvas.flush_events()
    # plt.pause(0.01)   

 def process_frame(self):
        # Capture frames
        ret = None
        if platform.system() == "Linux":
            self.video_frame = self.picam2.capture_array()
            # self.video_frame = cv2.flip(self.video_frame, 0)
            self.video_frame = cv2.remap(
                 self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
            )
            self.video_frame = cv2.flip(self.video_frame, 1)
        else:
            ret, self.video_frame = self.camera.read()

        if self.video_frame is None or ret is False:
            return
        
        corners, ids, rejected = self.detector.detectMarkers(self.video_frame)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs = self.estimate_pose(corners)
            ids = np.array(ids).flatten()
            tvecs = np.array(tvecs).reshape(len(ids), 3)
            rvecs = np.array(rvecs).reshape(len(ids), 3)
          
            self.rotation_matrices = []
            for rvec in rvecs:
                R, _ = cv2.Rodrigues(rvec) # convert to 3x3 rotation matrix
                self.rotation_matrices.append(R)
                
            # compute transformed position for each marker.This willl return the position of the cable joint
            # for i, R in enumerate(self.rotation_matrices):
                # final_pos = 
                # final_pos = R @ self.NOARK_dist + tvecs[i]
                # self.final_pos_cm = final_pos * 100
                self.table_frame_pos = tvecs[0]   # in m
                v1 = self.rotation_matrices[0][:,0]
                v2 = self.rotation_matrices[0][:,1]
                v3 = self.rotation_matrices[0][:,2]
                # print("v1:", v1)
                # print("v2:", v2)
                # print("v3:", v3)
                
                # print(
                #     #  f"Marker {ids[i]} → Final Pos: "
                #     f"X: {self.table_frame_pos[0]:.3f} cm, "
                #     f"Y: {self.table_frame_pos[1]:.3f} cm, "
                #     f"Z: {self.table_frame_pos[2]:.3f} cm")


                # To check orthogonality
                # Trans_v1 = v1.T
                # Ortho_check_1 = np.dot(Trans_v1, v3)
                # print("Dot Product v1.v2 (should be close to 0):", Ortho_check_1)

                # 3x4 extrinsic matrix
                t = self.table_frame_pos.reshape(3,1)
                self.extrinsic_matrix = np.hstack((self.rotation_matrices[0], t))
                # print("Extrinsic Matrix:\n", self.extrinsic_matrix)

                # # Storing it in Toml file
                # extrinsic_data = {"Extrinsic_Matrix": self.extrinsic_matrix.tolist()}
                extrinsic_data = {
                    "Calibration":{
                        "Extrinsic_Matrix": self.extrinsic_matrix.tolist(),
                        "Rotation_matrix": self.rotation_matrices[0].tolist(),
                        "Translation_vector": self.table_frame_pos.tolist(),
                        "aruco": self.marker_length,
                    }
                }

                with open("table_frame_extrinsics.toml", "w") as toml_file:
                    toml.dump(extrinsic_data, toml_file) 
        

            self._draw_axes(rvecs, tvecs)

            # Print X,Y,Z coordinates
            # for i, marker_id in enumerate(ids):
            #     # X, Y, Z = tvecs[i] * 100  # Convert to cm
            #     X,Y,Z = tvecs[i] 
                # self.tvec_dist = np.array([X, Y, Z]) 
                # print(f"Marker ID: {marker_id} - X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m")

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
        CAMERA_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/old_calibration/optimized_fisheye_160.toml"
    else:
        CAMERA_CALIB_PATH = r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"
    
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH)
    main.run()
