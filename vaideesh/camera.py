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
    MARKER_LENGTH = 0.05
    MARKER_SEPARATION = 0.01
    DEFAULT_IDS = [4, 8, 12]
    # MARKER_OFFSETS = {
    #     4: np.array([0.00, 0.1, -0.069]),
    #     8: np.array([0.00, 0.01, -0.069]),
    #     12: np.array([0.00, 0.0, -0.1075]),
    #     14: np.array([-0.09, 0.0, -0.069]),
    #     20: np.array([0.1, 0.0, -0.069]),
    # }
    MARKER_OFFSETS = {
        4: np.array([0, 0, -0.55]),
        8: np.array([0.0, 0, -0.126]),
        12: np.array([0.0, 0, -0.126]),
    }

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
    self.distortion_coeff = np.array(calib_data["calibration"]["dist_coeffs"])

    # For Table frame extrinsics
    table_calib_data = toml.load(table_calib_path)
    self.table_rotation_matrix = np.array(
        table_calib_data["Calibration"]["Rotation_matrix"]) 
    self.table_translation_vector = np.array(
        table_calib_data["Calibration"]["Translation_vector"])   
    self.table_extrinsic_matrix = np.array(
        table_calib_data["Calibration"]["Extrinsic_Matrix"])


    self.detector = self._init_detector()
    self.board = self._init_board()

    self.NOARK_dist = np.array([0.0, 0.0,-0.055])  # in meters 
    self.true_value = np.array([0.0, 0.0, 0.0])  # in meters
    self.final_pos_cm = None
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

    if platform.system() == "Linux":
        self._init_rpi_camera()
    else:
        self._init_camera()

 def _init_plot(self):

    plt.ion()
    self.fig, self.ax = plt.subplots()
    self.x_data, self.y_data = [], []
    self.line = self.ax.plot(self.x_data, self.y_data, 'b-')
    self.ax.set_xlim(-100, 100)
    self.ax.set_ylim(-50, 50)
    self.ax.set_xlabel('X(cm)')
    self.ax.set_ylabel('Y(cm)')
    self.ax.set_title('NOARK + Position')
    self.plot_initialized = True


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
 def _get_centroid(self,ids,rvecs, tvecs):
    ids = np.array(ids).flatten()
    tvecs = np.array(tvecs).reshape(len(ids),3)
    rvecs = np.array(rvecs).reshape(len(ids),3)

    _transformed = np.full((len(ids),3), np.nan)
    for index, _id in enumerate(ids):
        match np.array(_id):
            case 4:
                _transformed[index] = (
                    cv2.Rodrigues(rvecs[index])[0]
                    @Config.MARKER_OFFSETS[4].reshape(3,1)
                    + tvecs[index].reshape(3,1)
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
    # print(f"Marker ID: {_id}, Transformed position: {_transformed[index] * 100} cm")
    return np.nanmean(_transformed, axis=0).flatten()
 
    
 def _get_local_coordinates(self, ids, rvecs, tvecs):


         # Marker calibration to set origin at table frame
        pos_in_table_frame =  self.table_rotation_matrix.T @ self.table_translation_vector + self.table_translation_vector
        measured_value = pos_in_table_frame
        offset_value = measured_value - self.true_value  # Marker offset value
        # print(pos_in_table_frame) 
        # NOARK IN TABLE FRAME
        # _local_camera_t = self.table_rotation_matrix.T @ self._get_centroid(ids, rvecs, tvecs) +self.table_translation_vector
        _local_camera_t = self.table_rotation_matrix.T @ (self._get_centroid(ids, rvecs, tvecs) - self.table_translation_vector).reshape(3,1).T[0]
        # print(f"NOARK in Table Frame: {_local_camera_t * 100} cm")
        # _local_coordinates = _local_camera_t - offset_value
        return np.round(_local_camera_t, 3)

 


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
        if ids is not None:
            self.video_frame = aruco.drawDetectedMarkers(self.video_frame, corners, ids)
            rvecs, tvecs = self.estimate_pose(corners)

            # _centroid = self._get_centroid(ids, rvecs, tvecs)
            _local_coordinates = self._get_local_coordinates(ids, rvecs, tvecs)
            # print(f"Local Coordinates: {_local_coordinates * 100} cm")







        if ids is not None and len(ids) > 0:
            rvecs, tvecs = self.estimate_pose(corners)
            ids = np.array(ids).flatten()
            tvecs = np.array(tvecs).reshape(len(ids), 3)
            rvecs = np.array(rvecs).reshape(len(ids), 3)
          
            self.rotation_matrices = []
            for rvec in rvecs:
                R, _ = cv2.Rodrigues(rvec) # convert to 3x3 rotation matrix
                self.rotation_matrices.append(R)

            # Calculate NOARK position in camera frame
            for i, R in enumerate(self.rotation_matrices):
                final_pos = R @ self.NOARK_dist + tvecs[i]
                self.final_pos = np.round(final_pos, 3)
                self.final_pos[np.abs(self.final_pos) < 0.001] = 0.00

                np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
                print(self.final_pos * 100)

                # Marker calibration to set origin at table frame
                pos_in_table_frame = self.table_rotation_matrix .T @ self.table_translation_vector + self.table_translation_vector
                measured_value = pos_in_table_frame
                offset_value = measured_value - self.true_value  # Marker offset value
                # print(pos_in_table_frame)

                # Final noark position in table frame
                noark_in_table = self.table_rotation_matrix .T @ self.final_pos + self.table_translation_vector
                position_corrected = noark_in_table - offset_value
                self.noark_in_table_frame = np.round(position_corrected, 3)
               
               
                # print(self.noark_in_table_frame * 100)
        
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
        TABLE_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/table_frame_data.toml"
    else:
        CAMERA_CALIB_PATH = r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"
    
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH,table_calib_path=TABLE_CALIB_PATH)
    main.run()
