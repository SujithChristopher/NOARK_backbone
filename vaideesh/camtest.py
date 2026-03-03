import numpy as np
import cv2
from cv2 import aruco
import platform
import toml
import os
from datetime import datetime
import matplotlib.pyplot as plt


class MainClass:
 
 def __init__(self,cam_calib_path,table_calib_path):



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
        table_calib_data["Calibration"]["Rotation_matrix"]) 
    self.table_translation_vector = np.array(
        table_calib_data["Calibration"]["Translation_vector_1"])   



    self.frame_size = [1280, 800]
    self.final_pos_cm = None
    self.true_value = np.array([0.0, 0.0, 0.0])  # in meters
    self.noark_in_table_frame = None
    self.picam2, self.map1, self.map2 = None, None, None

    self.received_message = ""    
    self._curr_session = os.path.join('Session-' + datetime.today().strftime('%Y-%m-%d'), 'MovementData')
    self.plot_initialized = False

    self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        self.camera_matrix,
        self.distortion_coeff, 
        self.frame_size, 
        np.eye(3), 
        balance=0
    )
    print("camera_matrix:\n", self.camera_matrix)
    print("distortion_coeff:\n", self.distortion_coeff)
    print("new_camera_matrix:\n", self.new_camera_matrix)
    print("dist_coeff shape:", self.distortion_coeff.shape)  # must be (4,1)
    if platform.system() == "Linux":
        self._init_rpi_camera()


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


   
 def _draw_axes(self, rvecs, tvecs):
    for rvec, tvec in zip(rvecs, tvecs):
        cv2.drawFrameAxes(
            self.video_frame,
            self.new_camera_matrix,
            np.zeros(4),  # Use zero distortion for drawing axes
            rvec,
            tvec,
            0.048,
        )

 
 def process_frame(self):
   
    ret = None
    if platform.system() == "Linux":
        self.video_frame = self.picam2.capture_array()[:800, :1280]
        self.video_frame = cv2.flip(self.video_frame, 1)
        print(self.video_frame.size)


    if self.video_frame is None or ret is False:
        return
     # Visualize undistortion on the full frame
    # test_undistorted = cv2.fisheye.undistortImage(
    # self.video_frame, self.camera_matrix, self.distortion_coeff, Knew=self.new_camera_matrix
    # )
    # cv2.imshow("undistorted", test_undistorted)
    # cv2.imshow("original", self.video_frame)

    
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
        CAMERA_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/old_calibration/fisheye.toml"
        TABLE_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/vaideesh/table_frame_data.toml"
    
    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH,table_calib_path=TABLE_CALIB_PATH)
    main.run()
