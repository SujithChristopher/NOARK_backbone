import numpy as np
import cv2
from cv2 import aruco
import platform
import toml


class Config:
    FRAME_SIZE = (1280, 800)
    MARKER_LENGTH = 0.069
    MARKER_SEPARATION = 0.01
    TARGET_ID = 1


class MainClass:

    def __init__(self, cam_calib_path):
        self.frame_size = Config.FRAME_SIZE
        self.marker_length = Config.MARKER_LENGTH
        self.marker_separation = Config.MARKER_SEPARATION

        # Load calibration parameters
        calib_data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(
            calib_data["calibration"]["camera_matrix"]
        ).reshape(3, 3)
        self.distortion_coeff = np.array(
            calib_data["calibration"]["dist_coeffs"]
        ).flatten()[:4].reshape(4, 1)

        # Pre-compute undistortion maps
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
        self.board = self._init_board()
        self.picam2 = None
        self.video_frame = None

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

    def estimate_pose(self, corner):
        """Estimate pose for a single marker corner. Returns (3,1) float32 arrays."""
        marker_points = np.array(
            [
                [-self.marker_length / 2,  self.marker_length / 2, 0],
                [ self.marker_length / 2,  self.marker_length / 2, 0],
                [ self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0],
            ],
            dtype=np.float32,
        )
        success, rvec, tvec = cv2.solvePnP(
            marker_points,
            corner,
            self.new_camera_matrix,
            None,                         # distortion already removed by remap
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if success:
            return (
                np.array(rvec, dtype=np.float32).reshape(3, 1),
                np.array(tvec, dtype=np.float32).reshape(3, 1),
            )
        return None, None

    def _draw_axes(self, rvec, tvec):
        """Draw axes for a single marker. rvec and tvec must be (3,1) float32."""
        cv2.drawFrameAxes(
            self.video_frame,
            self.new_camera_matrix,
            None,
            rvec,
            tvec,
            0.069,
        )

    def process_frame(self):
        if platform.system() == "Linux":
            self.video_frame = self.picam2.capture_array()[:800, :1280]
            self.video_frame = cv2.flip(self.video_frame, 1)
            self.video_frame = cv2.remap(
                self.video_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
            )
            # drawFrameAxes requires a BGR color frame
            self.video_frame = cv2.cvtColor(self.video_frame, cv2.COLOR_GRAY2BGR)

        if self.video_frame is None:
            return True

        corners, ids, _ = self.detector.detectMarkers(self.video_frame)

        if ids is not None and len(ids) > 0:
            ids_flat = np.array(ids).flatten()

            if Config.TARGET_ID in ids_flat:
                idx = np.where(ids_flat == Config.TARGET_ID)[0][0]
                target_corner = corners[idx]  # shape (1, 4, 2)

                rvec, tvec = self.estimate_pose(target_corner)

                if rvec is not None:
                    print(f"Marker {Config.TARGET_ID} | "
                          f"tvec (cm): {np.round(tvec.flatten() * 100, 2)} | "
                          f"rvec: {np.round(rvec.flatten(), 3)}")

                    extrinsic_data = {
                        "Calibration": {
                            "aruco": self.marker_length,
                            "Translation_vector_1": tvec.flatten().tolist(),
                            "Rotation_vector_1": rvec.flatten().tolist(),
                        }
                    }
                    with open("vaideesh/table_frame_data.toml", "w") as toml_file:
                        toml.dump(extrinsic_data, toml_file)

                    self._draw_axes(rvec, tvec)

        self.video_frame = cv2.resize(self.video_frame, (350, 200))
        cv2.imshow("frame", self.video_frame)
        cv2.waitKey(1)
        return True

    def run(self):
        while True:
            self.process_frame()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if platform.system() == "Linux":
        CAMERA_CALIB_PATH = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
    else:
        CAMERA_CALIB_PATH = r"E:\CMC\pyprojects\programs_rpi\rpi_python\webcam_calib.toml"

    main = MainClass(cam_calib_path=CAMERA_CALIB_PATH)
    main.run()