import numpy as np
import cv2
from cv2 import aruco
from picamera2 import Picamera2
import toml
import libcamera
from threading import Thread
import time
import numpy as np
import toml

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 2
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeparation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeparation,
    dictionary=ARUCO_DICT,
)

frame_size = (1200, 800)


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
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if r is not None and t is not None:
            rvecs.append(r.reshape(1, 3).tolist())
            tvecs.append(t.reshape(1, 3).tolist())
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)


_fish_params = toml.load("/home/sujith/Documents/programs/undistort_best.toml")
_fish_matrix = np.array(_fish_params["calibration"]["camera_matrix"]).reshape(3, 3)
_fish_dist = np.array(_fish_params["calibration"]["dist_coeffs"])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    _fish_matrix, _fish_dist, np.eye(3), _fish_matrix, frame_size, cv2.CV_16SC2
)


class OptimizeLighting:
    def __init__(self, cam_calib_path, udp_stream=False):
        self.UDP_STREAM = udp_stream
        self.picam2 = Picamera2()
        self.config = self.picam2.create_video_configuration(
            {"format": "YUV420", "size": frame_size},
            controls={"FrameRate": 100, "ExposureTime": 1000},
            transform=libcamera.Transform(vflip=1),
        )

        self.picam2.configure(self.config)
        self.picam2.start()

        data = toml.load(cam_calib_path)
        self.camera_matrix = np.array(data["calibration"]["camera_matrix"]).reshape(
            3, 3
        )
        self.distortion_coeff = np.array(data["calibration"]["dist_coeffs"])
        self.parameter_scan = {
            "ExposureTime": np.arange(1000, 2000, 500)
        }  # 1000 to 6500 with increments of 500

        self.collect_data = {
            "Corners": [],
            "CornerCounter": [],
            "Ids": [],
            "ExposureTime": [],
            "ExIndex": [],
            "FrameCounter": [],
            "Tvec": [],
            "Rvec": [],
        }

        self.frame_counter = 0
        self.reset_fcounter = True
        self.exposure_trigger = True
        self.exposure_idx = 0
        self.frame_exposure_t = 0
        self.current_exposure = None
        self.stop_recording = False
        self.corner_counter = 0

        self.default_ids = [4, 8, 12, 14, 20]

    def camera_thread(self):
        print("ExposureTime: ", 1000)
        while True:
            video_frame = self.picam2.capture_array()[: frame_size[1], : frame_size[0]]
            self.frame_exposure_t = self.picam2.capture_metadata()["ExposureTime"]
            video_frame = cv2.remap(
                cv2.flip(video_frame, 1),
                map1,
                map2,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )

            corners, ids, rejected = detector.detectMarkers(video_frame)
            corners, ids, rejected, _ = detector.refineDetectedMarkers(
                image=video_frame,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected,
            )

            rvec, tvec = estimate_pose_single_markers(
                corners, 0.05, self.camera_matrix, self.distortion_coeff
            )
            shp = tvec.shape
            if (ids is not None) and all(
                item in self.default_ids for item in np.array(ids)
            ):
                self.collect_data["Corners"].append(corners)
                self.collect_data["Ids"].append(ids)

                self.collect_data["ExposureTime"].append(0)
                self.collect_data["ExIndex"].append(self.current_exposure)
                self.collect_data["FrameCounter"].append(0)
                self.collect_data["Tvec"].append(tvec)
                self.collect_data["Rvec"].append(rvec)
                self.corner_counter += 1

            else:
                self.collect_data["Corners"].append(None)
                self.collect_data["Ids"].append(None)
                self.collect_data["ExposureTime"].append(0)
                self.collect_data["ExIndex"].append(self.current_exposure)
                self.collect_data["FrameCounter"].append(0)
                self.collect_data["Tvec"].append(np.zeros(shp))
                self.collect_data["Rvec"].append(np.eye(3))

                self.reset_fcounter = False
            self.frame_counter += 1

            if self.frame_counter == 100:
                self.collect_data["CornerCounter"].append(self.corner_counter)
                self.corner_counter = 0

                self.exposure_idx += 1
                self.frame_counter = 0
                if self.exposure_idx == len(self.parameter_scan["ExposureTime"]):
                    self.stop_recording = True
                    self.picam2.close()
                    break
                print(
                    "ExposureTime: ",
                    self.parameter_scan["ExposureTime"][self.exposure_idx],
                )

                self.picam2.set_controls(
                    {
                        "ExposureTime": self.parameter_scan["ExposureTime"][
                            self.exposure_idx
                        ]
                    }
                )

            if self.stop_recording:
                self.picam2.close()
                print("Stopping recording")
                break
            cv2.imshow("frame", cv2.resize(video_frame, (350, 200)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            # print(video_frame.shape)
        self.process_recorded()

    def process_recorded(self):
        ex_id = np.argmax(self.collect_data["CornerCounter"])

        print("Frames detected in each exposure interval")
        print(self.collect_data["CornerCounter"])
        print("argsort value", np.argsort(self.collect_data["CornerCounter"]))
        print("Exposure index: ", np.argmax(self.collect_data["CornerCounter"]))
        print("Exposure time: ", self.parameter_scan["ExposureTime"][ex_id])

        tvec = np.array(
            self.collect_data["Tvec"],
        )

        tvec = tvec.reshape(tvec.shape[0], 3)
        norms = []
        print("shape")
        print(tvec.shape)

        # Iterate in steps of 500
        for i in range(0, tvec.shape[0], 500):
            chunk = tvec[i : i + 500]
            norm = np.linalg.norm(
                chunk, axis=1
            )  # Calculate the norm for each row in the chunk
            norms.append(norm)
            print(norm)
            print("asdf")
        # print(norms)
        print("Minimum SD of Tvec")
        pass

    def run(self):
        t1 = Thread(target=self.camera_thread)
        t1.start()
        t1.join()


if __name__ == "__main__":
    CAMERA_CALIBRATION_FILE = "/home/sujith/Documents/programs/calib_mono_faith3D.toml"
    main = OptimizeLighting(cam_calib_path=CAMERA_CALIBRATION_FILE)
    main.run()
