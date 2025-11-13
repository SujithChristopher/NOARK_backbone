import cv2
from picamera2 import Picamera2
import numpy as np
from cv2 import aruco
from libcamera import controls
import toml
import sys

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36H12)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)


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
        _, r, t = cv2.solvePnP(
            marker_points,
            corner,
            camera_matrix,
            distortion_coefficients,
            False,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if r is not None and t is not None:
            r = np.array(r).reshape(1, 3).tolist()
            t = np.array(t).reshape(1, 3).tolist()
            rvecs.append(r)
            tvecs.append(t)
    return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)


picam2 = Picamera2()
# main = {"size": (800, 480)}
main = {"size": (1536, 864)}
_c = {
    "FrameRate": 100,
    # "AfMode": controls.AfModeEnum.Manual,
    # "LensPosition": 0.4,
}
config = picam2.create_video_configuration(main, controls=_c)
picam2.configure(config)
picam2.start()


data = toml.load("settings.toml")
camera_matrix = np.array(data["calibration"]["camera_matrix"]).reshape(3, 3)
dist_coeffs = np.array(data["calibration"]["dist_coeffs"])

# camera_matrix = np.identity(3)
# dist_coeffs = np.zeros(dist_coeffs.shape)
# print(dist_coeffs)

FIRST_FRAME = True


while True:
    frame = picam2.capture_array()
    meta = picam2.capture_metadata()
    # print(meta)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    corners, ids, rejected_img_points = detector.detectMarkers(frame)
    corners, ids, rejected_img_points, _ = detector.refineDetectedMarkers(
        image=frame,
        board=board,
        detectedCorners=corners,
        detectedIds=ids,
        rejectedCorners=rejected_img_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )

    rotation_vecs, translation_vecs = estimate_pose_single_markers(
        corners=corners,
        marker_size=0.05,
        camera_matrix=camera_matrix,
        distortion_coefficients=dist_coeffs,
    )

    if FIRST_FRAME and rotation_vecs is not None:
        f_rvec = rotation_vecs
        f_tvec = translation_vecs
        FIRST_FRAME = False
        rmat = cv2.Rodrigues(f_rvec[0][0])[0]

    if rotation_vecs is not None and not FIRST_FRAME:
        translated = rmat.T @ (
            f_tvec[0][0].reshape(3, 1) - translation_vecs[0][0].reshape(3, 1)
        )
        _t = translated.T[0] * 100
        _tx = round(_t[0])
        _ty = round(_t[1])
        _tz = round(_t[2])

        print(
            f"X: {_tx}  , Y:{_ty}  , Z:{_tz}  , lensPosition: {meta['LensPosition']}",
            end="\r",
        )
        sys.stdout.flush()
