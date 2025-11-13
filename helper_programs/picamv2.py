from picamera2 import Picamera2
import cv2
import time
import numpy as np
from cv2 import aruco

import RPi.GPIO as GPIO

picam2 = Picamera2()
sz = picam2.sensor_modes[0]["size"]

main = {"size": (800, 400)}
# main = {'size': (1536, 864)}

controls = {"FrameRate": 100}
config = picam2.create_video_configuration(main, controls=controls)
picam2.configure(config)
picam2.start()

start_time = time.time()
frame_count = 0

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36H12)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
marker_size = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=marker_size,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)

marker_points = np.array(
    [
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0],
    ],
    dtype=np.float32,
)

while True:
    img = picam2.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    corners, ids, rejected_image_points = detector.detectMarkers(gray)

    elapsed_time = time.time() - start_time
    frame_count += 1

    if 88 in np.array(ids):
        gray = aruco.drawDetectedMarkers(image=gray, corners=corners, ids=ids)
    cv2.imshow("asdf", gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if elapsed_time >= 1.0:
        # print(gray.shape)
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        print(f"FPS {fps}")

cv2.destroyAllWindows()
