import numpy as np
import cv2
from picamera2 import Picamera2
import libcamera
from cv2 import aruco
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

frame_size = (1200, 800)

picam2 = Picamera2()
WIDTH = frame_size[0]
HEIGHT = frame_size[1]
main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
_c = {"FrameRate": 100, "ExposureTime": 3000}
config = picam2.create_video_configuration(
    main, controls=_c, transform=libcamera.Transform(vflip=1)
)
picam2.configure(config)
picam2.start()


ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_PARAMETERS.useAruco3Detection = 1
ARUCO_PARAMETERS.cornerRefinementMethod = 3
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=markerLength,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT,
)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
mp_detector = vision.PoseLandmarker.create_from_options(options)

while 1:
    image = picam2.capture_array()
    image = image[:HEIGHT, :WIDTH]
    image = cv2.flip(image, 1)
    corners, ids, rejectedpoints = detector.detectMarkers(image)
    corners, ids, rejectedpoints, _ = detector.refineDetectedMarkers(
        image=image,
        board=board,
        detectedCorners=corners,
        detectedIds=ids,
        rejectedCorners=rejectedpoints,
    )
    if ids is not None:
        image = aruco.drawDetectedMarkers(image, corners, ids)
    r_image = cv2.resize(image.copy(), (300, 200))
    img = cv2.cvtColor(r_image, cv2.COLOR_GRAY2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8))
    detection_result = mp_detector.detect(mp_image)
    # print(detection_result)
    # break

    # STEP 5: Process the detection result. In this case, visualize it.
    # annotated_image = draw_landmarks_on_image(cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_GRAY2BGR), detection_result)
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow("asdf", annotated_image)
    cv2.waitKey(1)
