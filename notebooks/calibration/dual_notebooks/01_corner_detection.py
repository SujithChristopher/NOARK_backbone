# %% Imports

import os

import cv2
import matplotlib.pyplot as plt
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from cv2 import aruco
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# %% Board dimentions

patternSize = (8, 12)
squareSize = 30

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def construct3DPoints(patternSize, squareSize):
    X = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    X[:, :2] = np.mgrid[0 : patternSize[0], 0 : patternSize[1]].T.reshape(-1, 2)
    X = X * squareSize
    return X


boardPoints = construct3DPoints(patternSize, squareSize)

# %% Path defs

from pathlib import Path

project_root = Path.cwd().parents[2]
data_root = "data"
recording_type = "calibration"
camera_type = "dual_160"
calib_folder_name = "dual_cam_calibration_checker_sz_30mm"

calib_data_folder = os.path.join(
    project_root, data_root, recording_type, camera_type, calib_folder_name
)
cam_ov9281_data = os.path.join(calib_data_folder, "cam1_ov9281.msgpack")
cam_ov9281_meta = os.path.join(calib_data_folder, "cam1_timestamp.msgpack")

cam_imx219_data = os.path.join(calib_data_folder, "cam0_imx219.msgpack")
cam_imx219_meta = os.path.join(calib_data_folder, "cam0_timestamp.msgpack")
os.path.exists(cam_imx219_meta)


# %% Metadata
def get_metadata(metaf):
    f = open(metaf, "rb")
    _ = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync, timestamps = _[:, 0], _[:, 1]
    sync = sync.astype(int).astype(bool)
    timestamp_dt = timestamps.astype("datetime64[us]")
    return sync, timestamp_dt


sync_cam0, timestamp_cam0 = get_metadata(cam_imx219_meta)
sync_cam1, timestamp_cam1 = get_metadata(cam_ov9281_meta)
