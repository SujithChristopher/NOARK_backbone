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

project_root = Path(__file__).parents[2]
data_root = "data"
recording_type = "calibration"
camera_type = "dual_160"
calib_folder_name = "radxa_calib_parallel"

calib_data_folder = os.path.join(
    project_root, data_root, recording_type, camera_type, calib_folder_name
)
cam0_frame_data = os.path.join(calib_data_folder, "cam0_frame.msgpack")
cam0_meta = os.path.join(calib_data_folder, "cam0_timestamp.msgpack")

cam1_frame_data = os.path.join(calib_data_folder, "cam1_frame.msgpack")
cam1_meta = os.path.join(calib_data_folder, "cam1_timestamp.msgpack")
os.path.exists(cam1_meta)
# print(f"Calibration frame data: {cam1_frame_data}")


# %% Metadata
def get_metadata(metaf):
    f = open(metaf, "rb")
    _ = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync, timestamps = _[:, 0], _[:, 1]
    sync = sync.astype(int).astype(bool)
    timestamp_dt = timestamps.astype("datetime64[us]")
    return sync, timestamp_dt


sync_cam0, timestamp_cam0 = get_metadata(cam0_meta)
sync_cam1, timestamp_cam1 = get_metadata(cam1_meta)

duration = timestamp_cam0[-1] - timestamp_cam0[0]
# duration in seconds
duration_s = duration.astype('timedelta64[s]').astype(float)
print(f"Duration of recording: {duration_s:.2f} seconds")

def get_video_unpacker(vidf):
    _video_file = open(vidf, "rb")
    _video_data = mp.Unpacker(_video_file, object_hook=mpn.decode)
    return _video_data

cam0_upak = get_video_unpacker(cam0_frame_data)
cam1_upak = get_video_unpacker(cam1_frame_data)

# calibration
def detectCorners(data):
    frame_id, _frame = data

    _frame = cv2.rotate(_frame.copy(), cv2.ROTATE_180)
    _frame = cv2.flip(_frame, 1)
    if len(_frame.shape) == 3:
        _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(_frame, patternSize)
    if ret:
        corners = cv2.cornerSubPix(
            _frame,
            corners,
            (5, 5),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    else:
        return None, frame_id
    return corners, frame_id

cam0_results = Parallel(n_jobs=20, verbose=0)(
    delayed(detectCorners)(frame) for frame in tqdm(enumerate(cam0_upak))
)

cam1_results = Parallel(n_jobs=20, verbose=0)(
    delayed(detectCorners)(frame) for frame in tqdm(enumerate(cam1_upak))
)


cam0_cb_corners = {
    'corners':[],
    'frame_idx':[],
    'timestamp':[],
    'sync':[]
}

cam1_cb_corners = {
    'corners':[],
    'frame_idx':[],
    'timestamp':[],
    'sync':[]
}

for corners, frame_idx in cam0_results:
    if corners is not None and len(corners) == 96:
        cam0_cb_corners['corners'].append(corners)
        cam0_cb_corners['frame_idx'].append(frame_idx)
        cam0_cb_corners['timestamp'].append(timestamp_cam0[frame_idx])
        cam0_cb_corners['sync'].append(timestamp_cam0[frame_idx])

for corners, frame_idx in cam1_results:
    if corners is not None and len(corners) == 96:
        cam1_cb_corners['corners'].append(corners)
        cam1_cb_corners['frame_idx'].append(frame_idx)
        cam1_cb_corners['timestamp'].append(timestamp_cam1[frame_idx])
        cam1_cb_corners['sync'].append(timestamp_cam1[frame_idx])


# Get the directory of the video file
def write_to_file(corners_dict, camera):
    import pickle
    corners_file = os.path.join(calib_data_folder, f"chessb_corners_{camera}.pkl")
    with open(corners_file, "wb") as f:
        pickle.dump(corners_dict, f)

write_to_file(cam0_cb_corners, 'cam0_frame')
write_to_file(cam1_cb_corners, 'cam1_frame')
# %%
