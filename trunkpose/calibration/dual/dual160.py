import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import cv2
    from cv2 import aruco
    import numpy as np
    import msgpack as mp
    import msgpack_numpy as mpn
    import os
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    from joblib import Parallel, delayed

    return Parallel, aruco, cv2, delayed, mp, mpn, np, os, plt, tqdm


@app.cell(hide_code=True)
def _(cv2, np):
    patternSize = (8, 12)
    _squareSize = 30
    imgSize = (1280, 800)
    _criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def _construct3DPoints(patternSize, squareSize):
        X = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
        X[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
        X = X * _squareSize
        return X
    boardPoints = _construct3DPoints(patternSize, _squareSize)
    return (patternSize,)


@app.cell
def _(os):
    _project_root = os.getcwd()  # NOARK_backbone (marimo cwd = where it was launched)
    _calib_folder_name = "dual_cam_calibration_checker_sz_30mm"

    webcam_calib_folder = os.path.join(
        _project_root, "data", "calibration", "dual_160", _calib_folder_name
    )

    camera = 'cam1_ov9281'
    webcam_calib_video = os.path.join(webcam_calib_folder, f"{camera}.msgpack")
    return camera, webcam_calib_folder, webcam_calib_video


@app.cell
def _(mp, mpn, webcam_calib_video):
    _video_file = open(webcam_calib_video, 'rb')
    _video_data = mp.Unpacker(_video_file, object_hook=mpn.decode)
    _video_length = 0
    last_frame = None
    for _frame in _video_data:
        last_frame = _frame
        _video_length = _video_length + 1
    _video_file.close()
    print('video length, ', _video_length)
    return (last_frame,)


@app.cell
def _(last_frame):
    last_frame.shape
    return


@app.cell
def _(webcam_calib_video):
    _video_pth = webcam_calib_video
    chessb_corners = []
    counter = 0
    return


@app.cell
def _(last_frame, plt):
    plt.imshow(last_frame)
    return


@app.cell
def _(Parallel, cv2, delayed, mp, mpn, patternSize, tqdm, webcam_calib_video):
    def detectCorners(data):
        frame_id, _frame = data
        if len(_frame.shape) == 3:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(_frame, patternSize)
        if ret:
            corners = cv2.cornerSubPix(_frame, corners, (5, 5), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        else:
            return (None, frame_id)
        return (corners, frame_id)
    chessb_corners_1 = []
    _video_file = open(webcam_calib_video, 'rb')
    _video_data = mp.Unpacker(_video_file, object_hook=mpn.decode)
    results = Parallel(n_jobs=20, verbose=0)((delayed(detectCorners)(frame) for frame in tqdm(enumerate(_video_data))))
    _video_file.close()
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clean up the inhomogeneous  part
    """)
    return


@app.cell
def _(results):
    chessb_corners_2 = []
    for _result, frame_id in results:
        if _result is not None and len(_result) == 96:
            chessb_corners_2.append([_result, frame_id])
    return (chessb_corners_2,)


@app.cell
def _(chessb_corners_2):
    len(chessb_corners_2)
    return


@app.cell
def _(mo):
    mo.stop(True, mo.md("**Execution halted — cells below will not run**"))
    return


@app.cell
def _(camera, chessb_corners_2, mp, mpn, os, webcam_calib_video):
    video_dir = os.path.dirname(webcam_calib_video)
    corners_file = os.path.join(video_dir, f'chessb_corners_{camera}.msgpack')
    with open(corners_file, 'wb') as _f:
        _packed_file = mp.packb(chessb_corners_2, default=mpn.encode)
        _f.write(_packed_file)
    return


@app.cell
def _(mo):
    mo.stop(True, mo.md("**Execution halted — cells below will not run**"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating combinations
    """)
    return


@app.cell
def _(np):
    np.random.seed(9)
    return


@app.cell
def _(os, webcam_calib_folder):
    webcam_corners_pth = os.path.join(webcam_calib_folder, "chessb_corners_cam0_imx219.msgpack")
    return (webcam_corners_pth,)


@app.cell
def _(mp, mpn, np, webcam_corners_pth):
    # load data
    _corners_file = open(webcam_corners_pth, 'rb')
    chessb_corners_3 = list(mp.Unpacker(_corners_file, object_hook=mpn.decode))[0]
    chessboard_data = {'cb_corners': [], 'frame_id': []}
    for c, id in chessb_corners_3:
        chessboard_data['cb_corners'].append(c)
        chessboard_data['frame_id'].append(id)
    chessb_corners_3 = chessboard_data['cb_corners']
    chessb_corners_3 = np.array(chessb_corners_3)
    return (chessb_corners_3,)


@app.cell(hide_code=True)
def _(chessb_corners_3, np, plt):
    rnd = np.random.choice(len(chessb_corners_3), 100)
    chessb_c = chessb_corners_3[rnd]
    _flat_corners = chessb_c.reshape(-1, 2)
    _x_coords = _flat_corners[:, 0]
    _y_coords = _flat_corners[:, 1]
    _fig = plt.figure(figsize=(10, 10))
    _gs = _fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    _ax_main = _fig.add_subplot(_gs[1, 0])
    _ax_histx = _fig.add_subplot(_gs[0, 0], sharex=_ax_main)
    _ax_histy = _fig.add_subplot(_gs[1, 1], sharey=_ax_main)
    _ax_main.scatter(_x_coords, _y_coords, alpha=0.05, s=2, color='blue')
    _ax_main.set_xlabel('X Pixel Coordinate')
    _ax_main.set_ylabel('Y Pixel Coordinate')
    _ax_main.invert_yaxis()
    _ax_histx.hist(_x_coords, bins=50, color='gray', alpha=0.7)
    _ax_histx.tick_params(axis='x', labelbottom=False)
    _ax_histy.hist(_y_coords, bins=50, orientation='horizontal', color='gray', alpha=0.7)
    _ax_histy.tick_params(axis='y', labelleft=False)
    _ax_main.set_xlim(0, 1280)
    _ax_main.set_ylim(800, 0)
    plt.suptitle('Chessboard Corner Spatial Distribution & Density', fontsize=14, y=0.92)
    plt.show()
    return


@app.cell
def _(chessb_corners_3, plt):
    _flat_corners = chessb_corners_3.reshape(-1, 2)
    _x_coords = _flat_corners[:, 0]
    _y_coords = _flat_corners[:, 1]
    _fig = plt.figure(figsize=(10, 10))
    _gs = _fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    _ax_main = _fig.add_subplot(_gs[1, 0])
    _ax_histx = _fig.add_subplot(_gs[0, 0], sharex=_ax_main)
    _ax_histy = _fig.add_subplot(_gs[1, 1], sharey=_ax_main)
    _ax_main.scatter(_x_coords, _y_coords, alpha=0.05, s=2, color='blue')
    _ax_main.set_xlabel('X Pixel Coordinate')
    _ax_main.set_ylabel('Y Pixel Coordinate')
    _ax_main.invert_yaxis()
    _ax_histx.hist(_x_coords, bins=50, color='gray', alpha=0.7)
    _ax_histx.tick_params(axis='x', labelbottom=False)
    _ax_histy.hist(_y_coords, bins=50, orientation='horizontal', color='gray', alpha=0.7)
    _ax_histy.tick_params(axis='y', labelleft=False)
    plt.suptitle('Chessboard Corner Spatial Distribution & Density', fontsize=14, y=0.92)
    plt.show()
    return


@app.cell
def _(cv2, np):
    patternSize_1 = (8, 12)
    _squareSize = 30
    imgSize_1 = (1280, 800)
    _criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def _construct3DPoints(patternSize, squareSize):
        X = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
        X[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
        X = X * _squareSize
        return X
    boardPoints_1 = _construct3DPoints(patternSize_1, _squareSize)
    return boardPoints_1, imgSize_1


@app.cell
def _():
    permute_value = 20
    return (permute_value,)


@app.cell
def _(
    Parallel,
    boardPoints_1,
    chessb_corners_3,
    cv2,
    delayed,
    imgSize_1,
    np,
    permute_value,
    tqdm,
):
    useFisheye = True

    def calibrate_single_iteration(_):
        rnd = np.random.choice(len(chessb_corners_3), 90)
        chessb_c = chessb_corners_3[rnd]
        worldPoints = []
        imagePoints = []
        useFisheye = True
        for _f in chessb_c:
            imagePoints.append(_f)
            worldPoints.append(boardPoints_1)
        if useFisheye:
            flagsCalib = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND
            K_init = np.array([[540.0, 0.0, 640.0], [0.0, 540.0, 400.0], [0.0, 0.0, 1.0]])
            D_init = np.zeros((4, 1))
            calibrateCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-09)
            ret, cameraMatrix, k, R, t = cv2.fisheye.calibrate(np.expand_dims(np.asarray(worldPoints), -2), imagePoints, imgSize_1, None, None, flags=flagsCalib, criteria=calibrateCriteria)
        else:
            flagsCalib = cv2.CALIB_RATIONAL_MODEL
            ret, cameraMatrix, k, R, t = cv2.calibrateCamera(worldPoints, imagePoints, imgSize_1, None, None, flags=flagsCalib)
        return {'ReError': ret, 'mat': cameraMatrix, 'dist': k, 'rvec': R, 'tvec': t, 'rnd': rnd}
    my_dict = {'ReError': [], 'mat': [], 'dist': [], 'rvec': [], 'tvec': [], 'rnd_value': []}
    results_1 = Parallel(n_jobs=20, verbose=0)((delayed(calibrate_single_iteration)(_) for _ in tqdm(range(permute_value))))
    for _result in results_1:
        my_dict['ReError'].append(_result['ReError'])
        my_dict['mat'].append(_result['mat'])
        my_dict['dist'].append(_result['dist'])
        my_dict['rvec'].append(_result['rvec'])
        my_dict['tvec'].append(_result['tvec'])
        my_dict['rnd_value'].append(_result['rnd'])
    return my_dict, useFisheye


@app.cell
def _(mp, mpn, my_dict, os, permute_value, useFisheye, webcam_calib_folder):
    if useFisheye:
        calibration_file = os.path.join(webcam_calib_folder, f'calibration_data_{permute_value}.msgpack')
    else:
        calibration_file = os.path.join(webcam_calib_folder, f'standard_calibration_data_{permute_value}.msgpack')
    with open(calibration_file, 'wb') as _f:
        _packed_file = mp.packb(my_dict, default=mpn.encode)
        _f.write(_packed_file)
    return


@app.cell
def _(my_dict, plt):
    plt.plot(my_dict['ReError'])
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection Error')
    plt.title('Reprojection Error over Iterations')
    plt.grid()
    return


@app.cell
def _(my_dict, np):
    min_rerr_idx = np.argmin(my_dict['ReError'])
    print('Minimum Reprojection Error:', my_dict['ReError'][min_rerr_idx], ' at index ', min_rerr_idx)
    return (min_rerr_idx,)


@app.cell
def _(min_rerr_idx, my_dict):
    rerr = my_dict['ReError'][min_rerr_idx]
    camera_mat = my_dict['mat'][min_rerr_idx]
    camera_dist = my_dict['dist'][min_rerr_idx]
    return camera_dist, camera_mat, rerr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(camera_dist, camera_mat, rerr):
    print(rerr, camera_mat, camera_dist)
    return


@app.cell
def _(my_dict, np, plt):
    # plot the translation vector
    _tvec = my_dict['tvec'][0]
    _tvec = np.array(_tvec).T[0]
    plt.figure(figsize=(10, 5))
    plt.plot(_tvec[0], label='X Translation')
    plt.plot(_tvec[1], label='Y Translation')
    plt.plot(_tvec[2], label='Z Translation')
    plt.xlabel('Frame')
    plt.ylabel('Translation Vector')
    plt.title('Translation Vector over Frames')
    plt.legend()
    tvec_std = np.std(_tvec, axis=1)
    # standard deviation of the translation vector
    print('Translation Vector Standard Deviation:', tvec_std)
    return


@app.cell
def _(camera_dist, camera_mat, cv2, mp, mpn, plt, webcam_calib_video):
    # show distorted and undistorted images
    _video_pth = webcam_calib_video
    _video_file = open(_video_pth, 'rb')
    _video_data = mp.Unpacker(_video_file, object_hook=mpn.decode)
    for _idx, frame in enumerate(_video_data):
        frame_undistorted = cv2.undistort(frame, camera_mat, camera_dist)  # frame = cv2.flip(frame, 0)
        if _idx == 0:  # frame = cv2.rotate(frame, cv2.ROTATE_180)
            plt.figure(figsize=(15, 8))
            plt.subplot(1, 2, 1)  # create a subplot with 2 images and break at first image and plot chessboard corners
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Distorted Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
            plt.title('Undistorted Image')
            plt.axis('off')
            plt.show()
            break
    return


@app.cell
def _(mo):
    mo.stop(True, mo.md("**Execution halted — cells below will not run**"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis by using a second video
    """)
    return


@app.cell
def _(os):
    import sys
    import polars as pl
    from datetime import datetime
    sys.path.insert(1, os.path.join(os.getcwd(), 'notebooks'))
    from scipy.spatial.transform import Rotation as R
    from pd_support import add_datetime_col, read_rigid_body_csv, get_rb_marker_name
    from scipy.interpolate import interp1d

    return (
        R,
        add_datetime_col,
        datetime,
        get_rb_marker_name,
        interp1d,
        pl,
        read_rigid_body_csv,
    )


@app.cell
def _(mp, mpn, np, os):
    _project_root = os.getcwd()  # NOARK_backbone
    _fov = '160_fov'
    permute_value_1 = 20
    useFisheye_1 = True
    recording_folder_name = '3marker_linear_2d_160fov_t1'
    reference_recording_folder = os.path.join(_project_root, 'data', 'recordings', _fov, '3marker_complete_data', recording_folder_name)
    reference_file = os.path.join(reference_recording_folder, 'webcam_color.msgpack')
    _timestamp_file = os.path.join(reference_recording_folder, 'webcam_timestamp.msgpack')
    with open(_timestamp_file, 'rb') as _f:
        _metadata = list(mp.Unpacker(_f, object_hook=mpn.decode))
        timestamp = np.array(_metadata)[:, 1]
        sync_pulse = np.array(_metadata)[:, 0]
    return (
        permute_value_1,
        recording_folder_name,
        reference_file,
        reference_recording_folder,
        sync_pulse,
        timestamp,
        useFisheye_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### load calibration combinations
    """)
    return


@app.cell
def _(mp, mpn, os, permute_value_1, useFisheye_1, webcam_calib_folder):
    if useFisheye_1:
        _calibration_data = os.path.join(webcam_calib_folder, f'calibration_data_{permute_value_1}.msgpack')
    else:
        _calibration_data = os.path.join(webcam_calib_folder, f'standard_calibration_data_{permute_value_1}.msgpack')
    with open(_calibration_data, 'rb') as _f:
        my_dict_1 = list(mp.Unpacker(_f, object_hook=mpn.decode))
    my_dict_1 = my_dict_1[0]
    return (my_dict_1,)


@app.cell
def _(mp, mpn, reference_file):
    ref_video_length = 0
    for _ in mp.Unpacker(open(reference_file, 'rb'), object_hook=mpn.decode):
        ref_video_length = ref_video_length + 1
    print('video length, ', ref_video_length)
    return (ref_video_length,)


@app.cell
def _(aruco, cv2, np):
    ARUCO_PARAMETERS = aruco.DetectorParameters()
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
    markerLength = 0.048
    markerSeperation = 0.01
    board = aruco.GridBoard(size=[1, 1], markerLength=markerLength, markerSeparation=markerSeperation, dictionary=ARUCO_DICT)

    def estimate_pose_single_markers(corners, marker_size, camera_matrix, distortion_coefficients=np.zeros((5, 1))):
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0], [marker_size / 2, marker_size / 2, 0], [marker_size / 2, -marker_size / 2, 0], [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        rvecs, _tvecs = ([], [])
        for corner in corners:
            _, r, t = cv2.solvePnP(marker_points, corner, camera_matrix, distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
            if r is not None and t is not None:
                rvecs.append(r.reshape(1, 3).tolist())
                _tvecs.append(t.reshape(1, 3).tolist())
            else:
                rvecs.append(np.array([[np.nan, np.nan, np.nan]]).tolist())
                _tvecs.append(np.array([[np.nan, np.nan, np.nan]]).tolist())
        return (np.array(rvecs, dtype=np.float32), np.array(_tvecs, dtype=np.float32))

    return board, detector, estimate_pose_single_markers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## load clalibration data
    """)
    return


@app.cell
def _(my_dict_1):
    _calib_data = my_dict_1
    return


@app.cell
def _(my_dict_1):
    len(my_dict_1['ReError'])
    return


@app.cell
def _(board, detector, mp, mpn, np, ref_video_length, reference_file, tqdm):
    # selecting random 50 frames
    np.random.seed(9)
    _random_reference_frames_idx = np.random.choice(ref_video_length, 300)
    _ref_data = mp.Unpacker(open(reference_file, 'rb'), object_hook=mpn.decode)
    ar_total_results = {'calib_idx': [], 'ar_data': []}
    ar_results = {'corners': [], 'ids': [], 'rejected': []}
    # _ref_frames = []
    vector_std = {'v1std': [], 'v2std': [], 'v3std': [], 'sum': [], 'r1std': [], 'r2std': [], 'r3std': [], 'r_sum': []}
    for _idx, _frame in tqdm(enumerate(_ref_data)):
        res = detector.detectMarkers(_frame)
        res = detector.refineDetectedMarkers(_frame, board, res[0], res[1], res[2])
        ar_results['corners'].append(res[0])
        ar_results['ids'].append(res[1])
        ar_results['rejected'].append(res[2])  # if idx in _random_reference_frames_idx:  # _frame = cv2.rotate(_frame, cv2.ROTATE_180)  # _frame = cv2.flip(_frame, 1)
    return (ar_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## path definition
    """)
    return


@app.cell
def _(timestamp):
    timestamp
    return


@app.cell
def _(datetime, pl, sync_pulse, timestamp):
    ar_df = {"time": timestamp, "sync": sync_pulse}
    ar_df = pl.from_dict(ar_df)
    if type(ar_df["time"][0]) is not datetime:
        ar_df = ar_df.with_columns(pl.col("time").str.to_datetime())
    return (ar_df,)


@app.cell
def _(
    add_datetime_col,
    os,
    pl,
    read_rigid_body_csv,
    recording_folder_name,
    reference_recording_folder,
):
    mocap_df, st_time = read_rigid_body_csv(
        os.path.join(reference_recording_folder, f"{recording_folder_name}.csv")
    )
    mocap_df = add_datetime_col(mocap_df, st_time, "seconds")
    mocap_df = pl.from_pandas(mocap_df)
    return (mocap_df,)


@app.cell
def _(get_rb_marker_name):
    # from notebooks.pd_support import get_rb_marker_name
    tr = get_rb_marker_name(2)
    tl = get_rb_marker_name(6)
    br = get_rb_marker_name(4)
    bl = get_rb_marker_name(8)

    # tr = get_rb_marker_name(8)
    # tl = get_rb_marker_name(6)
    # br = get_rb_marker_name(2)
    # bl = get_rb_marker_name(5)


    # tr = get_rb_marker_name(4)
    # tl = get_rb_marker_name(2)
    # br = get_rb_marker_name(3)
    # bl = get_rb_marker_name(1)

    to = get_rb_marker_name(1)
    return bl, br, tl, to, tr


@app.cell
def _(ar_df):
    ar_df['sync'][0]
    return


@app.cell
def _(ar_df, pl):
    ar_df_1 = ar_df.with_columns(pl.col('sync').cast(pl.Int8).cast(pl.Boolean))
    return (ar_df_1,)


@app.cell
def _(ar_df_1, ar_results, mocap_df, pl):
    start_pulse = ar_df_1['sync'].arg_true().head(1).item()
    offset = (~ar_df_1['sync'].slice(start_pulse)).arg_true().head(1).item()
    end_pulse = start_pulse + offset
    print(f'Start pulse: {start_pulse}, End pulse: {end_pulse}')
    ar_df_2 = ar_df_1[start_pulse:end_pulse]
    ar_corners = ar_results['corners'][start_pulse:end_pulse]
    ids = ar_results['ids'][start_pulse:end_pulse]
    time_diff = mocap_df['time'][0] - ar_df_2['time'][0]
    ar_df_2 = ar_df_2.with_columns([(pl.col('time') + time_diff).alias('time')])
    return ar_corners, ar_df_2, ids, time_diff


@app.cell
def _(mocap_df):
    mocap_df["time"][0]
    return


@app.cell
def _(ar_df_2):
    ar_df_2['time'][0]
    return


@app.cell
def _(time_diff):
    time_diff
    return


@app.cell
def _(R, bl, br, mocap_df, pl, tl, tr):
    mocap_mean = {'x': [], 'y': [], 'z': []}
    mocap_mean['x'] = mocap_df[[tr['x'], tl['x'], br['x'], bl['x']]].to_numpy().mean(axis=1)
    mocap_mean['y'] = mocap_df[[tr['y'], tl['y'], br['y'], bl['y']]].to_numpy().mean(axis=1)
    mocap_mean['z'] = mocap_df[[tr['z'], tl['z'], br['z'], bl['z']]].to_numpy().mean(axis=1)
    mocap_qt_0 = mocap_df[['rb_ang_x', 'rb_ang_y', 'rb_ang_z', 'rb_ang_w']][0].to_numpy()
    mocap_rotation = R.from_quat(mocap_qt_0).as_matrix()
    mocap_mean = pl.from_dict(mocap_mean)
    mt_dict = {'x': [], 'y': [], 'z': []}
    rmat_m = mocap_rotation[0]
    for _i in range(len(mocap_df['time'])):
        tvec_ar = rmat_m.T @ (mocap_mean[['x', 'y', 'z']][_i].to_numpy().reshape(3, 1) - mocap_mean[['x', 'y', 'z']][0].to_numpy().reshape(3, 1))
        tvec_ar = tvec_ar.T[0]
        mt_dict['x'].append(tvec_ar[0])
        mt_dict['y'].append(tvec_ar[1])
        mt_dict['z'].append(tvec_ar[2])
    mt_dict['time'] = mocap_df['time']
    return mocap_mean, mocap_rotation, mt_dict, rmat_m


@app.cell
def _(mocap_df, mocap_mean, rmat_m, to):
    to_marker = mocap_df[[to['x'], to['y'], to['z']]].to_numpy()[0]
    offset_tvvv = rmat_m.T @ (mocap_mean[["x", "y", "z"]].to_numpy()[0].reshape(3, 1) - to_marker.reshape(3,1))
    return offset_tvvv, to_marker


@app.cell
def _(to_marker):
    to_marker.reshape(3,1)
    return


@app.cell
def _(mocap_mean):
    mocap_mean[["x", "y", "z"]].to_numpy()[0].reshape(3, 1)
    return


@app.cell
def _(R, mocap_df, mocap_rotation, np):
    mc_angle_arr = mocap_df[["rb_ang_x", "rb_ang_y", "rb_ang_z", "rb_ang_w"]].to_numpy()
    mocap_angle = []
    mc_ang_x = []
    mc_ang_y = []
    mc_ang_z = []
    for _a in mc_angle_arr:
        try:
            _ax, _ay, _az = R.from_matrix(
                mocap_rotation[0].T @ R.from_quat(_a).as_matrix()
            ).as_euler("xyz", degrees=True)
            mc_ang_x.append(_ax)
            mc_ang_y.append(_ay)
            mc_ang_z.append(_az)
        except:
            _ax, _ay, _az = R.from_matrix(mocap_rotation[0].T @ np.eye(3)).as_euler(
                "xyz", degrees=True
            )
            mc_ang_x.append(_ax)
            mc_ang_y.append(_ay)
            mc_ang_z.append(_az)
    return mc_ang_x, mc_ang_y, mc_ang_z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interoplating
    """)
    return


@app.cell
def _(ar_df_2, interp1d, mc_ang_x, mc_ang_y, mc_ang_z, mt_dict, pl):
    mocap = pl.from_dict(mt_dict)
    x1 = interp1d(mocap['time'].dt.epoch(), mocap['x'], fill_value='extrapolate')
    y1 = interp1d(mocap['time'].dt.epoch(), mocap['y'], fill_value='extrapolate')
    z1 = interp1d(mocap['time'].dt.epoch(), mocap['z'], fill_value='extrapolate')
    _ax = interp1d(mocap['time'].dt.epoch(), mc_ang_x, fill_value='extrapolate')
    ay = interp1d(mocap['time'].dt.epoch(), mc_ang_y, fill_value='extrapolate')
    az = interp1d(mocap['time'].dt.epoch(), mc_ang_z, fill_value='extrapolate')
    mocap_ip = {'time': ar_df_2['time']}
    mocap_ip['x'] = x1(ar_df_2['time'].dt.epoch())
    mocap_ip['y'] = y1(ar_df_2['time'].dt.epoch())
    mocap_ip['z'] = z1(ar_df_2['time'].dt.epoch())
    mocap_ip['rx'] = _ax(ar_df_2['time'].dt.epoch())
    mocap_ip['ry'] = ay(ar_df_2['time'].dt.epoch())
    mocap_ip['rz'] = az(ar_df_2['time'].dt.epoch())
    mocap_ip = pl.from_dict(mocap_ip)
    return (mocap_ip,)


@app.cell
def _():
    default_ids = [12, 14, 20]
    return (default_ids,)


@app.cell
def _(my_dict_1):
    len(my_dict_1['mat'])
    return


@app.cell
def _(
    Parallel,
    ar_corners,
    cv2,
    delayed,
    estimate_pose_single_markers,
    ids,
    mocap_ip,
    my_dict_1,
    np,
    useFisheye_1,
):
    def process_single_iteration(i, fish_mat, fish_dist, all_corners_concat, corner_counts, mocap_x, mocap_y, mocap_z):
        """Process a single iteration of the calibration loop"""
        if useFisheye_1:
            _new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(fish_mat, fish_dist, (1280, 800), np.eye(3), balance=1)
            _undist_all = cv2.fisheye.undistortPoints(all_corners_concat, fish_mat, fish_dist, None, _new_cam)
        else:
            _new_cam, roi = cv2.getOptimalNewCameraMatrix(fish_mat, fish_dist, (1280, 800), 1.0, (1280, 800))
            fish_dist = fish_dist
            _undist_all = all_corners_concat
        _undist_corners = []
        _idx = 0
        for _count in corner_counts:
            if _count > 0:
                _undist_corners.append(_undist_all[_idx:_idx + _count])
                _idx = _idx + _count
            else:
                _undist_corners.append(None)
        _num_markers = len(_undist_corners)
        rvecs = np.full((_num_markers, 3), np.nan)
        _tvecs = np.full((_num_markers, 3), np.nan)
        for _idx, _c in enumerate(_undist_corners):
            if _c is not None:
                _num_detected_markers = _c.shape[0] // 4
                _reshaped_corners = [_c[_j * 4:(_j + 1) * 4].reshape(4, 1, 2) for _j in range(_num_detected_markers)]
                _rotation_vectors, _translation_vectors = estimate_pose_single_markers(corners=_reshaped_corners, marker_size=0.048, camera_matrix=fish_mat, distortion_coefficients=fish_dist)
                if _rotation_vectors is not None and len(_rotation_vectors) > 0:
                    rvecs[_idx] = _rotation_vectors[0][0]
                    _tvecs[_idx] = _translation_vectors[0][0]
        valid_idx = None
        for _idx in range(len(rvecs)):
            if not np.isnan(rvecs[_idx]).any():
                valid_idx = _idx
                break
        if valid_idx is None:
            return {'err_x': np.nan, 'err_y': np.nan, 'err_z': np.nan, 'mean_err': np.nan, 'idx': i}
        _rmat = cv2.Rodrigues(rvecs[valid_idx])[0]
        _rmat_T = _rmat.T
        tvec_diff = _tvecs - _tvecs[valid_idx]
        tvec_transformed = (_rmat_T @ tvec_diff.T).T
        transformed_tvecs = tvec_transformed.copy()
        _ex = np.nanmean(np.abs(transformed_tvecs[:, 0] - mocap_x))
        _ey = np.nanmean(np.abs(transformed_tvecs[:, 1] - mocap_y))
        _ez = np.nanmean(np.abs(transformed_tvecs[:, 2] - mocap_z))
        _mean_err = np.nanmean([_ex, _ey, _ez])
        _max_x = np.nanmax(np.abs(transformed_tvecs[:, 0] - mocap_x))
        _max_y = np.nanmax(np.abs(transformed_tvecs[:, 1] - mocap_y))
        _max_z = np.nanmax(np.abs(transformed_tvecs[:, 2] - mocap_z))
        return {'err_x': _max_x, 'err_y': _max_y, 'err_z': _max_z, 'mean_err': _mean_err, 'idx': _i, 'tvecs': transformed_tvecs}
    _all_corners_list = []
    _corner_counts = []
    print('Filtering corners for ID 12...')
    for _corner, _id in zip(ar_corners, ids):
        try:
            _id_index = _id.reshape(-1).tolist().index(12)
            _all_corners_list.append(np.array(_corner[_id_index]).reshape(-1, 2))
            _corner_counts.append(len(np.array(_corner[_id_index]).reshape(-1, 2)))
        except:
            _corner_counts.append(0)
    _all_corners_concat = np.vstack(_all_corners_list).reshape(-1, 1, 2)
    print(f'Found {sum((1 for c in _corner_counts if c > 0))} valid corners with ID 12 out of {len(_corner_counts)} total')
    if not _all_corners_list:
        raise ValueError('No valid corners found for ID 12! Check your filtering logic.')
    _all_corners_concat = np.vstack(_all_corners_list).reshape(-1, 1, 2)
    print(f'Concatenated corners shape: {_all_corners_concat.shape}')
    mocap_x = mocap_ip['x'].to_numpy()
    mocap_y = mocap_ip['y'].to_numpy()
    mocap_z = mocap_ip['z'].to_numpy()
    n_jobs = 12
    print(f"Processing {len(my_dict_1['mat'])} iterations with {n_jobs} parallel jobs...")
    results_2 = Parallel(n_jobs=n_jobs, verbose=10)((delayed(process_single_iteration)(_i, my_dict_1['mat'][_i], my_dict_1['dist'][_i], _all_corners_concat, _corner_counts, mocap_x, mocap_y, mocap_z) for _i in range(len(my_dict_1['mat']))))
    error_dict = {'err_x': [r['err_x'] for r in results_2], 'err_y': [r['err_y'] for r in results_2], 'err_z': [r['err_z'] for r in results_2], 'mean_err': [r['mean_err'] for r in results_2], 'idx': [r['idx'] for r in results_2], 'tvecs': [r['tvecs'] for r in results_2]}
    print('Processing complete!')
    print(f"Valid results: {sum((1 for x in error_dict['mean_err'] if not np.isnan(x)))}/{len(error_dict['mean_err'])}")
    print(f"Mean error (excluding NaN): {np.nanmean(error_dict['mean_err']):.4f}")
    return error_dict, mocap_x, mocap_y, mocap_z


@app.cell
def _(error_dict, np):
    error_field = 'mean_err' 
    min_index = np.argmin(error_dict[error_field])
    # min_index = 8
    print(error_dict[error_field][min_index], 'min index: ', min_index)
    return (min_index,)


@app.cell
def _(error_dict, min_index, mocap_x, mocap_y, mocap_z, np):
    tvec_transformed_best = np.array(error_dict['tvecs'][min_index])
    mocap_array = np.array([mocap_x, mocap_y, mocap_z]).T
    apriltag_array = tvec_transformed_best
    valid_mask = np.isfinite(mocap_array).all(axis=1) & np.isfinite(apriltag_array).all(axis=1)
    clean_mocap = mocap_array[valid_mask]
    clean_apriltag = apriltag_array[valid_mask]
    print(f'Original frames: {len(mocap_array)}')
    print(f'Cleaned frames: {len(clean_mocap)}')
    print(f'Dropped frames: {len(mocap_array) - len(clean_mocap)}')
    return clean_apriltag, clean_mocap, mocap_array


@app.cell
def _(clean_apriltag, clean_mocap, mocap_array, np):
    from scipy.spatial.transform import Rotation as R_scipy

    def align_trajectories(source, target):
        """
        Aligns source array to target array using the Kabsch algorithm.
        Both arrays should be of shape (N, 3).
        Returns the rotation matrix, translation vector, and Euler angles in degrees.
        """
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)  # 1. Calculate centroids and center the data
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:  # 2. Calculate covariance matrix H and perform SVD
            Vt[2, :] = Vt[2, :] * -1
            R = Vt.T @ U.T
        t = centroid_target - R @ centroid_source
        rotation_obj = R_scipy.from_matrix(R)  # 3. Calculate the optimal rotation matrix R
        angles_degrees = rotation_obj.as_euler('xyz', degrees=True)
        return (R, t, angles_degrees)
    R_opt, t_opt, euler_deg = align_trajectories(clean_mocap, clean_apriltag)  # 4. Handle reflection case
    print(f'Rotation around X-axis: {euler_deg[0]:.2f} degrees')
    print(f'Rotation around Y-axis: {euler_deg[1]:.2f} degrees')
    print(f'Rotation around Z-axis: {euler_deg[2]:.2f} degrees')
    # --- How to use it ---
    # Assuming 'clean_mocap' and 'clean_apriltag' from the previous step:
    # Apply the transformation to your original array as usual
    aligned_mocap = (R_opt @ mocap_array.T).T + t_opt  # 5. Calculate the translation vector t  # ---------------------------------------------------------  # NEW: Convert the 3x3 Rotation Matrix to Euler Angles  # ---------------------------------------------------------  # We use 'xyz' order (pitch, yaw, roll).  # The order matters in 3D space, but 'xyz' is the most intuitive  # for reading "rotation around X, then Y, then Z".
    return R_opt, R_scipy, aligned_mocap


@app.cell
def _(tvec_transformed):
    tvec_transformed
    return


@app.cell
def _(aligned_mocap, np, tvec_transformed):
    print(np.nanmax(aligned_mocap[:,0] - tvec_transformed[:, 0]))
    print(np.nanmax(aligned_mocap[:,2] - tvec_transformed[:, 2]))
    return


@app.cell
def _(aligned_mocap, plt, tvec_transformed):
    plt.plot(aligned_mocap[:,0], aligned_mocap[:,2])
    # plt.plot(mocap_x, mocap_z)
    plt.plot(tvec_transformed[:, 0], tvec_transformed[:, 2])
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluating a section to see everything is right
    """)
    return


@app.cell
def _(
    aligned_mocap,
    ar_corners,
    cv2,
    estimate_pose_single_markers,
    ids,
    min_index,
    my_dict_1,
    np,
    useFisheye_1,
):
    _all_corners_list = []
    _corner_counts = []
    print('Filtering corners for ID 12...')
    for _corner, _id in zip(ar_corners, ids):
        try:
            _id_index = _id.reshape(-1).tolist().index(12)
            _all_corners_list.append(np.array(_corner[_id_index]).reshape(-1, 2))
            _corner_counts.append(len(np.array(_corner[_id_index]).reshape(-1, 2)))
        except:
            _corner_counts.append(0)
    _all_corners_concat = np.vstack(_all_corners_list).reshape(-1, 1, 2)
    _i = min_index
    _fish_mat = my_dict_1['mat'][_i]
    _fish_dist = my_dict_1['dist'][_i]
    if useFisheye_1:
        _new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_fish_mat, _fish_dist, (1280, 800), np.eye(3), balance=1)
        _undist_all = cv2.fisheye.undistortPoints(_all_corners_concat, _fish_mat, _fish_dist, None, _new_cam)
    else:
        _new_cam = _fish_mat
        _fish_dist = _fish_dist[:4]
        _undist_all = _all_corners_concat
    _undist_corners = []
    _idx = 0
    for _count in _corner_counts:
        if _count > 0:
            _undist_corners.append(_undist_all[_idx:_idx + _count])
            _idx = _idx + _count
        else:
            _undist_corners.append(None)
    _num_markers = len(_undist_corners)
    rvecs = np.full((_num_markers, 3), np.nan)
    _tvecs = np.full((_num_markers, 3), np.nan)
    for _idx, _c in enumerate(_undist_corners):
        if _c is not None:
            _num_detected_markers = _c.shape[0] // 4
            _reshaped_corners = [_c[_j * 4:(_j + 1) * 4].reshape(4, 1, 2) for _j in range(_num_detected_markers)]
            _rotation_vectors, _translation_vectors = estimate_pose_single_markers(corners=_reshaped_corners, marker_size=0.047, camera_matrix=_new_cam, distortion_coefficients=_fish_dist)
            if _rotation_vectors is not None and len(_rotation_vectors) > 0:
                rvecs[_idx] = _rotation_vectors[0][0]
                _tvecs[_idx] = _translation_vectors[0][0]
    valid_idx = None
    for _idx in range(len(rvecs)):
        if not np.isnan(rvecs[_idx]).any():
            valid_idx = _idx
            break
    _rmat = cv2.Rodrigues(rvecs[valid_idx])[0]
    _rmat_T = _rmat.T
    tvec_diff = _tvecs - _tvecs[valid_idx]
    tvec_transformed = (_rmat_T @ tvec_diff.T).T
    transformed_tvecs = tvec_transformed.copy()
    _ex = np.nanmean(np.abs(transformed_tvecs[:, 0] - aligned_mocap[:, 0]))
    _ey = np.nanmean(np.abs(transformed_tvecs[:, 1] - aligned_mocap[:, 1]))
    _ez = np.nanmean(np.abs(transformed_tvecs[:, 2] - aligned_mocap[:, 2]))
    _mean_err = np.nanmean([_ex, _ey, _ez])
    _ex_arr = np.abs(transformed_tvecs[:, 0] - aligned_mocap[:, 0])
    _px = np.nanpercentile(_ex_arr, 95)
    _ex_p = np.nanmax(_ex_arr[_ex_arr <= _px])
    _ey_arr = np.abs(transformed_tvecs[:, 1] - aligned_mocap[:, 1])
    _py = np.nanpercentile(_ey_arr, 95)
    _ey_p = np.nanmax(_ey_arr[_ey_arr <= _py])
    _ez_arr = np.abs(transformed_tvecs[:, 2] - aligned_mocap[:, 2])
    _pz = np.nanpercentile(_ez_arr, 95)
    _ez_p = np.nanmax(_ez_arr[_ez_arr <= _pz])
    _max_x = np.nanmax(np.abs(transformed_tvecs[:, 0] - aligned_mocap[:, 0]))
    _max_y = np.nanmax(np.abs(transformed_tvecs[:, 1] - aligned_mocap[:, 1]))
    _max_z = np.nanmax(np.abs(transformed_tvecs[:, 2] - aligned_mocap[:, 2]))
    print(f'Max error: X={_max_x}, Y={_max_y}, Z={_max_z}')
    print(f'Max error 95th percentile: X={_ex_p}, Y={_ey_p}, Z={_ez_p}')
    return (tvec_transformed,)


@app.cell
def _(
    aligned_mocap,
    ar_corners,
    cv2,
    estimate_pose_single_markers,
    ids,
    min_index,
    my_dict_1,
    np,
    offset_tvvv,
    useFisheye_1,
):
    _all_corners_list = []
    _corner_counts = []
    print('Filtering corners for ID 12...')
    for _corner, _id in zip(ar_corners, ids):
        try:
            _id_index = _id.reshape(-1).tolist().index(12)
            _all_corners_list.append(np.array(_corner[_id_index]).reshape(-1, 2))
            _corner_counts.append(len(np.array(_corner[_id_index]).reshape(-1, 2)))
        except:
            _corner_counts.append(0)
    _all_corners_concat = np.vstack(_all_corners_list).reshape(-1, 1, 2)
    _i = min_index
    _fish_mat = my_dict_1['mat'][_i]
    _fish_dist = my_dict_1['dist'][_i]
    if useFisheye_1:
        _new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_fish_mat, _fish_dist, (1280, 800), np.eye(3), balance=1)
        _undist_all = cv2.fisheye.undistortPoints(_all_corners_concat, _fish_mat, _fish_dist, None, _new_cam)
    else:
        _new_cam = _fish_mat
        _fish_dist = _fish_dist[:4]
        _undist_all = _all_corners_concat
    _undist_corners = []
    _idx = 0
    for _count in _corner_counts:
        if _count > 0:
            _undist_corners.append(_undist_all[_idx:_idx + _count])
            _idx = _idx + _count
        else:
            _undist_corners.append(None)
    _num_markers = len(_undist_corners)
    rvecs_1 = np.full((_num_markers, 3), np.nan)
    _tvecs = np.full((_num_markers, 3), np.nan)
    for _idx, _c in enumerate(_undist_corners):
        if _c is not None:
            _num_detected_markers = _c.shape[0] // 4
            _reshaped_corners = [_c[_j * 4:(_j + 1) * 4].reshape(4, 1, 2) for _j in range(_num_detected_markers)]
            _rotation_vectors, _translation_vectors = estimate_pose_single_markers(corners=_reshaped_corners, marker_size=0.047, camera_matrix=_new_cam, distortion_coefficients=_fish_dist)
            if _rotation_vectors is not None and len(_rotation_vectors) > 0:
                rvecs_1[_idx] = _rotation_vectors[0][0]
                _tvecs[_idx] = _translation_vectors[0][0]
    valid_idx_1 = None
    for _idx in range(len(rvecs_1)):
        if not np.isnan(rvecs_1[_idx]).any():
            valid_idx_1 = _idx
            break
    _rmat = cv2.Rodrigues(rvecs_1[valid_idx_1])[0]
    _rmat_T = _rmat.T
    _h_offset = offset_tvvv
    _newtvec = []
    for _rr, _tt in zip(rvecs_1, _tvecs):
        _rt = (cv2.Rodrigues(_rr)[0] @ _h_offset + _tt).T[0]
        _newtvec.append(_rt)
    _newtvec = np.array(_newtvec)
    tvec_diff_1 = _newtvec - _newtvec[valid_idx_1]
    tvec_transformed_1 = (_rmat_T @ tvec_diff_1.T).T
    transformed_tvecs_1 = tvec_transformed_1.copy()
    _ex = np.nanmean(np.abs(transformed_tvecs_1[:, 0] - aligned_mocap[:, 0]))
    _ey = np.nanmean(np.abs(transformed_tvecs_1[:, 1] - aligned_mocap[:, 1]))
    _ez = np.nanmean(np.abs(transformed_tvecs_1[:, 2] - aligned_mocap[:, 2]))
    _mean_err = np.nanmean([_ex, _ey, _ez])
    _ex_arr = np.abs(transformed_tvecs_1[:, 0] - aligned_mocap[:, 0])
    _px = np.nanpercentile(_ex_arr, 95)
    _ex_p = np.nanmax(_ex_arr[_ex_arr <= _px])
    _ey_arr = np.abs(transformed_tvecs_1[:, 1] - aligned_mocap[:, 1])
    _py = np.nanpercentile(_ey_arr, 95)
    _ey_p = np.nanmax(_ey_arr[_ey_arr <= _py])
    _ez_arr = np.abs(transformed_tvecs_1[:, 2] - aligned_mocap[:, 2])
    _pz = np.nanpercentile(_ez_arr, 95)
    _ez_p = np.nanmax(_ez_arr[_ez_arr <= _pz])
    _max_x = np.nanmax(np.abs(transformed_tvecs_1[:, 0] - aligned_mocap[:, 0]))
    _max_y = np.nanmax(np.abs(transformed_tvecs_1[:, 1] - aligned_mocap[:, 1]))
    _max_z = np.nanmax(np.abs(transformed_tvecs_1[:, 2] - aligned_mocap[:, 2]))
    print(f'Max error: X={_max_x}, Y={_max_y}, Z={_max_z}')
    print(f'Max error 95th percentile: X={_ex_p}, Y={_ey_p}, Z={_ez_p}')
    return (
        rvecs_1,
        transformed_tvecs_1,
        tvec_diff_1,
        tvec_transformed_1,
        valid_idx_1,
    )


@app.cell
def _(tvec_diff_1):
    tvec_diff_1
    return


@app.cell
def _(R_opt, R_scipy, cv2, mocap_df, np, rvecs_1, valid_idx_1):
    mocap_quats_all = mocap_df[['rb_ang_x', 'rb_ang_y', 'rb_ang_z', 'rb_ang_w']].to_numpy().astype(float)
    quat_norms = np.linalg.norm(mocap_quats_all, axis=1)
    bad_quat_mask = (quat_norms < 1e-06) | np.any(np.isnan(mocap_quats_all), axis=1)
    print(f'Bad quaternion frames (dropout): {bad_quat_mask.sum()} / {len(mocap_quats_all)}')
    mocap_quats_clean = mocap_quats_all.copy()
    mocap_quats_clean[bad_quat_mask] = np.nan
    mocap_rotmats_all = np.full((len(mocap_quats_clean), 3, 3), np.nan)
    valid_quat_mask = ~bad_quat_mask
    mocap_rotmats_all[valid_quat_mask] = R_scipy.from_quat(mocap_quats_clean[valid_quat_mask]).as_matrix()
    R_mocap_0 = mocap_rotmats_all[0]
    mocap_rotmats_rel = np.full_like(mocap_rotmats_all, np.nan)
    for _i in range(len(mocap_rotmats_all)):
        if not np.isnan(mocap_rotmats_all[_i]).any():
            mocap_rotmats_rel[_i] = R_mocap_0.T @ mocap_rotmats_all[_i]
    R_aruco_ref = cv2.Rodrigues(rvecs_1[valid_idx_1])[0]
    aruco_rotmats_rel = np.full((len(rvecs_1), 3, 3), np.nan)
    for _i, rv in enumerate(rvecs_1):
        if not np.isnan(rv).any():
            Ri = cv2.Rodrigues(rv)[0]
            aruco_rotmats_rel[_i] = R_aruco_ref.T @ Ri
    aruco_rotmats_aligned = np.full_like(aruco_rotmats_rel, np.nan)
    for _i in range(len(aruco_rotmats_rel)):
        if not np.isnan(aruco_rotmats_rel[_i]).any():
            aruco_rotmats_aligned[_i] = R_opt @ aruco_rotmats_rel[_i]

    def geodesic_error_deg(R1, R2):
        _R_diff = R1.T @ R2
        cos_angle = np.clip((np.trace(_R_diff) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    n_frames = min(len(aruco_rotmats_aligned), len(mocap_rotmats_rel))
    print(f'ArUco frames: {len(aruco_rotmats_aligned)}, Mocap frames: {len(mocap_rotmats_rel)}, Using: {n_frames}')
    angle_errors = np.full(n_frames, np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_aligned[_i]).any() and (not np.isnan(mocap_rotmats_rel[_i]).any()):
            angle_errors[_i] = geodesic_error_deg(aruco_rotmats_aligned[_i], mocap_rotmats_rel[_i])
    euler_errors = np.full((n_frames, 3), np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_aligned[_i]).any() and (not np.isnan(mocap_rotmats_rel[_i]).any()):
            _R_diff = aruco_rotmats_aligned[_i].T @ mocap_rotmats_rel[_i]
            euler_errors[_i] = R_scipy.from_matrix(_R_diff).as_euler('xyz', degrees=True)
    _mean_rot_err = np.nanmean(angle_errors)
    _median_rot_err = np.nanmedian(angle_errors)
    _p95_rot_err = np.nanpercentile(angle_errors, 95)
    _max_rot_err = np.nanmax(angle_errors)
    print(f'Orientation error (geodesic):')
    print(f'  Mean:   {_mean_rot_err:.3f}°')
    print(f'  Median: {_median_rot_err:.3f}°')
    print(f'  95th %: {_p95_rot_err:.3f}°')
    print(f'  Max:    {_max_rot_err:.3f}°')
    euler_errors = np.full((len(aruco_rotmats_aligned), 3), np.nan)
    for _i in range(len(aruco_rotmats_aligned)):
        if not np.isnan(aruco_rotmats_aligned[_i]).any():
            _R_diff = aruco_rotmats_aligned[_i].T @ mocap_rotmats_rel[_i]
            euler_errors[_i] = R_scipy.from_matrix(_R_diff).as_euler('xyz', degrees=True)
    print(f'\nPer-axis Euler error (mean absolute):')
    print(f'  X: {np.nanmean(np.abs(euler_errors[:, 0])):.3f}°')
    print(f'  Y: {np.nanmean(np.abs(euler_errors[:, 1])):.3f}°')
    print(f'  Z: {np.nanmean(np.abs(euler_errors[:, 2])):.3f}°')
    return (
        aruco_rotmats_aligned,
        geodesic_error_deg,
        mocap_rotmats_rel,
        n_frames,
    )


@app.cell
def _(R_scipy, aruco_rotmats_aligned, mocap_rotmats_rel, n_frames, np, plt):
    mocap_euler = np.full((n_frames, 3), np.nan)
    for _i in range(n_frames):
        if not np.isnan(mocap_rotmats_rel[_i]).any():
            mocap_euler[_i] = R_scipy.from_matrix(mocap_rotmats_rel[_i]).as_euler('xyz', degrees=True)
    aruco_euler = np.full((n_frames, 3), np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_aligned[_i]).any():
            aruco_euler[_i] = R_scipy.from_matrix(aruco_rotmats_aligned[_i]).as_euler('xyz', degrees=True)
    _fig, _axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    _colors_mocap = ['#e74c3c', '#27ae60', '#2980b9']
    _colors_aruco = ['#f1948a', '#82e0aa', '#85c1e9']
    for _ax, _j, _label, _cm, _ca in zip(_axes, range(3), _axis_labels, _colors_mocap, _colors_aruco):
        _ax.plot(mocap_euler[:, _j], color=_cm, label='Mocap', linewidth=1.2)
        _ax.plot(aruco_euler[:, _j], color=_ca, label='ArUco', linewidth=1.0, linestyle='--')
        _ax.set_ylabel(f'{_label} (°)')
        _ax.legend(loc='upper right')
        _ax.grid(True, alpha=0.3)
        print(f'{_label}  |  Mocap range: [{np.nanmin(mocap_euler[:, _j]):.1f}°, {np.nanmax(mocap_euler[:, _j]):.1f}°]  |  ArUco range: [{np.nanmin(aruco_euler[:, _j]):.1f}°, {np.nanmax(aruco_euler[:, _j]):.1f}°]')
    _axes[-1].set_xlabel('Frame')
    _fig.suptitle('Mocap vs ArUco — Raw Euler Angles (xyz)', fontsize=13)
    plt.tight_layout()
    plt.show()
    return (mocap_euler,)


@app.cell
def _(
    R_scipy,
    aruco_rotmats_aligned,
    geodesic_error_deg,
    mocap_euler,
    mocap_rotmats_rel,
    n_frames,
    np,
    plt,
):
    # ─────────────────────────────────────────────────────────────────
    # Find the fixed frame offset between ArUco and mocap orientations
    # Use mean rotation over all valid frames (robust vs single frame)
    valid_frames = [_i for _i in range(n_frames) if not np.isnan(aruco_rotmats_aligned[_i]).any() and (not np.isnan(mocap_rotmats_rel[_i]).any())]
    R_diffs = np.array([aruco_rotmats_aligned[_i].T @ mocap_rotmats_rel[_i] for _i in valid_frames])
    R_offset = R_scipy.from_matrix(R_diffs).mean().as_matrix()
    euler_offset = R_scipy.from_matrix(R_offset).as_euler('xyz', degrees=True)
    print(f'Estimated frame offset (xyz Euler):')
    print(f'  X: {euler_offset[0]:.2f}°  Y: {euler_offset[1]:.2f}°  Z: {euler_offset[2]:.2f}°')
    aruco_rotmats_corrected = np.full_like(aruco_rotmats_aligned, np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_aligned[_i]).any():
            aruco_rotmats_corrected[_i] = aruco_rotmats_aligned[_i] @ R_offset
    aruco_euler_corrected = np.full((n_frames, 3), np.nan)
    for _i in range(n_frames):
    # scipy's mean() averages rotations correctly on SO(3) manifold
        if not np.isnan(aruco_rotmats_corrected[_i]).any():
            aruco_euler_corrected[_i] = R_scipy.from_matrix(aruco_rotmats_corrected[_i]).as_euler('xyz', degrees=True)
    _fig, _axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    _axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    _colors_mocap = ['#e74c3c', '#27ae60', '#2980b9']
    _colors_aruco = ['#f1948a', '#82e0aa', '#85c1e9']
    for _ax, _j, _label, _cm, _ca in zip(_axes, range(3), _axis_labels, _colors_mocap, _colors_aruco):
    # Apply offset correction to ArUco rotations
        _ax.plot(mocap_euler[:, _j], color=_cm, label='Mocap', linewidth=1.2)
        _ax.plot(aruco_euler_corrected[:, _j], color=_ca, label='ArUco (corrected)', linewidth=1.0, linestyle='--')
        _ax.set_ylabel(f'{_label} (°)')
        _ax.legend(loc='upper right')
        _ax.grid(True, alpha=0.3)
        print(f'{_label}  |  Mocap: [{np.nanmin(mocap_euler[:, _j]):.1f}°, {np.nanmax(mocap_euler[:, _j]):.1f}°]  |  ArUco corrected: [{np.nanmin(aruco_euler_corrected[:, _j]):.1f}°, {np.nanmax(aruco_euler_corrected[:, _j]):.1f}°]')
    _axes[-1].set_xlabel('Frame')
    # Re-extract Euler angles and re-plot to verify alignment
    _fig.suptitle('Mocap vs ArUco — After Frame Offset Correction', fontsize=13)
    plt.tight_layout()
    plt.show()
    angle_errors_corrected = np.full(n_frames, np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_corrected[_i]).any() and (not np.isnan(mocap_rotmats_rel[_i]).any()):
            angle_errors_corrected[_i] = geodesic_error_deg(aruco_rotmats_corrected[_i], mocap_rotmats_rel[_i])
    print(f'\nOrientation error after correction (geodesic):')
    print(f'  Mean:   {np.nanmean(angle_errors_corrected):.3f}°')
    print(f'  Median: {np.nanmedian(angle_errors_corrected):.3f}°')
    print(f'  95th %: {np.nanpercentile(angle_errors_corrected, 95):.3f}°')
    # Recompute geodesic errors with corrected rotations
    print(f'  Max:    {np.nanmax(angle_errors_corrected):.3f}°')
    return angle_errors_corrected, aruco_rotmats_corrected


@app.cell
def _(angle_errors_corrected, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))
    valid_errors = angle_errors_corrected[~np.isnan(angle_errors_corrected)]
    # ── Raw data with outliers flagged ──────────────────────────────
    _p95 = np.nanpercentile(angle_errors_corrected, 95)
    clean_errors = valid_errors[valid_errors <= _p95]
    n_outliers = np.sum(valid_errors > _p95)
    n_total = len(valid_errors)
    _bp1 = _axes[0].boxplot(valid_errors, patch_artist=True, medianprops=dict(color='black', linewidth=2), flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=3, alpha=0.4), boxprops=dict(facecolor='#aed6f1', alpha=0.8), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
    _axes[0].set_title('Full Distribution\n(including outliers)', fontsize=11)
    _axes[0].set_ylabel('Geodesic Orientation Error (°)')
    # ── Left: full distribution ──────────────────────────────────────
    _axes[0].set_xticks([1])
    _axes[0].set_xticklabels(['ArUco vs Mocap'])
    _axes[0].axhline(np.mean(valid_errors), color='#e74c3c', linestyle='--', linewidth=1.2, label=f'Mean: {np.mean(valid_errors):.1f}°')
    _axes[0].legend(fontsize=9)
    _axes[0].text(1.32, np.percentile(valid_errors, 75), f'Q3: {np.percentile(valid_errors, 75):.1f}°\nQ1: {np.percentile(valid_errors, 25):.1f}°\nMedian: {np.median(valid_errors):.1f}°', fontsize=8.5, va='center', color='#2c3e50')
    _bp2 = _axes[1].boxplot(clean_errors, patch_artist=True, medianprops=dict(color='black', linewidth=2), flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=3, alpha=0.4), boxprops=dict(facecolor='#a9dfbf', alpha=0.8), whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
    _axes[1].set_title(f'≤95th Percentile (≤{_p95:.1f}°)\n{n_outliers} outlier frames removed ({100 * n_outliers / n_total:.1f}%)', fontsize=11)
    _axes[1].set_ylabel('Geodesic Orientation Error (°)')
    _axes[1].set_xticks([1])
    _axes[1].set_xticklabels(['ArUco vs Mocap'])
    _axes[1].axhline(np.mean(clean_errors), color='#e74c3c', linestyle='--', linewidth=1.2, label=f'Mean: {np.mean(clean_errors):.1f}°')
    _axes[1].legend(fontsize=9)
    _axes[1].text(1.32, np.percentile(clean_errors, 75), f'Q3: {np.percentile(clean_errors, 75):.1f}°\nQ1: {np.percentile(clean_errors, 25):.1f}°\nMedian: {np.median(clean_errors):.1f}°', fontsize=8.5, va='center', color='#2c3e50')
    _fig.suptitle('Orientation Error: ArUco vs Motion Capture\n(Geodesic Angular Distance)', fontsize=13, fontweight='bold')
    # Annotate stats
    plt.tight_layout()
    plt.show()
    print(f'\nFull distribution  — Mean: {np.mean(valid_errors):.2f}°  Median: {np.median(valid_errors):.2f}°')
    print(f'≤95th percentile   — Mean: {np.mean(clean_errors):.2f}°  Median: {np.median(clean_errors):.2f}°')
    # ── Right: 95th percentile clipped ──────────────────────────────
    print(f'Outliers (>{_p95:.1f}°): {n_outliers} frames ({100 * n_outliers / n_total:.1f}%)')
    return


@app.cell
def _(
    R_scipy,
    angle_errors_corrected,
    aruco_rotmats_corrected,
    mocap_rotmats_rel,
    n_frames,
    np,
    plt,
):
    # ─────────────────────────────────────────────────────────────────
    # Recompute per-axis Euler errors using corrected rotations
    euler_errors_corrected = np.full((n_frames, 3), np.nan)
    for _i in range(n_frames):
        if not np.isnan(aruco_rotmats_corrected[_i]).any() and (not np.isnan(mocap_rotmats_rel[_i]).any()):
            _R_diff = aruco_rotmats_corrected[_i].T @ mocap_rotmats_rel[_i]
            euler_errors_corrected[_i] = R_scipy.from_matrix(_R_diff).as_euler('xyz', degrees=True)
    _axis_labels = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    colors_box = ['#aed6f1', '#a9dfbf', '#f9e79f']
    colors_mean = ['#2980b9', '#27ae60', '#d4ac0d']
    p95_geodesic = np.nanpercentile(angle_errors_corrected, 95)
    # Build per-axis clipped arrays (≤95th percentile of abs error)
    data_full = []
    data_clean = []
    for _j in range(3):
        _raw = euler_errors_corrected[:, _j]
        valid = _raw[~np.isnan(_raw)]
        p95_j = np.nanpercentile(np.abs(valid), 95)
        clean = valid[np.abs(valid) <= p95_j]
        data_full.append(valid)  # raw signed errors per axis
        data_clean.append(clean)  # ≤95th percentile of |error|
    _fig, _axes = plt.subplots(2, 4, figsize=(18, 9))
    _fig.suptitle('Orientation Error: ArUco vs Motion Capture', fontsize=14, fontweight='bold')
    valid_geo = angle_errors_corrected[~np.isnan(angle_errors_corrected)]
    p95_geo = np.nanpercentile(valid_geo, 95)
    clean_geo = valid_geo[valid_geo <= p95_geo]
    n_out_geo = np.sum(valid_geo > p95_geo)
    all_data_full = [valid_geo] + data_full
    all_data_clean = [clean_geo] + data_clean
    col_labels = ['Geodesic\n(Overall)', 'X (Roll)', 'Y (Pitch)', 'Z (Yaw)']
    col_colors = ['#d7bde2', '#aed6f1', '#a9dfbf', '#f9e79f']
    # Plot: 4 columns — Geodesic + Roll + Pitch + Yaw
    # Each column: full (top) and ≤95th (bottom)
    col_means = ['#7d3c98', '#2980b9', '#27ae60', '#d4ac0d']
    for _col in range(4):
        for _row, _data in enumerate([all_data_full[_col], all_data_clean[_col]]):
            _ax = _axes[_row, _col]
            bp = _ax.boxplot(_data, patch_artist=True, medianprops=dict(color='black', linewidth=2), flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=2.5, alpha=0.35), boxprops=dict(facecolor=col_colors[_col], alpha=0.85), whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4))
            mean_val = np.mean(_data)
            _ax.axhline(mean_val, color=col_means[_col], linestyle='--', linewidth=1.3, label=f'Mean: {mean_val:.1f}°')
            if _col > 0:
                _ax.axhline(0, color='grey', linestyle=':', linewidth=1.0, alpha=0.6)
            q1, _med, q3 = np.percentile(_data, [25, 50, 75])
            _ax.text(1.38, q3, f'Q3: {q3:.1f}°\nMed: {_med:.1f}°\nQ1: {q1:.1f}°', fontsize=7.5, va='center', color='#2c3e50')
            _ax.set_xticks([1])
            _ax.set_xticklabels([col_labels[_col]], fontsize=9)
            _ax.set_ylabel('Error (°)', fontsize=8)
            _ax.legend(fontsize=8, loc='upper right')
            _ax.grid(True, alpha=0.2, axis='y')
            if _row == 0:
                _ax.set_title(f'{col_labels[_col]}\nFull distribution', fontsize=9)
            else:
                p95_val = np.nanpercentile(np.abs(all_data_full[_col]), 95)
                n_out = np.sum(np.abs(all_data_full[_col]) > p95_val)
                n_tot = len(all_data_full[_col])
                _ax.set_title(f'≤95th %ile (±{p95_val:.1f}°)\n{n_out} outliers removed ({100 * n_out / n_tot:.1f}%)', fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"{'Axis':<12} {'Mean abs':>10} {'Median abs':>12} {'95th %ile':>10} {'Max abs':>10}")
    print('─' * 58)
    names = ['Geodesic', 'Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    for name, full in zip(names, all_data_full):
        abs_data = np.abs(full)
    # Summary table
        print(f'{name:<12} {np.mean(abs_data):>9.2f}°  {np.median(abs_data):>10.2f}°  {np.nanpercentile(abs_data, 95):>8.2f}°  {np.max(abs_data):>8.2f}°')  # Zero reference line for signed Euler plots  # Stats annotation
    return (euler_errors_corrected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## since NOARK + is plannar, we only need to see, y and z, the axis is kind of flipped.
    """)
    return


@app.cell
def _(ar_df_2):
    ar_df_2['time']
    return


@app.cell
def _(aligned_mocap, mocap_x, np, tvec_transformed_1):
    print('Z', np.nanmax(np.abs(tvec_transformed_1[:, 2] - aligned_mocap[:, 2])))
    print('X', np.nanmax(np.abs(tvec_transformed_1[:, 0] - mocap_x)))
    return


@app.cell
def _(aligned_mocap, ar_df_2, pl, transformed_tvecs_1):
    # ── Convert ar_df['time'] to seconds from start ──────────────────
    ar_time_us = ar_df_2['time'].cast(pl.Int64).to_numpy()  # microseconds as int
    ar_time_s = (ar_time_us - ar_time_us[0]) / 1000000.0  # seconds from t=0
    n_traj = min(len(ar_time_s), len(transformed_tvecs_1), len(aligned_mocap))
    # Trim to match trajectory array length (may differ by 1 frame)
    t_axis = ar_time_s[:n_traj]
    return n_traj, t_axis


@app.cell
def _(
    aligned_mocap,
    euler_errors_corrected,
    n_frames,
    n_traj,
    np,
    plt,
    t_axis,
    transformed_tvecs_1,
    tvec_transformed_1,
):
    import matplotlib.gridspec as gridspec
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'font.family': 'serif'})
    pos_data, pos_tick_labels = ([], ['X', 'Y', 'Z'])
    for _col in range(3):
        _raw = np.abs(transformed_tvecs_1[:, _col] - aligned_mocap[:, _col]) * 100
        _raw = _raw[~np.isnan(_raw)]
        _p95 = np.nanpercentile(_raw, 95)
        pos_data.append(_raw[_raw <= _p95])
    ori_data, ori_tick_labels = ([], ['Roll', 'Pitch', 'Yaw'])
    for _j in range(3):
        _raw = euler_errors_corrected[:n_frames, _j]
        _raw = _raw[~np.isnan(_raw)]
        _p95 = np.nanpercentile(np.abs(_raw), 95)
        ori_data.append(_raw[np.abs(_raw) <= _p95])
    _fig = plt.figure(figsize=(18, 14))
    _gs = gridspec.GridSpec(3, 2, figure=_fig, height_ratios=[1, 1, 1], hspace=0.08, wspace=0.32)
    _fig = plt.figure(figsize=(14, 11))
    _gs = gridspec.GridSpec(4, 2, figure=_fig, height_ratios=[1.6, 1, 1, 1], hspace=0.1, wspace=0.3)
    BOX_PROPS = dict(patch_artist=True, widths=0.45, medianprops=dict(color='black', linewidth=2.2), whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4), flierprops=dict(marker='o', markersize=3, alpha=0.35, markerfacecolor='#e74c3c', linestyle='none'))
    ax_pos = _fig.add_subplot(_gs[0, 0])
    _bp1 = ax_pos.boxplot(pos_data, **BOX_PROPS)
    pos_fill = ['#aed6f1', '#a9dfbf', '#f9e79f']
    pos_means = ['#2980b9', '#27ae60', '#d4ac0d']
    for patch, fc in zip(_bp1['boxes'], pos_fill):
        patch.set_facecolor(fc)
        patch.set_alpha(0.88)
    for _j, (mc, _data) in enumerate(zip(pos_means, pos_data)):
        ax_pos.axhline(np.mean(_data), xmin=_j / 3 + 0.04, xmax=(_j + 1) / 3 - 0.04, color=mc, linestyle='--', linewidth=1.6)
    ax_pos.set_xticks([1, 2, 3])
    ax_pos.set_xticklabels(pos_tick_labels, fontsize=12)
    ax_pos.set_ylabel('Position Error (cm)', fontsize=13)
    ax_pos.set_title('Position Error  (≤95th percentile)', fontsize=13, fontweight='bold', pad=8)
    ax_pos.grid(True, axis='y', alpha=0.25)
    ax_pos.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    for _j, _data in enumerate(pos_data):
        _med = np.median(_data)
        ax_pos.text(_j + 1, ax_pos.get_ylim()[0] if ax_pos.get_ylim()[0] > 0 else 0, f'md={_med:.2f}', ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    ax_ori = _fig.add_subplot(_gs[0, 1])
    _bp2 = ax_ori.boxplot(ori_data, **BOX_PROPS)
    ori_fill = ['#f5cba7', '#d2b4de', '#a9cce3']
    ori_means = ['#ca6f1e', '#7d3c98', '#1a5276']
    for patch, fc in zip(_bp2['boxes'], ori_fill):
        patch.set_facecolor(fc)
        patch.set_alpha(0.88)
    for _j, mc in enumerate(ori_means):
        ax_ori.axhline(np.mean(ori_data[_j]), xmin=_j / 3 + 0.04, xmax=(_j + 1) / 3 - 0.04, color=mc, linestyle='--', linewidth=1.6)
    ax_ori.axhline(0, color='grey', linestyle=':', linewidth=1.1, alpha=0.55)
    ax_ori.set_xticks([1, 2, 3])
    ax_ori.set_xticklabels(ori_tick_labels, fontsize=12)
    ax_ori.set_ylabel('Orientation Error (°)', fontsize=13)
    ax_ori.set_title('Orientation Error  (≤95th percentile)', fontsize=13, fontweight='bold', pad=8)
    ax_ori.grid(True, axis='y', alpha=0.25)
    for _j, _data in enumerate(ori_data):
        _med = np.median(_data)
        ax_ori.text(_j + 1, ax_ori.get_ylim()[0] if ax_ori.get_ylim()[0] > 0 else min((d.min() for d in ori_data)), f'md={_med:.2f}°', ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    traj_cfg = [(tvec_transformed_1[:, 1] * 100, aligned_mocap[:, 1] * 100, 'X', 'tab:blue', 'tab:orange'), (tvec_transformed_1[:, 0] * 100, aligned_mocap[:, 0] * 100, 'Y', 'tab:green', 'tab:red'), (tvec_transformed_1[:, 2] * 100, aligned_mocap[:, 2] * 100, 'Z', 'tab:purple', 'tab:red')]
    traj_axes = []
    for _row, (cam, moc, axis, c_cam, c_moc) in enumerate(traj_cfg):
        _ax = _fig.add_subplot(_gs[_row + 1, :])
        _ax.plot(t_axis, cam[:n_traj], label=f'ArUco {axis}', color=c_cam, linewidth=1.1)
        _ax.plot(t_axis, moc[:n_traj], label=f'MoCap {axis}', color=c_moc, linewidth=1.1, linestyle='--', alpha=0.85)
        _ax.set_ylabel(f'{axis}  (cm)', fontsize=12)
        _ax.legend(loc='upper right', fontsize=10, ncol=2)
        _ax.grid(True, alpha=0.22)
        _ax.margins(x=0.005)
        traj_axes.append(_ax)
    for _ax in traj_axes[:-1]:
        _ax.tick_params(labelbottom=False)
    traj_axes[-1].set_xlabel('Time (s)', fontsize=13)
    _fig.subplots_adjust(top=0.94, bottom=0.06, left=0.07, right=0.97, hspace=0.1, wspace=0.3)
    _fig.suptitle('ArUco Marker Tracking vs Motion Capture — Position & Orientation', fontsize=16, fontweight='bold')
    plt.savefig('mocap_aruco_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('mocap_aruco_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('Saved PDF + PNG')
    return


@app.cell
def _(
    aligned_mocap,
    angle_errors_corrected,
    euler_errors_corrected,
    n_frames,
    np,
    transformed_tvecs_1,
):
    from scipy import stats as scipy_stats

    def print_stats_table(data_list, labels, unit, title):
        col_w = 12
        cols = ['Mean±SD', 'Median', 'P25', 'P75', 'P95', 'Max']
        print(f"\n{'═' * 72}")
        print(f'  {title}')
        print(f"{'═' * 72}")
        print(f"{'Axis':<10}" + ''.join((f'{c:>{col_w}}' for c in cols)))
        print(f"{'─' * 72}")
        for _label, _raw in zip(labels, data_list):
            abs_raw = np.abs(_raw)
            p25, p50, p75, _p95 = np.nanpercentile(abs_raw, [25, 50, 75, 95])
            mean = np.nanmean(abs_raw)
            sd = np.nanstd(abs_raw)
            maxi = np.nanmax(abs_raw)
            print(f"{_label:<10}{f'{mean:.3f}±{sd:.3f}':>{col_w}}{p50:>{col_w}.3f}{p25:>{col_w}.3f}{p75:>{col_w}.3f}{_p95:>{col_w}.3f}{maxi:>{col_w}.3f}")
        print(f"{'─' * 72}")
        pooled = np.abs(np.concatenate(data_list))
        p25, p50, p75, _p95 = np.nanpercentile(pooled, [25, 50, 75, 95])
        mean = np.nanmean(pooled)
        sd = np.nanstd(pooled)
        maxi = np.nanmax(pooled)
        print(f"{'Pooled':<10}{f'{mean:.3f}±{sd:.3f}':>{col_w}}{p50:>{col_w}.3f}{p25:>{col_w}.3f}{p75:>{col_w}.3f}{_p95:>{col_w}.3f}{maxi:>{col_w}.3f}")
        print(f"{'═' * 72}")
        print(f'  All values in {unit}. Mean±SD and percentiles computed on |error|.')
        print(f'  n frames per axis: {[len(d) for d in data_list]}')
    pos_data_full = []
    for _col in range(3):
        _raw = (transformed_tvecs_1[:, _col] - aligned_mocap[:, _col]) * 100
        pos_data_full.append(_raw[~np.isnan(_raw)])
    print_stats_table(pos_data_full, labels=['X', 'Y', 'Z'], unit='cm', title='POSITION ERROR — ArUco vs Motion Capture')
    ori_data_full = []
    for _j in range(3):
        _raw = euler_errors_corrected[:n_frames, _j]
        ori_data_full.append(_raw[~np.isnan(_raw)])
    print_stats_table(ori_data_full, labels=['Roll', 'Pitch', 'Yaw'], unit='degrees', title='ORIENTATION ERROR — ArUco vs Motion Capture (after frame offset correction)')
    geo_valid = angle_errors_corrected[~np.isnan(angle_errors_corrected)]
    p25, p50, p75, _p95 = np.nanpercentile(geo_valid, [25, 50, 75, 95])
    print(f"\n{'═' * 72}")
    print(f'  GEODESIC ORIENTATION ERROR (overall, degrees)')
    print(f"{'═' * 72}")
    print(f'  Mean ± SD : {np.mean(geo_valid):.3f} ± {np.std(geo_valid):.3f}°')
    print(f'  Median    : {p50:.3f}°')
    print(f'  P25 / P75 : {p25:.3f}° / {p75:.3f}°')
    print(f'  P95       : {_p95:.3f}°')
    print(f'  Max       : {np.max(geo_valid):.3f}°')
    print(f'  n frames  : {len(geo_valid)}')
    print(f"{'═' * 72}")
    return


@app.cell
def _(ar_df_2):
    ar_df_2
    return


@app.cell
def _(aligned_mocap, plt, tvec_transformed_1):
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    _axes[0].plot(tvec_transformed_1[:, 1], label='cam x', color='tab:blue')
    _axes[0].plot(aligned_mocap[:, 1], label='mocap x', color='tab:orange', linestyle='--')
    _axes[0].set_ylabel('Translation (X)')
    _axes[0].legend(loc='upper right')
    _axes[0].grid(True, alpha=0.3)
    _axes[1].plot(tvec_transformed_1[:, 0], label='cam y', color='tab:green')
    _axes[1].plot(aligned_mocap[:, 0], label='mocap y', color='tab:red', linestyle='--')
    _axes[1].set_ylabel('Translation (Y)')
    _axes[1].legend(loc='upper right')
    _axes[1].grid(True, alpha=0.3)
    _axes[2].plot(tvec_transformed_1[:, 2], label='cam z', color='tab:purple')
    _axes[2].plot(aligned_mocap[:, 2], label='mocap z', color='tab:red', linestyle='--')
    _axes[2].set_ylabel('Translation (Z)')
    _axes[2].set_xlabel('Frames / Time')
    _axes[2].legend(loc='upper right')
    _axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('Camera vs Mocap Comparison', y=1.02, fontsize=14)
    plt.show()
    return


@app.cell
def _():
    # plt.plot(np.array(t_dict["z"]), label='AR X')
    # plt.plot(mocap_ip["z"], label='Mocap X')
    # plt.legend()
    return


@app.cell
def _(error_dict):
    backup_dict = error_dict.copy()
    return


@app.cell
def _(error_dict, np):
    np.min(error_dict['err_x'])
    return


@app.cell
def _(mo):
    mo.stop(True, mo.md("**Execution halted — cells below will not run**"))
    return


@app.cell
def _(min_index, my_dict_1):
    import toml
    _data = toml.load('E:\\CMC\\pyprojects\\programs_rpi\\NOARK_backbone\\notebooks\\calibration\\output\\template.toml')
    _data['calibration']['camera_matrix'] = my_dict_1['mat'][min_index].tolist()
    _data['calibration']['dist_coeffs'] = my_dict_1['dist'][min_index].tolist()
    _data['camera']['resolution'] = (1280, 800)
    with open('E:\\CMC\\pyprojects\\programs_rpi\\NOARK_backbone\\notebooks\\calibration\\output\\good.toml', 'w') as _f:
        toml.dump(_data, _f)
    return (toml,)


@app.cell
def _(min_index):
    min_index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## estimation
    """)
    return


@app.cell
def _():
    # with open(os.path.join(_webcam_calib_video), "rb") as _f:
    #     data = list(mp.Unpacker(_f, object_hook=mpn.decode))
    # img_size = data[0]
    # video_data = data[1:]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check minimum graph
    """)
    return


@app.cell
def _(ar_df_2, mocap_ip, np, plt, t_dict):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].plot(ar_df_2['time'], mocap_ip['x'], label='mocap x')
    _axes[0].plot(ar_df_2['time'], t_dict['x'], label='ar x')
    _axes[0].set_xlabel('time')
    _axes[0].set_ylabel('x in meters')
    _axes[0].legend()
    _axes[1].plot(ar_df_2['time'], mocap_ip['y'], label='mocap y')
    _axes[1].plot(ar_df_2['time'], t_dict['y'], label='aruco y')
    _axes[1].set_xlabel('time')
    _axes[1].set_ylabel('y in meters')
    _axes[1].legend()
    _axes[2].plot(ar_df_2['time'], mocap_ip['z'], label='mocap z')
    _axes[2].plot(ar_df_2['time'], np.array(t_dict['z']), label='aruco z')
    _axes[2].set_xlabel('time')
    _axes[2].set_ylabel('z in meters')
    _axes[2].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(error_dict, np):
    _cumulative_error = (
        np.array(error_dict["max_x"])
        + np.array(error_dict["max_y"])
        + np.array(error_dict["max_z"])
    )
    np.argmin(_cumulative_error)
    return


@app.cell
def _(error_dict, index):
    mtx1 = error_dict["mtx"][index]
    dist1 = error_dict["dist_coeffs"][index]
    return dist1, mtx1


@app.cell
def _(error_dict, index):
    error_dict["dist_coeffs"][index]
    return


@app.cell
def _(dist1, mtx1, toml):
    _data = toml.load('E:\\CMC\\pyprojects\\programs_rpi\\rpi_python\\calib_test.toml')
    _data['calibration']['camera_matrix'] = mtx1.tolist()
    _data['calibration']['dist_coeffs'] = dist1.tolist()
    _data['camera']['resolution'] = (1200, 480)
    with open('E:\\CMC\\pyprojects\\programs_rpi\\rpi_python\\calib_mono_faith3D.toml', 'w') as _f:
        toml.dump(_data, _f)
    return


@app.cell
def _(
    R,
    corners,
    cv2,
    default_ids,
    error_dict,
    estimate_pose_single_markers,
    ids,
    np,
):
    key = 'mean_err'
    index = np.argmin(error_dict[key])
    index = 65
    camera_matrix = error_dict['mtx'][index]
    dist_coeffs = error_dict['dist_coeffs'][index]
    rvecs_2 = []
    _tvecs = []
    for _c, _i in zip(corners, ids):
        if (_i is not None and len(_i) > 0) and all((item in default_ids for item in np.array(_i))):
            _rotation_vectors, _translation_vectors = estimate_pose_single_markers(corners=_c, marker_size=0.05, camera_matrix=camera_matrix, distortion_coefficients=dist_coeffs)
            rvecs_2.append(_rotation_vectors[0][0])
            _tvecs.append(_translation_vectors[0][0])
        else:
            rvecs_2.append(np.array([np.nan, np.nan, np.nan]))
            _tvecs.append(np.array([np.nan, np.nan, np.nan]))
    _tvecs = np.array(_tvecs)
    rvecs_2 = np.array(rvecs_2)
    t_dict = {'x': [], 'y': [], 'z': []}
    _rmat = cv2.Rodrigues(rvecs_2[1])[0]
    ang_dict = {'rx': [], 'ry': [], 'rz': []}
    for _i in range(len(_tvecs)):
        _tvec = _rmat.T @ (_tvecs[_i].reshape(3, 1) - _tvecs[1].reshape(3, 1))
        _tvec = _tvec.T[0]
        t_dict['x'].append(_tvec[0])
        t_dict['y'].append(_tvec[1])
        t_dict['z'].append(_tvec[2])
        _ang = R.from_matrix(_rmat.T @ R.from_rotvec(rvecs_2[_i]).as_matrix()).as_euler('XYZ', degrees=True)
        ang_dict['rx'].append(_ang[0])
        ang_dict['ry'].append(_ang[1])
        ang_dict['rz'].append(_ang[2])
    return index, t_dict


@app.cell
def _(index):
    index
    return


@app.cell
def _(ar_df_2, mocap_ip, np, plt, t_dict):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].plot(ar_df_2['time'], mocap_ip['x'], label='mocap x')
    _axes[0].plot(ar_df_2['time'], t_dict['x'], label='ar x')
    _axes[0].set_xlabel('time')
    _axes[0].set_ylabel('x in meters')
    _axes[0].legend()
    _axes[1].plot(ar_df_2['time'], mocap_ip['z'], label='mocap z')
    _axes[1].plot(ar_df_2['time'], np.array(t_dict['z']), label='aruco z')
    _axes[1].set_xlabel('time')
    _axes[1].set_ylabel('z in meters')
    _axes[1].legend()
    _axes[2].plot(ar_df_2['time'], mocap_ip['y'], label='mocap y')
    _axes[2].plot(ar_df_2['time'], t_dict['y'], label='aruco y')
    _axes[2].set_xlabel('time')
    _axes[2].set_ylabel('y in meters')
    _axes[2].legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
