# %% Imports
import os

import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from pathlib import Path

# %% Path defs

project_root = Path(__file__).parents[3]
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

# %% Helpers


def get_metadata(metaf):
    f = open(metaf, "rb")
    _ = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync, timestamps = _[:, 0], _[:, 1]
    sync = sync.astype(int).astype(bool)
    timestamp_dt = timestamps.astype("datetime64[us]")
    return sync, timestamp_dt


def file_size_mb(path):
    return os.path.getsize(path) / (1024**2)


def fmt_size(mb):
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


# %% Load metadata

sync_cam0, timestamp_cam0 = get_metadata(cam_imx219_meta)
sync_cam1, timestamp_cam1 = get_metadata(cam_ov9281_meta)

# %% Compute stats


def camera_stats(name, timestamps, data_path, meta_path):
    n_frames = len(timestamps)
    duration_us = (timestamps[-1] - timestamps[0]).astype("timedelta64[us]").astype(float)
    duration_s = duration_us / 1e6

    fps = n_frames / duration_s

    data_mb = file_size_mb(data_path)
    meta_mb = file_size_mb(meta_path)
    total_mb = data_mb + meta_mb

    bytes_per_frame = (data_mb * 1024**2) / n_frames

    # Inter-frame intervals in ms
    intervals_us = np.diff(timestamps.astype(np.int64))
    interval_mean_ms = intervals_us.mean() / 1000
    interval_std_ms = intervals_us.std() / 1000

    print(f"\n{'='*50}")
    print(f"  Camera: {name}")
    print(f"{'='*50}")
    print(f"  Frames          : {n_frames}")
    print(f"  Duration        : {duration_s:.2f} s  ({duration_s/60:.2f} min)")
    print(f"  Actual FPS      : {fps:.2f} Hz")
    print(f"  Frame interval  : {interval_mean_ms:.2f} ms ± {interval_std_ms:.2f} ms")
    print(f"  Data file size  : {fmt_size(data_mb)}")
    print(f"  Meta file size  : {fmt_size(meta_mb)}")
    print(f"  Total size      : {fmt_size(total_mb)}")
    print(f"  Bytes per frame : {bytes_per_frame:.0f} B")

    # Projections for 1 hour
    print(f"\n  --- 1-hour projection (data only) ---")
    print(f"  {'Hz':>6}  {'Frames':>10}  {'Size':>10}")
    print(f"  {'-'*30}")
    for hz in [60, 45, 30, 15]:
        frames_1h = hz * 3600
        size_1h_mb = (bytes_per_frame * frames_1h) / (1024**2)
        print(f"  {hz:>6}  {frames_1h:>10,}  {fmt_size(size_1h_mb):>10}")

    return fps, bytes_per_frame


# %% Print stats

fps0, bpf0 = camera_stats("cam0 IMX219", timestamp_cam0, cam_imx219_data, cam_imx219_meta)
fps1, bpf1 = camera_stats("cam1 OV9281", timestamp_cam1, cam_ov9281_data, cam_ov9281_meta)

# %% Combined dual-camera 1-hour projection

print(f"\n{'='*50}")
print("  Dual-camera combined 1-hour projection")
print(f"{'='*50}")
print(f"  {'Hz':>6}  {'Frames':>10}  {'Size':>12}")
print(f"  {'-'*35}")
for hz in [60, 45, 30, 15]:
    frames_1h = hz * 3600
    size_1h_mb = ((bpf0 + bpf1) * frames_1h) / (1024**2)
    print(f"  {hz:>6}  {frames_1h:>10,}  {fmt_size(size_1h_mb):>12}")
