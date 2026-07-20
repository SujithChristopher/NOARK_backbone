# %% Imports
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from tqdm.auto import tqdm

# %% Config

SAMPLE_FRAMES = 150          # frames used for all tests (~2.5s at 60fps)
CHESSBOARD_PATTERN = (8, 12) # cols, rows of inner corners
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# %% Path defs

project_root = Path(__file__).parents[3]
calib_data_folder = (
    project_root / "data" / "calibration" / "dual_160"
    / "dual_cam_calibration_checker_sz_30mm"
)

cam_imx219_data = calib_data_folder / "cam0_imx219.msgpack"
cam_imx219_meta = calib_data_folder / "cam0_timestamp.msgpack"
cam_ov9281_data = calib_data_folder / "cam1_ov9281.msgpack"
cam_ov9281_meta = calib_data_folder / "cam1_timestamp.msgpack"

# %% Helpers — metadata / IO


def get_metadata(metaf):
    with open(metaf, "rb") as f:
        _ = np.array(list(mp.Unpacker(f, object_hook=mpn.decode)))
    sync = _[:, 0].astype(int).astype(bool)
    timestamps = _[:, 1].astype("datetime64[us]")
    return sync, timestamps


def read_sample_frames(data_path, n):
    frames = []
    with open(data_path, "rb") as f:
        for frame in mp.Unpacker(f, object_hook=mpn.decode):
            frames.append(frame)
            if len(frames) >= n:
                break
    return frames


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def fmt_size(mb):
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.2f} MB"


def fmt_ratio(ratio):
    return f"{ratio:.2f}x" if ratio else "—"


# %% Helpers — ffmpeg


def check_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def numpy_pix_fmt(frame):
    if frame.ndim == 2:
        return ("gray16le", False) if frame.dtype == np.uint16 else ("gray", False)
    if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
        return "bgr24", True
    raise ValueError(f"Unsupported frame: shape={frame.shape} dtype={frame.dtype}")


def encode_frames(frames, fps, out_path, extra_args):
    """Pipe raw frames to ffmpeg. Returns output file size in bytes, or None on failure."""
    pix_fmt, _ = numpy_pix_fmt(frames[0])
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "pipe:0",
    ] + extra_args + [str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        return None
    return os.path.getsize(out_path)


def decode_frames(video_path, decode_pix_fmt, n_frames, frame_shape):
    """Decode frames from video via ffmpeg pipe. Returns list of numpy arrays."""
    h, w = frame_shape[:2]
    is_color = len(frame_shape) == 3 and frame_shape[2] == 3
    dtype = np.uint16 if decode_pix_fmt == "gray16le" else np.uint8
    n_ch = 3 if (is_color and decode_pix_fmt == "bgr24") else 1
    frame_bytes = h * w * n_ch * dtype().itemsize

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vframes", str(n_frames),
        "-f", "rawvideo", "-pix_fmt", decode_pix_fmt,
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < frame_bytes:
        return []

    raw = np.frombuffer(r.stdout, dtype=dtype)
    n_actual = len(r.stdout) // frame_bytes
    out = []
    stride = h * w * n_ch
    for i in range(min(n_actual, n_frames)):
        chunk = raw[i * stride: (i + 1) * stride]
        out.append(chunk.reshape((h, w, n_ch) if n_ch == 3 else (h, w)))
    return out


# %% Helpers — chessboard


def to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def detect_corners(frame):
    gray = to_gray(frame)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_PATTERN)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
    return ret, (corners if ret else None)


def detect_corners_batch(frames, label=""):
    results = []
    for frame in tqdm(frames, desc=f"  corners [{label}]", leave=False):
        results.append(detect_corners(frame))
    return results


def corner_stats(raw_det, cmp_det, label):
    """Compare corner detections of a format vs raw. Returns stats dict."""
    n = min(len(raw_det), len(cmp_det))
    raw_hits = [i for i in range(n) if raw_det[i][0]]
    cmp_hits = set(i for i in range(n) if cmp_det[i][0])

    both = [i for i in raw_hits if i in cmp_hits]
    raw_only = [i for i in raw_hits if i not in cmp_hits]
    cmp_only = [i for i in range(n) if cmp_det[i][0] and not raw_det[i][0]]

    # Collect every individual corner error across all matched frames
    all_diffs = []   # (M*N_corners, 2)  where M = len(both)
    for i in both:
        diff = (raw_det[i][1] - cmp_det[i][1])[:, 0, :]  # (N_corners, 2)
        all_diffs.append(diff)

    dev = {}
    if all_diffs:
        diffs = np.concatenate(all_diffs, axis=0)          # (total_corners, 2)
        errors = np.linalg.norm(diffs, axis=1)             # Euclidean per corner
        dev = {
            "mean_px":   float(np.mean(errors)),
            "std_px":    float(np.std(errors)),
            "median_px": float(np.median(errors)),
            "p95_px":    float(np.percentile(errors, 95)),
            "max_px":    float(np.max(errors)),
            "dx_mean_px": float(np.mean(np.abs(diffs[:, 0]))),
            "dy_mean_px": float(np.mean(np.abs(diffs[:, 1]))),
            "n_corners":  len(errors),
        }

    return {
        "label": label,
        "n": n,
        "raw_detected": len(raw_hits),
        "fmt_detected": len(cmp_hits),
        "both": len(both),
        "raw_only": len(raw_only),
        "fmt_only": len(cmp_only),
        **dev,
    }


# %% Main analysis


def analyse_camera(name, data_path, meta_path, timestamps, tmpdir):
    n_frames = len(timestamps)
    dur_us = (timestamps[-1] - timestamps[0]).astype("timedelta64[us]").astype(float)
    duration_s = dur_us / 1e6
    fps = n_frames / duration_s
    intervals_us = np.diff(timestamps.astype(np.int64))

    data_mb = file_size_mb(data_path)
    meta_mb = file_size_mb(meta_path)

    n_sample = min(SAMPLE_FRAMES, n_frames)
    print(f"\n  [{name}] reading {n_sample} sample frames...")
    frames = read_sample_frames(data_path, n_sample)

    f0 = frames[0]
    pix_fmt, is_color = numpy_pix_fmt(f0)
    h, w = f0.shape[:2]
    channels = f0.shape[2] if f0.ndim == 3 else 1

    raw_bpf = f0.nbytes
    mp_bpf = (data_mb * 1024 ** 2) / n_frames

    result = dict(
        name=name, n_frames=n_frames, duration_s=duration_s, fps=fps,
        interval_mean_ms=intervals_us.mean() / 1000,
        interval_std_ms=intervals_us.std() / 1000,
        frame_shape=(h, w, channels), dtype=str(f0.dtype), pix_fmt=pix_fmt,
        raw_bpf=raw_bpf, mp_bpf=mp_bpf,
        data_mb=data_mb, meta_mb=meta_mb,
        raw_total_mb=(raw_bpf * n_frames) / (1024 ** 2),
        ffv1_bpf=None, h264_bpf=None,
        lossless_ffv1=None, lossless_h264=None,
        corner_ffv1=None, corner_h264=None,
    )

    # ── chessboard on raw frames ──────────────────────────────────────────────
    print(f"  [{name}] detecting chessboard corners (raw)...")
    raw_det = detect_corners_batch(frames, "raw")

    if has_ffmpeg:
        tag = name.replace(" ", "_")
        # H.264 lossless pix_fmt: yuv444p for color (truly lossless in Y), gray for mono
        h264_enc_pix = "yuv444p" if is_color else "gray"

        # ── FFV1 ──────────────────────────────────────────────────────────────
        ffv1_path = Path(tmpdir) / f"{tag}_ffv1.mkv"
        print(f"  [{name}] encoding FFV1 ({n_sample} frames)...")
        ffv1_bytes = encode_frames(
            frames, fps, ffv1_path,
            ["-c:v", "ffv1", "-level", "3", "-slices", "16"],
        )
        if ffv1_bytes:
            result["ffv1_bpf"] = ffv1_bytes / n_sample

            # pixel-level lossless check (spot: frame 0)
            decoded_spot = decode_frames(ffv1_path, pix_fmt, 1, f0.shape)
            result["lossless_ffv1"] = (
                np.array_equal(frames[0], decoded_spot[0]) if decoded_spot else False
            )

            # chessboard on FFV1-decoded frames
            print(f"  [{name}] decoding FFV1 → corner detection...")
            ffv1_frames = decode_frames(ffv1_path, pix_fmt, n_sample, f0.shape)
            if ffv1_frames:
                ffv1_det = detect_corners_batch(ffv1_frames, "FFV1")
                result["corner_ffv1"] = corner_stats(raw_det, ffv1_det, "FFV1")

        # ── H.264 lossless ───────────────────────────────────────────────────
        h264_path = Path(tmpdir) / f"{tag}_h264.mp4"
        print(f"  [{name}] encoding H.264 lossless ({n_sample} frames)...")
        h264_bytes = encode_frames(
            frames, fps, h264_path,
            ["-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
             "-pix_fmt", h264_enc_pix],
        )
        if h264_bytes:
            result["h264_bpf"] = h264_bytes / n_sample

            # pixel-level lossless check
            dec_pix = "bgr24" if is_color else "gray"
            decoded_spot = decode_frames(h264_path, dec_pix, 1, f0.shape)
            result["lossless_h264"] = (
                np.array_equal(frames[0], decoded_spot[0]) if decoded_spot else False
            )

            # chessboard on H.264-decoded frames
            print(f"  [{name}] decoding H.264 → corner detection...")
            h264_frames = decode_frames(h264_path, dec_pix, n_sample, f0.shape)
            if h264_frames:
                h264_det = detect_corners_batch(h264_frames, "H.264")
                result["corner_h264"] = corner_stats(raw_det, h264_det, "H.264")

    return result


# %% Run

has_ffmpeg = check_ffmpeg()
if not has_ffmpeg:
    print("WARNING: ffmpeg not found — conversion stats skipped.")

print("Loading metadata...")
_, ts_cam0 = get_metadata(cam_imx219_meta)
_, ts_cam1 = get_metadata(cam_ov9281_meta)

tmpdir = tempfile.mkdtemp(prefix="noark_videocheck_")
try:
    r0 = analyse_camera("cam0 IMX219", cam_imx219_data, cam_imx219_meta, ts_cam0, tmpdir)
    r1 = analyse_camera("cam1 OV9281", cam_ov9281_data, cam_ov9281_meta, ts_cam1, tmpdir)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# %% Print reports

def print_report(r):
    bpf = {
        "Raw":     r["raw_bpf"],
        "Msgpack": r["mp_bpf"],
        "FFV1":    r["ffv1_bpf"],
        "H.264 LL": r["h264_bpf"],
    }
    h, w, c = r["frame_shape"]

    print(f"\n{'=' * 64}")
    print(f"  {r['name']}")
    print(f"{'=' * 64}")
    print(f"  Frame          : {w}x{h}  ch={c}  dtype={r['dtype']}  ({r['pix_fmt']})")
    print(f"  Frames         : {r['n_frames']:,}")
    print(f"  Duration       : {r['duration_s']:.2f} s  ({r['duration_s']/60:.2f} min)")
    print(f"  FPS            : {r['fps']:.2f} Hz")
    print(f"  Interval       : {r['interval_mean_ms']:.2f} ms ± {r['interval_std_ms']:.2f} ms")

    # ── size table ────────────────────────────────────────────────────────────
    print(f"\n  {'Format':<12}  {'Per frame':>11}  {'Full recording':>15}  "
          f"{'vs raw':>8}  {'Lossless':>10}")
    print(f"  {'-' * 62}")

    def size_row(label, _bpf, lossless=None):
        if _bpf is None:
            return
        total = fmt_size((_bpf * r["n_frames"]) / (1024 ** 2))
        ratio = fmt_ratio(r["raw_bpf"] / _bpf) if _bpf else "—"
        ls = {True: "verified", False: "FAIL", None: "—"}.get(lossless, "—")
        print(f"  {label:<12}  {_bpf:>9.0f} B  {total:>15}  {ratio:>8}  {ls:>10}")

    size_row("Raw",      r["raw_bpf"])
    size_row("Msgpack",  r["mp_bpf"])
    size_row("FFV1",     r["ffv1_bpf"],  r["lossless_ffv1"])
    size_row("H.264 LL", r["h264_bpf"],  r["lossless_h264"])

    # ── chessboard table ──────────────────────────────────────────────────────
    print(f"\n  --- Chessboard corner detection vs raw  (sample={SAMPLE_FRAMES} frames) ---")
    print(f"  {'Format':<10}  {'Raw det':>7}  {'Fmt det':>7}  {'Both':>5}  "
          f"{'RawOnly':>7}  {'FmtOnly':>7}")
    print(f"  {'-' * 52}")

    def corner_summary_row(cs):
        if cs is None:
            return
        print(f"  {cs['label']:<10}  {cs['raw_detected']:>7}  {cs['fmt_detected']:>7}  "
              f"{cs['both']:>5}  {cs['raw_only']:>7}  {cs['fmt_only']:>7}")

    corner_summary_row(r["corner_ffv1"])
    corner_summary_row(r["corner_h264"])

    # per-format deviation breakdown
    for cs in [r["corner_ffv1"], r["corner_h264"]]:
        if cs is None or "mean_px" not in cs:
            continue
        n_tot = cs["n_corners"]
        print(f"\n  Corner deviation  [{cs['label']}]  "
              f"({n_tot:,} corners from {cs['both']} frames)")
        print(f"  {'Metric':<22}  {'Value':>12}")
        print(f"  {'-' * 36}")
        rows = [
            ("mean error",    f"{cs['mean_px']:.5f} px"),
            ("std error",     f"{cs['std_px']:.5f} px"),
            ("median error",  f"{cs['median_px']:.5f} px"),
            ("95th pct",      f"{cs['p95_px']:.5f} px"),
            ("max error",     f"{cs['max_px']:.5f} px"),
            ("mean |dx|",     f"{cs['dx_mean_px']:.5f} px"),
            ("mean |dy|",     f"{cs['dy_mean_px']:.5f} px"),
        ]
        for metric, val in rows:
            print(f"  {metric:<22}  {val:>12}")

    # ── 1-hour projection ─────────────────────────────────────────────────────
    print(f"\n  --- 1-hour projection ---")
    header = f"  {'Hz':>4}  {'Frames':>8}  {'Raw':>10}  {'Msgpack':>10}"
    if r["ffv1_bpf"]:  header += f"  {'FFV1':>10}"
    if r["h264_bpf"]:  header += f"  {'H.264 LL':>10}"
    print(header)
    print(f"  {'-' * 70}")
    for hz in [60, 45, 30, 15]:
        f1h = hz * 3600
        row = (f"  {hz:>4}  {f1h:>8,}  {fmt_size((r['raw_bpf']*f1h)/1024**2):>10}"
               f"  {fmt_size((r['mp_bpf']*f1h)/1024**2):>10}")
        if r["ffv1_bpf"]:
            row += f"  {fmt_size((r['ffv1_bpf']*f1h)/1024**2):>10}"
        if r["h264_bpf"]:
            row += f"  {fmt_size((r['h264_bpf']*f1h)/1024**2):>10}"
        print(row)


print_report(r0)
print_report(r1)

# %% Combined dual-camera

print(f"\n{'=' * 64}")
print("  Dual-camera combined 1-hour projection")
print(f"{'=' * 64}")
raw_d  = r0["raw_bpf"]  + r1["raw_bpf"]
mp_d   = r0["mp_bpf"]   + r1["mp_bpf"]
ffv1_d = (r0["ffv1_bpf"] or 0) + (r1["ffv1_bpf"] or 0) or None
h264_d = (r0["h264_bpf"] or 0) + (r1["h264_bpf"] or 0) or None

header = f"  {'Hz':>4}  {'Frames':>8}  {'Raw':>10}  {'Msgpack':>10}"
if ffv1_d: header += f"  {'FFV1':>10}"
if h264_d: header += f"  {'H.264 LL':>10}"
print(header)
print(f"  {'-' * 70}")
for hz in [60, 45, 30, 15]:
    f1h = hz * 3600
    row = (f"  {hz:>4}  {f1h:>8,}  {fmt_size((raw_d*f1h)/1024**2):>10}"
           f"  {fmt_size((mp_d*f1h)/1024**2):>10}")
    if ffv1_d: row += f"  {fmt_size((ffv1_d*f1h)/1024**2):>10}"
    if h264_d: row += f"  {fmt_size((h264_d*f1h)/1024**2):>10}"
    print(row)
