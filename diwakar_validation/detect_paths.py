"""Undistort-the-image vs undistort-the-corners, for every corner refinement, with identical detectors.

Path "ud"  : raw fisheye crop -> remap to the undistorted pinhole image (as the tracker does, see
             rerun_tracker.py) -> detect -> refine. Corners are already pinhole.
Path "raw" : detect -> refine on the raw fisheye crop -> cv2.fisheye.undistortPoints into the same
             pinhole image (meta['undistorted'] K).
Refinements: none, contour (OpenCV), edge (contour + Diwakar's edge_refine.refine_tags_multi),
april (rapidtag CORNER_REFINE_APRILTAG). OpenCV detector = the tracker's settings (adaptive threshold
window 15 only, perspectiveRemovePixelPerCell 4, aruco3 off); rapidtag gets the same threshold window.

Writes out/corners/<take>_<path>_<refine>.csv in corners.csv format.
Usage: python detect_paths.py [take ...]   (default: the 10-03 T0 takes, which build the jigs, and T1)
edge_refine.py comes from Diwakar's tracker (github SujithChristopher/NOARKGames_BKD, tracker/);
set BKD_TRACKER to its folder if it is not at the default below.
"""

import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import rapidtag

from common import OUT, UV_COLS, take_dir

BKD_TRACKER = os.environ.get("BKD_TRACKER", os.path.expandvars(r"%LOCALAPPDATA%\Temp\bkd_repo\NOARKGames_BKD\tracker"))
sys.path.insert(0, BKD_TRACKER)
from edge_refine import refine_tags_multi  # noqa: E402

TAKES = ["2026-10-03_T0_test_r1", "2026-10-03_T0_test_r2", "2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]
PATHS = ["ud", "raw"]
REFINES = ["none", "contour", "edge", "april"]
TAGS = {4, 8, 12, 14, 20, 24, 28, 32}
W, H = 1280, 800
CHUNK = 4000  # images.csv rows per worker job
(OUT / "corners").mkdir(exist_ok=True)


def _opencv(settings, method):
    p = cv2.aruco.DetectorParameters()
    p.useAruco3Detection = bool(settings.get("aruco3_detection", False))
    p.cornerRefinementMethod = method
    p.perspectiveRemovePixelPerCell = int(settings.get("perspective_pixels_per_cell", 4))
    win = int(settings.get("adaptive_thresh_win_size", 15))
    p.adaptiveThreshWinSizeMin = p.adaptiveThreshWinSizeMax = win
    p.adaptiveThreshWinSizeStep = 1
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


class _Rapid:
    """rapidtag with the ArucoDetector.detectMarkers return shape."""

    def __init__(self, settings):
        self.p = rapidtag.DetectorParameters()
        self.p.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
        win = int(settings.get("adaptive_thresh_win_size", 15))
        self.p.adaptive_thresh_win_size_min = self.p.adaptive_thresh_win_size_max = win
        self.p.adaptive_thresh_win_size_step = 1

    def detectMarkers(self, img):  # noqa: N802
        cs, ids = rapidtag.detect_markers(img, "DICT_APRILTAG_36h11", self.p)
        if not ids:
            return [], None, None
        return [np.asarray(q, np.float32).reshape(1, 4, 2) for q in cs], np.asarray(ids), None


def detect_all(dets, img):
    """refine -> list of (marker_id, corners (4, 2)) in img's pixels."""
    out = {}
    cs, ids, _ = dets["none"].detectMarkers(img)
    out["none"] = (cs, ids)
    cs, ids, _ = dets["contour"].detectMarkers(img)
    out["contour"] = (cs, ids)
    out["edge"] = (refine_tags_multi([img], [cs])[0] if ids is not None else cs, ids)
    out["april"] = dets["april"].detectMarkers(img)[:2]
    res = {}
    for r, (cs, ids) in out.items():
        res[r] = [] if ids is None else [(int(i), np.asarray(q, np.float64).reshape(4, 2))
                                         for q, i in zip(cs, np.ravel(ids)) if int(i) in TAGS]
    return res


def run(args):
    name, a, b = args
    d = take_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    s = meta["settings"]
    lens, maps = [], []
    for k in (0, 1):
        K = np.array(meta["lens"][f"camera_matrix_cam{k}"])
        D = np.array(meta["lens"][f"dist_coeffs_cam{k}"]).reshape(4, 1)
        P = np.array(meta["undistorted"][f"camera_matrix_cam{k}"])
        lens.append((K, D, P))
        maps.append(cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_16SC2))
    dets = {"none": _opencv(s, cv2.aruco.CORNER_REFINE_NONE), "contour": _opencv(s, cv2.aruco.CORNER_REFINE_CONTOUR),
            "april": _Rapid(s)}
    im = pd.read_csv(d / "images.csv").iloc[a:b]
    files = {k: open(d / f"images_cam{k}.bin", "rb") for k in (0, 1)}
    rows = {(p, r): [] for p in PATHS for r in REFINES}
    for r in im.itertuples(index=False):
        f = files[r.cam]
        f.seek(r.offset)
        crop = np.frombuffer(f.read(r.width * r.height), np.uint8).reshape(r.height, r.width)
        K, D, P = lens[r.cam]

        # raw path: detect on the fisheye crop, undistort the corner points
        for ref, tags in detect_all(dets, crop).items():
            for i, q in tags:
                uv = cv2.fisheye.undistortPoints((q + [r.x0, r.y0]).reshape(-1, 1, 2), K, D, P=P)
                rows[("raw", ref)].append((r.sample, r.cam, i, *uv.ravel()))

        # ud path: remap the crop (largest undistorted box fed only by crop pixels), detect there
        m1, m2 = maps[r.cam]
        src = m1.astype(np.int32)
        inside = ((src[..., 0] >= r.x0 + 1) & (src[..., 0] < r.x0 + r.width - 2)
                  & (src[..., 1] >= r.y0 + 1) & (src[..., 1] < r.y0 + r.height - 2))
        ys, xs = np.nonzero(inside)
        if not len(xs):
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        while not inside[y0:y1, x0:x1].all() and x1 - x0 > 20 and y1 - y0 > 20:
            x0, x1, y0, y1 = x0 + 1, x1 - 1, y0 + 1, y1 - 1
        full = np.zeros((H, W), np.uint8)
        full[r.y0:r.y0 + r.height, r.x0:r.x0 + r.width] = crop
        ud = cv2.remap(full, m1[y0:y1, x0:x1], m2[y0:y1, x0:x1], interpolation=cv2.INTER_LINEAR)
        for ref, tags in detect_all(dets, ud).items():
            for i, q in tags:
                rows[("ud", ref)].append((r.sample, r.cam, i, *(q + [x0, y0]).ravel()))
    return name, {k: pd.DataFrame(v, columns=["sample", "cam", "marker_id", *UV_COLS]) for k, v in rows.items()}


if __name__ == "__main__":
    takes = sys.argv[1:] or TAKES
    jobs = []
    for t in takes:
        n = sum(1 for _ in open(take_dir(t) / "images.csv")) - 1
        jobs += [(t, a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
    parts = {t: {} for t in takes}
    with Pool(8) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(run, jobs), 1):
            for k, df in res.items():
                parts[name].setdefault(k, []).append(df)
            print(f"{i}/{len(jobs)} {name}", flush=True)
    for t in takes:
        for (p, r), dfs in parts[t].items():
            df = pd.concat(dfs).sort_values(["sample", "cam", "marker_id"], kind="stable")
            df.to_csv(OUT / "corners" / f"{t}_{p}_{r}.csv", index=False, float_format="%.4f")
            print(t, p, r, len(df), flush=True)
