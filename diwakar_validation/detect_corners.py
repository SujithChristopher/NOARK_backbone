"""Re-detect tag corners from the recorded raw (fisheye) crops, with and without contour refinement.

OpenCV ArUco DICT_APRILTAG_36h11 on each crop, CORNER_REFINE_NONE and CORNER_REFINE_CONTOUR, or rapidtag
(>= 0.1.12) with CORNER_REFINE_APRILTAG (mode "april"), then
cv2.fisheye.undistortPoints into the undistorted pinhole image (same space as the logged corners.csv).
Only tag ids of the jig are kept. Writes out/corners/<take>_<mode>.csv in corners.csv format.
Usage: python detect_corners.py [--modes none,contour,april] [take ...]  (default: none,contour; all non-T0 takes)
"""

import json
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import rapidtag

from common import DATA, OUT, UV_COLS, take_dir

MODES = {"none": cv2.aruco.CORNER_REFINE_NONE, "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
         "april": rapidtag.CORNER_REFINE_APRILTAG}
TAGS = {4, 8, 12, 14, 20, 24, 28, 32}
(OUT / "corners").mkdir(exist_ok=True)


class _Rapid:
    """rapidtag detector with the cv2.aruco.ArucoDetector.detectMarkers return shape."""

    def __init__(self, mode):
        self.p = rapidtag.DetectorParameters()
        self.p.corner_refinement_method = mode

    def detectMarkers(self, img):  # noqa: N802
        cs, ids = rapidtag.detect_markers(img, "DICT_APRILTAG_36h11", self.p)
        if not ids:
            return [], None, None
        return [np.asarray(q, np.float32) for q in cs], np.asarray(ids), None


def run(args):
    name, modes = args
    d = take_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    lens = [(np.array(meta["lens"][f"camera_matrix_cam{k}"]), np.array(meta["lens"][f"dist_coeffs_cam{k}"]),
             np.array(meta["undistorted"][f"camera_matrix_cam{k}"])) for k in (0, 1)]
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    dets = {}
    for m in modes:
        if m == "april":
            dets[m] = _Rapid(MODES[m])
            continue
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = MODES[m]
        dets[m] = cv2.aruco.ArucoDetector(dic, p)
    im = pd.read_csv(d / "images.csv")
    files = {k: open(d / f"images_cam{k}.bin", "rb") for k in (0, 1)}
    rows = {m: [] for m in modes}
    for r in im.itertuples(index=False):
        f = files[r.cam]
        f.seek(r.offset)
        img = np.frombuffer(f.read(r.width * r.height), np.uint8).reshape(r.height, r.width)
        K, D, P = lens[r.cam]
        for m, det in dets.items():
            cs, ids, _ = det.detectMarkers(img)
            if ids is None:
                continue
            for q, i in zip(cs, ids.ravel()):
                if int(i) not in TAGS:
                    continue
                uv = cv2.fisheye.undistortPoints(q.reshape(-1, 1, 2).astype(np.float64) + [r.x0, r.y0], K, D, P=P)
                rows[m].append((r.sample, r.cam, int(i), *uv.ravel()))
    out = {}
    for m in modes:
        df = pd.DataFrame(rows[m], columns=["sample", "cam", "marker_id", *UV_COLS])
        df.to_csv(OUT / "corners" / f"{name}_{m}.csv", index=False, float_format="%.4f")
        out[m] = len(df)
    logged = sum(1 for _ in open(d / "corners.csv")) - 1
    return name, logged, out


if __name__ == "__main__":
    argv = sys.argv[1:]
    modes = ["none", "contour"]
    if argv[:1] == ["--modes"]:
        modes, argv = argv[1].split(","), argv[2:]
    takes = argv or sorted(p.name for p in DATA.glob("2026-*") if "_T0_" not in p.name)
    # big takes first so the pool stays busy
    takes.sort(key=lambda t: -(take_dir(t) / "images_cam0.bin").stat().st_size)
    with Pool(8) as pool:
        for name, logged, out in pool.imap_unordered(run, [(t, modes) for t in takes]):
            print(name, "logged", logged, out, flush=True)
