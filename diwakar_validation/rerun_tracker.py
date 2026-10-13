"""Re-run Diwakar's tracker corner pipeline (NOARKGames_BKD tracker/main.py + edge_refine.py) on the
recorded raw crops and compare with the logged corners.csv.

Tracker pipeline per camera frame (settings from meta.json):
  raw fisheye -> cv2.fisheye.initUndistortRectifyMap(K, D, I, K_ud, 1280x800, CV_16SC2) + remap INTER_LINEAR
  -> ArucoDetector DICT_APRILTAG_36h11, CORNER_REFINE_CONTOUR, aruco3 off, perspectiveRemovePixelPerCell 4,
     single adaptive threshold window 15 px
  -> edge_refine.refine_tags_multi (halfway-grey edge lines, corners = line intersections).
The crop is pasted into a blank raw frame, remapped, and searched inside the undistorted box whose pixels all
come from the crop.
Usage: python rerun_tracker.py <bkd_tracker_dir> [take] [max_samples] [--no-edge]
Writes out/corners/<take>_bkd[_noedge].csv in corners.csv format.
"""

import json
import sys

import cv2
import numpy as np
import pandas as pd

from common import OUT, UV_COLS, take_dir

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
EDGE = "--no-edge" not in sys.argv
sys.path.insert(0, argv[0])
from edge_refine import refine_tags_multi  # noqa: E402

NAME = argv[1] if len(argv) > 1 else "2026-10-03_T1_grid_r1"
MAX_N = int(argv[2]) if len(argv) > 2 else 0
TAGS = {4, 8, 12, 14, 20, 24, 28, 32}
W, H = 1280, 800


def detector(s):
    p = cv2.aruco.DetectorParameters()
    p.useAruco3Detection = bool(s.get("aruco3_detection", False))
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    p.perspectiveRemovePixelPerCell = int(s.get("perspective_pixels_per_cell", 4))
    win = int(s.get("adaptive_thresh_win_size", 15))
    p.adaptiveThreshWinSizeMin = p.adaptiveThreshWinSizeMax = win
    p.adaptiveThreshWinSizeStep = 1
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def main():
    d = take_dir(NAME)
    meta = json.loads((d / "meta.json").read_text())
    s = meta["settings"]
    maps = []
    for k in (0, 1):
        K = np.array(meta["lens"][f"camera_matrix_cam{k}"])
        D = np.array(meta["lens"][f"dist_coeffs_cam{k}"]).reshape(4, 1)
        P = np.array(meta["undistorted"][f"camera_matrix_cam{k}"])
        maps.append(cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_16SC2))
    det = [detector(s), detector(s)]
    im = pd.read_csv(d / "images.csv")
    if MAX_N:
        keep = np.unique(im["sample"])[:MAX_N]
        im = im[im["sample"].isin(keep)]
    files = {k: open(d / f"images_cam{k}.bin", "rb") for k in (0, 1)}
    rows = []
    for r in im.itertuples(index=False):
        f = files[r.cam]
        f.seek(r.offset)
        crop = np.frombuffer(f.read(r.width * r.height), np.uint8).reshape(r.height, r.width)
        m1, m2 = maps[r.cam]
        # undistorted pixels whose source lies inside the crop (1 px margin for bilinear)
        src = m1.astype(np.int32)
        inside = ((src[..., 0] >= r.x0 + 1) & (src[..., 0] < r.x0 + r.width - 2)
                  & (src[..., 1] >= r.y0 + 1) & (src[..., 1] < r.y0 + r.height - 2))
        ys, xs = np.nonzero(inside)
        if not len(xs):
            continue
        # largest box fully inside: shrink the bounding box until all of it is inside
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        while not inside[y0:y1, x0:x1].all() and x1 - x0 > 20 and y1 - y0 > 20:
            x0, x1, y0, y1 = x0 + 1, x1 - 1, y0 + 1, y1 - 1
        full = np.zeros((H, W), np.uint8)
        full[r.y0:r.y0 + r.height, r.x0:r.x0 + r.width] = crop
        ud = cv2.remap(full, m1[y0:y1, x0:x1], m2[y0:y1, x0:x1], interpolation=cv2.INTER_LINEAR)
        cs, ids, _ = det[r.cam].detectMarkers(ud)
        if ids is None:
            continue
        if EDGE:
            cs = refine_tags_multi([ud], [cs])[0]
        for q, i in zip(cs, ids.ravel()):
            if int(i) in TAGS:
                rows.append((r.sample, r.cam, int(i), *(np.asarray(q).reshape(4, 2) + [x0, y0]).ravel()))
    df = pd.DataFrame(rows, columns=["sample", "cam", "marker_id", *UV_COLS])
    out = OUT / "corners" / f"{NAME}_bkd{'' if EDGE else '_noedge'}.csv"
    df.to_csv(out, index=False, float_format="%.4f")

    log = pd.read_csv(d / "corners.csv")
    log = log[log["sample"].isin(im["sample"].unique())]
    j = df.merge(log, on=["sample", "cam", "marker_id"], suffixes=("", "_log"))
    e = (j[UV_COLS].to_numpy() - j[[c + "_log" for c in UV_COLS]].to_numpy()).reshape(-1, 4, 2)
    dist = np.linalg.norm(e, axis=2).ravel()
    print(f"{NAME} edge={EDGE}: mine {len(df)} tags, logged {len(log)}, matched {len(j)}")
    print(f"  corner diff px: median {np.median(dist):.4f}  p90 {np.percentile(dist, 90):.4f}"
          f"  p99 {np.percentile(dist, 99):.4f}  max {dist.max():.3f}  exact(<1e-3) {np.mean(dist < 1e-3):.1%}")
    print(f"  mean signed du,dv: {e[..., 0].mean():+.4f} {e[..., 1].mean():+.4f}")


if __name__ == "__main__":
    main()
