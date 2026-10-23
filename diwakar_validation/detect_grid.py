"""Two refinement layers on top of Diwakar's edge refinement, undistorted-image path (as the tracker).

Base: raw fisheye crop -> remap to the undistorted pinhole box -> OpenCV CONTOUR with the tracker's
settings -> edge_refine.refine_tags_multi (detect_paths.py's "ud_edge").

Layer "grid" (grid_refine.py): one homography per tag fitted to every black/white transition of its
36h11 pattern, the interior cell boundaries as well as the outer edges.

Layer "trk": keeps a tag measured when the detector misses it. On the T1 holds the tag set flickers
(steep tags drop in and out of the CONTOUR detection) and every change moves the jig pose by up to
~1.5 mm, which is the heavy tail of edge refinement's hold jitter. A tag seen in the last MAX_GAP
samples but not detected now is re-found by edge_refine seeded with its previous corners (constant
velocity from the two last sightings); it is accepted when edge_refine finds all four edges and the
cell-centre grey levels match the tag's pattern (grid_refine.pattern_score >= MIN_SCORE). A detected
tag whose edge refinement fails (it would keep the biased CONTOUR corners) is treated as missed.

Sources written (out/corners/<take>_<source>.csv, corners.csv format plus a `how` column, det / trk):
  ud_edge_trk   edge + trk
  ud_grid       edge -> grid
  ud_grid_trk   edge + trk -> grid
Plus out/corners/<take>_grid_info.csv: per tag of ud_grid_trk, grid fit points, interior points,
residual, ok. And, without the re-found tags (the trk sources' `det` rows), the detector's tags minus
those whose edge refinement failed:
  ud_edge_drop  edge, failed tags left out
  ud_grid_drop  edge -> grid, failed tags left out

Usage: python detect_grid.py [take ...]   (default: the 10-03 T0 takes, T1, T2 and T3)
       python detect_grid.py --derive [take ...]   only rewrite the _drop sources from the trk files
edge_refine.py from Diwakar's tracker (BKD_TRACKER, as detect_paths.py).
"""

import json
import os
import sys
from multiprocessing import Pool

import cv2
import grid_refine
import numpy as np
import pandas as pd
from common import DATA, OUT, UV_COLS, take_dir

BKD_TRACKER = os.environ.get("BKD_TRACKER", os.path.expandvars(r"%LOCALAPPDATA%\Temp\bkd_repo\NOARKGames_BKD\tracker"))
sys.path.insert(0, BKD_TRACKER)
from edge_refine import refine_tags_multi

TAKES = (["2026-10-03_T0_test_r1", "2026-10-03_T0_test_r2"]
         + sorted(p.name for p in DATA.glob("2026-10-03_T[123]_*")))
SOURCES = ["ud_edge_trk", "ud_grid", "ud_grid_trk"]
TAGS = {4, 8, 12, 14, 20, 24, 28, 32}
W, H = 1280, 800
CHUNK = 4000          # images.csv rows per worker job; tracking restarts at each chunk
MAX_GAP = 4           # samples; T2/T3 keep images for every 2nd sample only
MIN_SCORE = 0.7
BOX_MARGIN = 3.0      # px; seeds must lie this far inside the undistorted box
(OUT / "corners").mkdir(exist_ok=True)


def _detector(settings):
    p = cv2.aruco.DetectorParameters()
    p.useAruco3Detection = bool(settings.get("aruco3_detection", False))
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    p.perspectiveRemovePixelPerCell = int(settings.get("perspective_pixels_per_cell", 4))
    win = int(settings.get("adaptive_thresh_win_size", 15))
    p.adaptiveThreshWinSizeMin = p.adaptiveThreshWinSizeMax = win
    p.adaptiveThreshWinSizeStep = 1
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


def _ud_box(maps, r, crop):
    """The tracker's undistorted image, cut to the largest box fed only by crop pixels (detect_paths)."""
    m1, m2 = maps
    src = m1.astype(np.int32)
    inside = ((src[..., 0] >= r.x0 + 1) & (src[..., 0] < r.x0 + r.width - 2)
              & (src[..., 1] >= r.y0 + 1) & (src[..., 1] < r.y0 + r.height - 2))
    ys, xs = np.nonzero(inside)
    if not len(xs):
        return None, 0, 0
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    while not inside[y0:y1, x0:x1].all() and x1 - x0 > 20 and y1 - y0 > 20:
        x0, x1, y0, y1 = x0 + 1, x1 - 1, y0 + 1, y1 - 1
    full = np.zeros((H, W), np.uint8)
    full[r.y0:r.y0 + r.height, r.x0:r.x0 + r.width] = crop
    return cv2.remap(full, m1[y0:y1, x0:x1], m2[y0:y1, x0:x1], interpolation=cv2.INTER_LINEAR), x0, y0


def _predict(track, sample):
    """Seed corners (full undistorted image) for `sample` from a tag's last two sightings."""
    (s1, q1), prev = track["last"], track.get("prev")
    if prev is None or s1 - prev[0] > 2 * MAX_GAP:
        return q1
    s0, q0 = prev
    return q1 + (q1 - q0) * (sample - s1) / (s1 - s0)


def run(args):
    name, a, b = args
    d = take_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    maps = []
    for k in (0, 1):
        K = np.array(meta["lens"][f"camera_matrix_cam{k}"])
        D = np.array(meta["lens"][f"dist_coeffs_cam{k}"]).reshape(4, 1)
        P = np.array(meta["undistorted"][f"camera_matrix_cam{k}"])
        maps.append(cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_16SC2))
    det = _detector(meta["settings"])
    im = pd.read_csv(d / "images.csv").iloc[a:b]
    files = {k: open(d / f"images_cam{k}.bin", "rb") for k in (0, 1)}
    tracks = {0: {}, 1: {}}
    rows = {s: [] for s in SOURCES}
    info_rows = []
    for r in im.itertuples(index=False):
        f = files[r.cam]
        f.seek(r.offset)
        crop = np.frombuffer(f.read(r.width * r.height), np.uint8).reshape(r.height, r.width)
        ud, x0, y0 = _ud_box(maps[r.cam], r, crop)
        if ud is None:
            continue
        off = np.array([x0, y0], np.float64)
        bh, bw = ud.shape
        cs, ids, _ = det.detectMarkers(ud)
        edge, failed = {}, set()
        if ids is not None:
            refined = refine_tags_multi([ud], [cs])[0]
            for c, q, i in zip(cs, refined, np.ravel(ids)):
                if int(i) not in TAGS:
                    continue
                q = np.asarray(q, np.float64).reshape(4, 2)
                edge[int(i)] = q
                if np.abs(q - np.asarray(c, np.float64).reshape(4, 2)).max() < 1e-6:
                    failed.add(int(i))

        # trk: detected tags with a working edge refinement, plus tags re-found from their track
        kept = {i: q for i, q in edge.items() if i not in failed}
        how = dict.fromkeys(kept, "det")
        track = tracks[r.cam]
        seeds = {}
        for i, tr in track.items():
            if i in kept or r.sample - tr["last"][0] > MAX_GAP:
                continue
            q = _predict(tr, r.sample) - off
            if (q.min() >= BOX_MARGIN and q[:, 0].max() <= bw - 1 - BOX_MARGIN
                    and q[:, 1].max() <= bh - 1 - BOX_MARGIN):
                seeds[i] = q
        if seeds:
            ids_s = list(seeds)
            seed_cs = [seeds[i].reshape(1, 4, 2).astype(np.float32) for i in ids_s]
            for i, s0, q in zip(ids_s, seed_cs, refine_tags_multi([ud], [seed_cs])[0]):
                q = np.asarray(q, np.float64).reshape(4, 2)
                if np.abs(q - s0.reshape(4, 2)).max() < 1e-6:
                    continue  # edge_refine did not find the tag's edges
                if grid_refine.pattern_score(ud, q, i) < MIN_SCORE:
                    continue
                kept[i], how[i] = q, "trk"
        for i, q in kept.items():
            prev = track.get(i)
            track[i] = {"last": (r.sample, q + off), "prev": prev["last"] if prev else None}

        # grid layer on edge corners (ud_grid) and on the trk set (ud_grid_trk); shared tags fitted once
        grid = {}
        for source, tags, refine, label in (("ud_edge_trk", kept, False, how), ("ud_grid", edge, True, {}),
                                            ("ud_grid_trk", kept, True, how)):
            for i, q in sorted(tags.items()):
                if refine:
                    key = (i, q.tobytes())
                    if key not in grid:
                        grid[key] = grid_refine.refine_tag(ud, q, i, fit_delta=False)
                    q, inf = grid[key]
                    if source == "ud_grid_trk":
                        info_rows.append((r.sample, r.cam, i, inf["ok"], inf["points"], inf["interior"], inf["rms_px"]))
                rows[source].append((r.sample, r.cam, i, *(q + off).ravel(), label.get(i, "det")))
    cols = ["sample", "cam", "marker_id", *UV_COLS, "how"]
    return (name, {s: pd.DataFrame(v, columns=cols) for s, v in rows.items()},
            pd.DataFrame(info_rows, columns=["sample", "cam", "marker_id", "ok", "points", "interior", "rms_px"]))


def derive_drop(take):
    for src in ("ud_edge", "ud_grid"):
        df = pd.read_csv(OUT / "corners" / f"{take}_{src}_trk.csv")
        df[df["how"] == "det"].to_csv(OUT / "corners" / f"{take}_{src}_drop.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    takes = [a for a in sys.argv[1:] if not a.startswith("--")] or TAKES
    if "--derive" in sys.argv:
        for t in takes:
            derive_drop(t)
        sys.exit()
    jobs = []
    for t in takes:
        n = sum(1 for _ in open(take_dir(t) / "images.csv")) - 1
        jobs += [(t, a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
    parts = {t: {} for t in takes}
    infos = {t: [] for t in takes}
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for i, (name, res, info) in enumerate(pool.imap_unordered(run, jobs), 1):
            for k, df in res.items():
                parts[name].setdefault(k, []).append(df)
            infos[name].append(info)
            print(f"{i}/{len(jobs)} {name}", flush=True)
    for t in takes:
        for s, dfs in parts[t].items():
            df = pd.concat(dfs).sort_values(["sample", "cam", "marker_id"], kind="stable")
            df.to_csv(OUT / "corners" / f"{t}_{s}.csv", index=False, float_format="%.4f")
            print(t, s, len(df), "trk", int((df["how"] == "trk").sum()), flush=True)
        info = pd.concat(infos[t]).sort_values(["sample", "cam", "marker_id"], kind="stable")
        info.to_csv(OUT / "corners" / f"{t}_grid_info.csv", index=False, float_format="%.4f")
        derive_drop(t)
