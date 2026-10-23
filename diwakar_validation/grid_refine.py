"""Grid refinement of AprilTag 36h11 corners: one homography fitted to every black/white transition of
the tag's known bit pattern, the interior cell boundaries as well as the four outer edges.

Why: edge refinement (edge_refine.py) places the four outer edges from ~128 grey-level crossings. The
tag's own pattern holds two to three times more straight transitions, all on the 9 + 9 lines u = k,
v = k (k = 0..8 cells) of its grid, and a homography ties them together: parallel lines share a
vanishing point and stay evenly spaced in cross-ratio. Fitting the homography to all of them uses
every crossing for every corner.

It also separates print bleed from tag size. A halfway crossing sits off the true black/white
boundary by delta cells towards the white side (ink spread, blur through a non-linear response). On
the outer edges that offset is indistinguishable from a bigger tag. Interior transitions come in both
polarities on the same lines, so delta is identifiable: it is fitted per tag and the corners are those
of the bleed-free 8 x 8 square.

Per tag, starting from refined corners (edge_refine's):
  1. runs: maximal straight pieces of each grid line along which the colours on both sides stay the
     same (no junction inside); PER_CELL profiles per cell of run, END_MARGIN cells kept free at each
     run end (corners and junctions are not 1-D edges);
  2. profiles of NO bilinear grey levels along the image normal, black side to white side, half a
     cell across for interior runs (to the centres of the two cells it separates) and edge_refine's
     clip(0.6 cell, 2.5, 4) px for outer runs; interior runs only where that cell is >= MIN_CELL_PX;
  3. black / white levels per run from the profile ends, halfway crossing nearest the middle;
  4. Gauss-Newton on the image -> tag homography (8) and delta (1), residual = image distance (px)
     from each crossing to its line, Huber weights; profiles re-sampled once around the first fit.
A tag keeps its input corners when too few crossings survive, the fit residual is large, delta is
implausible, or a corner would move more than max(MAX_SHIFT_PX, MAX_SHIFT_CELLS cell).
"""

from functools import cache

import cv2
import numpy as np

CELLS = 8              # black border + 6 x 6 data bits, the square the detector's corners bound
PER_CELL = 5           # profiles per cell of run length
NO = 13                # samples per profile
END_MARGIN = 0.35      # cells kept free at each run end
MIN_CELL_PX = 4.0      # interior runs need cells at least this wide across them
OUTER_R = (2.5, 4.0)   # px; outer-run profile half-width range (edge_refine's R_MIN, R_FINE)
MIN_CONTRAST = 20.0    # grey levels between a run's black and white ends
MIN_POINTS = 40
ROUNDS = 2             # profile sampling rounds
ITERS = 6              # Gauss-Newton steps per round
HUBER_PX = 0.25
MAX_RMS_PX = 0.6
MAX_DELTA = 0.25       # cells
MAX_SHIFT_PX, MAX_SHIFT_CELLS = 1.5, 0.3
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_SQUARE = np.array([[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]], np.float64)


@cache
def tag_pattern(marker_id):
    """(CELLS + 2) x (CELLS + 2) colours, 1 = white, with the white quiet zone around the black border;
    cell (r, c) of the 8 x 8 tag is [r + 1, c + 1] and spans u in [c, c + 1], v in [r, r + 1]."""
    img = cv2.aruco.generateImageMarker(_DICT, int(marker_id), CELLS, borderBits=1)
    grid = np.ones((CELLS + 2, CELLS + 2), np.int8)
    grid[1:-1, 1:-1] = img > 127
    return grid


@cache
def tag_runs(marker_id):
    """Straight transition runs of a tag: rows (axis, k, s, t0, t1, outer). axis 0 = line u = k
    (vertical), 1 = line v = k; s = +1 if the white side is towards +u (+v); t0..t1 = extent along the
    line, cells; outer = on the tag's outer edge."""
    g = tag_pattern(marker_id)
    runs = []
    for axis in (0, 1):
        for k in range(CELLS + 1):
            cur = None
            for t in range(CELLS):
                if axis == 0:   # cells left (col k - 1) and right (col k) of u = k, row t
                    a, b = g[t + 1, k], g[t + 1, k + 1]
                else:           # cells above (row k - 1) and below (row k) of v = k, column t
                    a, b = g[k, t + 1], g[k + 1, t + 1]
                s = 0 if a == b else (1 if b else -1)
                if cur is not None and s == cur[2]:
                    cur[4] = t + 1
                    continue
                if cur is not None and cur[2]:
                    runs.append(cur)
                cur = [axis, k, s, t, t + 1, k in (0, CELLS)]
            if cur is not None and cur[2]:
                runs.append(cur)
    return np.array(runs, np.float64)


def _jac_h(H, p):
    """Image point and 2 x 2 Jacobian d(image)/d(tag) of homography H at tag points p (N, 2)."""
    x = np.c_[p, np.ones(len(p))] @ H.T
    w = x[:, 2:3]
    img = x[:, :2] / w
    J = (H[None, :2, :2] - img[:, :, None] * H[None, 2:3, :2]) / w[:, :, None]
    return img, J


def _profiles(img, H, delta, runs, outer_only):
    """Sample every run of one tag under homography H (tag -> image). Returns base points, unit normals
    (black -> white), offsets (N, NO) px, run index per profile, line (axis, k, s) per profile, and
    the per-profile cell size across the line (px)."""
    pts, rid = [], []
    for i, (axis, k, s, t0, t1, outer) in enumerate(runs):
        if outer_only and not outer:
            continue
        a, b = t0 + END_MARGIN, t1 - END_MARGIN
        n = max(2, round(PER_CELL * (b - a)))
        t = np.linspace(a, b, n)
        c = np.full(n, k + s * delta)
        pts.append(np.c_[c, t] if axis == 0 else np.c_[t, c])
        rid.append(np.full(n, i))
    p = np.concatenate(pts)
    rid = np.concatenate(rid)
    axis, k, s, outer = runs[rid, 0], runs[rid, 1], runs[rid, 2], runs[rid, 5]
    base, J = _jac_h(H, p)
    across = np.where(axis[:, None] == 0, J[:, :, 0], J[:, :, 1])     # image of one cell across the line
    along = np.where(axis[:, None] == 0, J[:, :, 1], J[:, :, 0])
    nrm = np.c_[-along[:, 1], along[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm *= np.sign((nrm * across).sum(1) * s)[:, None]                 # towards the white side
    cell = np.abs((nrm * across).sum(1))
    r = np.where(outer > 0, np.clip(0.6 * cell, *OUTER_R), 0.5 * cell)
    offs = r[:, None] * np.linspace(-1.0, 1.0, NO)[None, :]
    keep = (outer > 0) | (cell >= MIN_CELL_PX)
    h, w = img.shape[:2]
    xy = base[:, None, :] + offs[..., None] * nrm[:, None, :]
    keep &= ((xy[..., 0] >= 0) & (xy[..., 1] >= 0) & (xy[..., 0] <= w - 1) & (xy[..., 1] <= h - 1)).all(1)
    return base[keep], nrm[keep], offs[keep], rid[keep], np.c_[axis, k, s][keep], xy[keep]


def _crossings(img, xy, offs, rid, nrun):
    """Halfway crossing per profile (px along the normal), with run-level black / white from the ends."""
    x0, y0 = np.floor(xy.reshape(-1, 2).min(0)).astype(int)
    x1, y1 = np.ceil(xy.reshape(-1, 2).max(0)).astype(int) + 2
    sub = img[y0:y1, x0:x1].astype(np.float32)
    prof = cv2.remap(sub, (xy[..., 0] - x0).astype(np.float32), (xy[..., 1] - y0).astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cnt = np.bincount(rid, minlength=nrun).astype(np.float64)
    black = np.bincount(rid, prof[:, :2].mean(1), minlength=nrun) / np.maximum(cnt, 1)
    white = np.bincount(rid, prof[:, -2:].mean(1), minlength=nrun) / np.maximum(cnt, 1)
    sgn = prof - ((black + white) / 2)[rid, None]
    up = (sgn[:, :-1] < 0) & (sgn[:, 1:] >= 0)
    rank = np.abs(np.arange(NO - 1) - (NO - 2) / 2)
    j = np.where(up, rank, 1e9).argmin(1)
    rows = np.arange(len(j))
    ok = up[rows, j] & ((white - black)[rid] >= MIN_CONTRAST)
    s0, s1 = sgn[rows, j], sgn[rows, j + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = offs[rows, j] + s0 / (s0 - s1) * (offs[:, 1] - offs[:, 0])
    return x, ok & np.isfinite(x)


def _fit(P, line, G, delta, fit_delta, huber):
    """Gauss-Newton on G (image -> tag, normalised image coordinates) and delta. P (N, 3) homogeneous
    normalised image points; line (N, 3) = (axis, k, s); huber in normalised units. Returns G, delta, residuals (normalised units)."""
    axis, k, s = line.T
    a = np.c_[axis == 0, axis == 1].astype(np.float64)
    for _ in range(ITERS):
        L = np.c_[a, -(k + s * delta)]
        q = P @ G.T
        num = (L * q).sum(1)
        b = L @ G                                                     # G^T L, per point
        nn = np.hypot(b[:, 0], b[:, 1])
        d = num / nn
        bpad = np.c_[b[:, :2], np.zeros(len(b))]
        JG = L[:, :, None] * (P[:, None, :] / nn[:, None, None] - (num / nn**3)[:, None, None] * bpad[:, None, :])
        J = JG.reshape(len(P), 9)[:, :8]
        if fit_delta:
            dnum = -s * q[:, 2]
            dnn = (b[:, 0] * -s * G[2, 0] + b[:, 1] * -s * G[2, 1]) / nn
            J = np.c_[J, dnum / nn - num * dnn / nn**2]
        w = np.minimum(1.0, huber / np.maximum(np.abs(d), 1e-12))
        A = J.T @ (J * w[:, None])
        step = np.linalg.solve(A + 1e-9 * np.trace(A) * np.eye(len(A)), -J.T @ (w * d))
        G = G + np.r_[step[:8], 0.0].reshape(3, 3)
        if fit_delta:
            delta = delta + step[8]
        if np.abs(step).max() < 1e-10:
            break
    L = np.c_[a, -(k + s * delta)]
    b = L @ G
    return G, delta, (L * (P @ G.T)).sum(1) / np.hypot(b[:, 0], b[:, 1])



def refine_tag(img, corners, marker_id, fit_delta=True, outer_only=False):
    """Grid-refined corners (4, 2) of one tag in `img`, from refined `corners` (4, 2), and an info dict
    (ok, points, interior, delta, rms_px). On failure the input corners are returned with ok = False."""
    q = np.asarray(corners, np.float64).reshape(4, 2)
    runs = tag_runs(marker_id)
    info = {"ok": False, "points": 0, "interior": 0, "delta": np.nan, "rms_px": np.nan}
    side = np.linalg.norm(q - np.roll(q, -1, 0), axis=1)
    cell = side.mean() / CELLS
    centre = q.mean(0)
    T = np.array([[1 / cell, 0, -centre[0] / cell], [0, 1 / cell, -centre[1] / cell], [0, 0, 1.0]])
    H = cv2.getPerspectiveTransform(_SQUARE.astype(np.float32), q.astype(np.float32)).astype(np.float64)
    G = np.linalg.inv(T @ H)
    G /= G[2, 2]
    delta = 0.0
    for _ in range(ROUNDS):
        Hc = np.linalg.inv(T) @ np.linalg.inv(G)
        try:
            base, nrm, offs, rid, line, xy = _profiles(img, Hc, delta, runs, outer_only)
        except ValueError:
            return q, info
        if len(base) < MIN_POINTS:
            return q, info
        x, ok = _crossings(img, xy, offs, rid, len(runs))
        if ok.sum() < MIN_POINTS:
            return q, info
        P = base[ok] + x[ok, None] * nrm[ok]
        Pn = np.c_[(P - centre) / cell, np.ones(len(P))]
        G, delta, res = _fit(Pn, line[ok], G, delta, fit_delta, HUBER_PX / cell)
        G /= G[2, 2]
    H = np.linalg.inv(T) @ np.linalg.inv(G)
    x = np.c_[_SQUARE, np.ones(4)] @ H.T
    new = x[:, :2] / x[:, 2:3]
    res_px = res * cell
    w = np.minimum(1.0, HUBER_PX / np.maximum(np.abs(res_px), 1e-12))
    rms = float(np.sqrt((w * res_px**2).sum() / w.sum()))
    shift = np.linalg.norm(new - q, axis=1).max()
    info.update(points=int(ok.sum()), interior=int((runs[rid[ok], 5] == 0).sum()), delta=float(delta), rms_px=rms)
    good = (np.isfinite(new).all() and rms <= MAX_RMS_PX and abs(delta) <= MAX_DELTA
            and shift <= max(MAX_SHIFT_PX, MAX_SHIFT_CELLS * cell))
    if not good:
        return q, info
    info["ok"] = True
    return new, info


def refine_tags(img, corners, ids, **kw):
    """refine_tag for every detectMarkers tag. Returns (tuple of (1, 4, 2) float32, list of info)."""
    out, infos = [], []
    for c, i in zip(corners, np.ravel(ids)):
        new, info = refine_tag(img, c, int(i), **kw)
        out.append(new.reshape(1, 4, 2).astype(np.float32))
        infos.append(info)
    return tuple(out), infos


def pattern_score(img, corners, marker_id):
    """Correlation between the grey levels at the 8 x 8 cell centres (through the corners' homography)
    and the tag's pattern: about 1 for the right tag in the right orientation, about 0 otherwise."""
    q = np.asarray(corners, np.float32).reshape(4, 2)
    H = cv2.getPerspectiveTransform(_SQUARE.astype(np.float32), q)
    c = np.arange(CELLS) + 0.5
    uv = np.stack(np.meshgrid(c, c), -1).reshape(-1, 1, 2).astype(np.float32)
    xy = cv2.perspectiveTransform(uv, H).reshape(-1, 2)
    h, w = img.shape[:2]
    if (xy < 0).any() or (xy[:, 0] > w - 1).any() or (xy[:, 1] > h - 1).any():
        return -1.0
    v = cv2.remap(img.astype(np.float32), xy[:, 0:1], xy[:, 1:2], interpolation=cv2.INTER_LINEAR).ravel()
    p = tag_pattern(marker_id)[1:-1, 1:-1].ravel().astype(np.float64)
    if v.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(v, p)[0, 1])
