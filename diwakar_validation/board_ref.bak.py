"""
Rigid-body board model for the multi-marker NOARK device.

The markers are glued to one rigid object, so the whole constellation has a
single pose. Once each marker's fixed pose *on the device* is known (from
calibrate_board.py), all detected corners — from however many markers happen
to be visible — feed one joint solvePnP. Depth is then constrained by the
spread between markers instead of each marker's own apparent size, which
cuts Z jitter dramatically and removes the IPPE tilt-ambiguity flips.

Board frame = the reference marker's frame (identity pose for the reference).
Geometry is stored in board_geometry.json next to this file.
"""

import json

import cv2
import numpy as np
from scipy.optimize import least_squares

MARKER_LENGTH = 0.05

# Offsets from each marker's center to the handle grip point, expressed
# in the marker's own frame (+X = printed-right, +Y = printed-up, +Z = out
# of face). Derived from the CAD model — markers are glued with +Y
# (printed-up) aligned to device-up.
MARKER_OFFSETS = {
    4:  np.array([-0.002, -0.016, -0.056]),   # front-right    (angled 60° from +X, into -Y)
    8:  np.array([ 0.002, -0.016, -0.056]),   # front-left     (angled 60° from +X, into +Y)
    12: np.array([ 0.001, -0.016, -0.059]),   # front
    14: np.array([ 0.000, -0.066, -0.065]),   # front-top      (angled 10° from +X toward +Z)
    20: np.array([ 0.125, -0.017, -0.054]),   # back-right     (-Y side)
    24: np.array([-0.125, -0.017, -0.054]),   # back-left      (+Y side, mirror of 20)
    28: np.array([-0.161, -0.019,  0.104]),   # back-far-left  wing (empirical, via derive_offsets.py)
    32: np.array([ 0.157, -0.017,  0.115]),   # back-far-right wing (empirical, via derive_offsets.py)
}


def marker_object_points(length: float = MARKER_LENGTH, aspect: float = 1.0) -> np.ndarray:
    """Corner coordinates in the marker's own frame, in ArUco detection
    order (TL, TR, BR, BL). Width `length` along printed-right (X), height
    `length * aspect` along printed-up (Y): printed stickers are not exactly
    square (the printer scales ~1 % differently along and across the paper).
    aspect = 1 gives the square cv2.SOLVEPNP_IPPE_SQUARE requires."""
    hx, hy = length / 2, length * aspect / 2
    return np.array(
        [[-hx,  hy, 0],
         [ hx,  hy, 0],
         [ hx, -hy, 0],
         [-hx, -hy, 0]],
        dtype=np.float64,   # float32 would round away the tiny aspect steps calibrate_rig differentiates by
    )


class BoardGeometry:
    """Fixed pose of every marker in the board frame, plus the grip point.

    marker_poses maps id -> (R, t) where R (3,3) and t (3,) take a point
    from the marker's frame into the board frame: p_board = R @ p_marker + t.
    marker_aspect = tag height / width (fitted by calibrate_rig.py; 1 = square).
    """

    def __init__(self, marker_length: float, reference_id: int,
                 marker_poses: dict, grip_point: np.ndarray, marker_aspect: float = 1.0):
        self.marker_length = float(marker_length)
        self.marker_aspect = float(marker_aspect)
        self.reference_id = int(reference_id)
        self.marker_poses = marker_poses
        self.grip_point = np.asarray(grip_point, dtype=np.float64).flatten()

    def corners_in_board(self, marker_id: int) -> np.ndarray:
        R, t = self.marker_poses[marker_id]
        local = marker_object_points(self.marker_length, self.marker_aspect).astype(np.float64)
        return (local @ R.T) + t

    @classmethod
    def load(cls, path: str) -> "BoardGeometry":
        with open(path) as f:
            data = json.load(f)
        poses = {
            int(mid): (np.array(m["rotation"], dtype=np.float64).reshape(3, 3),
                       np.array(m["translation"], dtype=np.float64).flatten())
            for mid, m in data["markers"].items()
        }
        return cls(data["marker_length"], data["reference_id"],
                   poses, np.array(data["grip_point"]), data.get("marker_aspect", 1.0))

    def save(self, path: str, meta: dict = None) -> None:
        data = {
            "marker_length": self.marker_length,
            "marker_aspect": self.marker_aspect,
            "reference_id": self.reference_id,
            "grip_point": self.grip_point.tolist(),
            "markers": {
                str(mid): {"rotation": R.tolist(), "translation": t.tolist()}
                for mid, (R, t) in self.marker_poses.items()
            },
        }
        if meta:
            data["meta"] = meta
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def reprojection_error(obj_pts, img_pts, rvec, tvec, camera_matrix) -> float:
    """Mean pixel distance between detected corners and the pose's projection."""
    proj, _ = cv2.projectPoints(
        np.asarray(obj_pts, dtype=np.float64), rvec, tvec, camera_matrix, np.zeros(5)
    )
    return float(np.linalg.norm(proj.reshape(-1, 2) - np.asarray(img_pts).reshape(-1, 2), axis=1).mean())


def _board_points(board: BoardGeometry, corners, ids):
    """(object points in the board frame, image points) for the markers this
    camera saw and the geometry knows about. (None, None) if there are none."""
    if ids is None:
        return None, None
    obj_pts, img_pts = [], []
    for corner, _id in zip(corners, np.asarray(ids).flatten()):
        _id = int(_id)
        if _id not in board.marker_poses:
            continue
        obj_pts.append(board.corners_in_board(_id))
        img_pts.append(np.asarray(corner).reshape(4, 2))
    if not obj_pts:
        return None, None
    return (np.concatenate(obj_pts).astype(np.float64),
            np.concatenate(img_pts).astype(np.float64))


# A joint fit is thrown away when it disagrees with the images by more than
# this, or has moved this far from the pose it started at. With weak geometry
# (one small marker at the edge of the workspace) an unbounded fit can wander
# — offline it once returned a pose 210 mm out (2026-09-21).
DUAL_MAX_REPROJ_PX = 3.0
DUAL_MAX_JUMP_M = 0.02


def estimate_board_pose_dual(board: BoardGeometry, corners0, ids0, corners1, ids1,
                             K0, K1, Rx, tx, guess):
    """One board pose in cam0's frame, fitted to ALL corners from BOTH cameras.

    Both cameras see one rigid object, so one pose predicts both images: a
    board point b is at p0 = R b + t in cam0 and at p1 = Rx^T (p0 - tx) in
    cam1. Fitting once uses every corner, instead of solving each camera alone
    and averaging the two answers — worth about a third of the jitter while
    still (0.72 -> 0.47 mm median over one T1 grid, 2026-09-21).

    guess: (rvec, tvec) to start from — the averaged pose, which is also what
    is returned when the fit is rejected.

    Returns (rvec (3,), tvec (3,), mean_reproj_px, accepted) or None when
    neither camera has a usable marker.
    """
    obj0, img0 = _board_points(board, corners0, ids0)
    obj1, img1 = _board_points(board, corners1, ids1)
    if obj0 is None and obj1 is None:
        return None
    zero_dist = np.zeros(5)
    RxT = np.asarray(Rx, dtype=np.float64).T
    tx = np.asarray(tx, dtype=np.float64).flatten()

    def errors(x):
        rvec, tvec = x[:3], x[3:]
        out = []
        if obj0 is not None:
            proj = cv2.projectPoints(obj0, rvec, tvec, K0, zero_dist)[0].reshape(-1, 2)
            out.append(proj - img0)
        if obj1 is not None:
            r1 = cv2.Rodrigues(RxT @ cv2.Rodrigues(rvec)[0])[0].flatten()
            t1 = RxT @ (tvec - tx)
            proj = cv2.projectPoints(obj1, r1, t1, K1, zero_dist)[0].reshape(-1, 2)
            out.append(proj - img1)
        return np.concatenate(out)

    x0 = np.concatenate([np.asarray(guess[0], dtype=np.float64).reshape(3),
                         np.asarray(guess[1], dtype=np.float64).reshape(3)])
    try:
        res = least_squares(lambda x: errors(x).ravel(), x0, method="lm", xtol=1e-9)
        x = res.x
    except Exception:
        return guess[0], guess[1], float("nan"), False

    err = float(np.linalg.norm(errors(x), axis=1).mean())
    jump = float(np.linalg.norm(x[3:] - x0[3:]))
    if (not np.all(np.isfinite(x)) or err > DUAL_MAX_REPROJ_PX
            or jump > DUAL_MAX_JUMP_M):
        return guess[0], guess[1], err, False
    return x[:3], x[3:], err, True


def _project_fisheye(obj: np.ndarray, rvec, tvec, K, D) -> np.ndarray:
    """Board points -> raw fisheye pixels (N, 2)."""
    pix, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), np.asarray(rvec, np.float64).reshape(3, 1),
                                       np.asarray(tvec, np.float64).reshape(3, 1), K, D)
    return pix.reshape(-1, 2)


def _initial_pose_raw(board: BoardGeometry, corners, ids, K, D):
    """A starting pose from one camera's RAW corners: straighten just those
    corners to normalised coordinates (identity intrinsics), then the ordinary
    board solve. Returns (rvec, tvec) in that camera's frame, or None."""
    obj, img = _board_points(board, corners, ids)
    if obj is None:
        return None
    norm = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D).reshape(-1, 4, 2)
    result = estimate_board_pose(board, [q.reshape(1, 4, 2) for q in norm],
                                 [i for i in np.asarray(ids).flatten()
                                  if int(i) in board.marker_poses],
                                 np.eye(3), None)
    if result is None:
        return None
    return result[0], result[1]


def estimate_board_pose_raw(board: BoardGeometry, corners0, ids0, corners1, ids1,
                            lens0, lens1, Rx, tx, guess=None):
    """One board pose in cam0's frame from BOTH cameras' corners, found on the
    RAW (distorted) images — no image straightening anywhere.

    Each corner is predicted through its own camera's fisheye model and
    compared in raw pixels, where the sensor's noise is the same everywhere; in
    a straightened image the sides are stretched (1/cos^2 of the angle), so the
    same noise would count up to several times more there.

    A board point b sits at p0 = R b + t in cam0 and p1 = Rx^T (p0 - tx) in cam1
    (Rx, tx: cam1 -> cam0, as everywhere in the tracker). Either camera may
    have no markers; the solve then uses the other alone.

    lens0/lens1: (K, D) fisheye intrinsics. guess: (rvec, tvec) to start from —
    the previous frame's pose; without one, a pose from the camera seeing more
    markers.

    Returns (rvec (3,), tvec (3,), mean reprojection px cam0, cam1) with nan
    for a camera that saw nothing, or None when neither camera has a marker or
    no start could be found.
    """
    obj0, img0 = _board_points(board, corners0, ids0)
    obj1, img1 = _board_points(board, corners1, ids1)
    if obj0 is None and obj1 is None:
        return None
    RxT = np.asarray(Rx, dtype=np.float64).T
    tx = np.asarray(tx, dtype=np.float64).flatten()

    if guess is None:
        n0 = 0 if obj0 is None else len(obj0)
        n1 = 0 if obj1 is None else len(obj1)
        if n0 >= n1:
            guess = _initial_pose_raw(board, corners0, ids0, *lens0)
        else:
            start = _initial_pose_raw(board, corners1, ids1, *lens1)
            if start is not None:                       # cam1 frame -> cam0 frame
                R1 = cv2.Rodrigues(np.asarray(start[0], np.float64).reshape(3, 1))[0]
                guess = (cv2.Rodrigues(Rx @ R1)[0].flatten(),
                         Rx @ np.asarray(start[1]).flatten() + tx)
        if guess is None:
            return None

    def residuals(x):
        rvec, tvec = x[:3], x[3:]
        out = []
        if obj0 is not None:
            out.append(_project_fisheye(obj0, rvec, tvec, *lens0) - img0)
        if obj1 is not None:
            r1 = cv2.Rodrigues(RxT @ cv2.Rodrigues(rvec)[0])[0].flatten()
            t1 = RxT @ (tvec - tx)
            out.append(_project_fisheye(obj1, r1, t1, *lens1) - img1)
        return np.concatenate(out)

    x0 = np.concatenate([np.asarray(guess[0], dtype=np.float64).reshape(3),
                         np.asarray(guess[1], dtype=np.float64).reshape(3)])
    try:
        x = least_squares(lambda v: residuals(v).ravel(), x0, method="lm", xtol=1e-9).x
    except Exception:
        return None
    if not np.all(np.isfinite(x)):
        return None
    err = np.linalg.norm(residuals(x), axis=1)
    n0 = 0 if obj0 is None else len(obj0)
    e0 = float(err[:n0].mean()) if n0 else float("nan")
    e1 = float(err[n0:].mean()) if obj1 is not None else float("nan")
    return x[:3], x[3:], e0, e1


def _huber(r: np.ndarray, delta: float) -> tuple:
    """(cost, per-corner weight) of residuals r (n, 2) under a Huber loss on
    each corner's pixel distance e. delta <= 0: plain least squares.

        cost(e) = e^2 / 2              e <= delta
                  delta (e - delta/2)  e >  delta
        weight  = 1                    e <= delta
                  delta / e            e >  delta

    A corner far off then pulls with a weight that falls as 1/e instead of
    counting e^2: one tag 10 px out no longer outweighs a hundred good corners."""
    e = np.linalg.norm(r, axis=1)
    if delta <= 0:
        return float(0.5 * (e ** 2).sum()), np.ones(len(e))
    far = e > delta
    cost = np.where(far, delta * (e - 0.5 * delta), 0.5 * e ** 2)
    w = np.ones(len(e))
    w[far] = delta / e[far]
    return float(cost.sum()), w


def estimate_board_pose_dual_gn(board: BoardGeometry, corners0, ids0, corners1, ids1,
                                K0, K1, Rx, tx, guess, max_iter=10, return_iters=False,
                                robust_px: float = 0.0, min_step: float = 1e-8):
    """The same fit as estimate_board_pose_dual — one board pose in cam0's frame
    from all corners of both cameras, straightened pixels — done the standard
    way for a multi-camera rig: Levenberg-Marquardt with the derivatives
    written out, not estimated by trial.

    A small pose change delta = (w, s) moves a board point p0 = R b + t (cam0
    frame) by  w x p0 + s,  so

        dp0/d delta = [ -[p0]x | I ]           p1 = Rx^T (p0 - tx):  dp1/d delta = Rx^T [ -[p0]x | I ]
        du/dp       = [ fx/Z  0     -fx X/Z^2 ]
                      [ 0     fy/Z  -fy Y/Z^2 ]
        J = du/dp . dp/d delta   (2 x 6 per corner)

    and each step solves (J^T J + lambda diag(J^T J)) delta = J^T r, r = seen - predicted.

    robust_px > 0: Huber loss with that threshold (see _huber) — every step is
    weighted by each corner's current weight, (J^T W J + ...) delta = J^T W r, so
    a noisy or half-hidden tag cannot drag the pose. The frame is then judged
    by the MEDIAN corner error (the mean stays high by design when one tag is
    off, and would reject a pose the loss handled well). robust_px = 0: plain
    least squares, judged by the mean, exactly as before.

    Stops when a kept step is shorter than min_step (rad and m together; 1e-8
    = 10 nm), when lambda passes 1e6, or after max_iter steps.

    Accepted on DUAL_MAX_REPROJ_PX and DUAL_MAX_JUMP_M. Returns (rvec, tvec,
    reprojection px (mean, or median when robust), accepted) [+ (iterations,
    why it stopped) with return_iters: "small step", "lambda", "singular" or
    "cap" = ran all max_iter], or None when no camera has a marker.
    """
    obj0, img0 = _board_points(board, corners0, ids0)
    obj1, img1 = _board_points(board, corners1, ids1)
    if obj0 is None and obj1 is None:
        return None
    Rx = np.asarray(Rx, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64).flatten()
    cams = []                                   # (board pts, pixels, K, A, c): p = A (p0 - c)
    if obj0 is not None:
        cams.append((obj0, img0, np.asarray(K0, np.float64), np.eye(3), np.zeros(3)))
    if obj1 is not None:
        cams.append((obj1, img1, np.asarray(K1, np.float64), Rx.T, tx))

    def residuals_and_jacobian(R, t, want_j):
        rs, js = [], []
        for obj, img, K, A, c in cams:
            p0 = obj @ R.T + t
            p = (p0 - c) @ A.T
            X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
            if np.any(Z <= 1e-6):
                return None, None
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            rs.append(img - np.stack([fx * X / Z + cx, fy * Y / Z + cy], axis=1))
            if want_j:
                n = len(p0)
                dudp = np.zeros((n, 2, 3))
                dudp[:, 0, 0] = fx / Z
                dudp[:, 0, 2] = -fx * X / Z ** 2
                dudp[:, 1, 1] = fy / Z
                dudp[:, 1, 2] = -fy * Y / Z ** 2
                D = np.zeros((n, 3, 6))             # [ -[p0]x | I ]
                D[:, 0, 1], D[:, 0, 2] = p0[:, 2], -p0[:, 1]
                D[:, 1, 0], D[:, 1, 2] = -p0[:, 2], p0[:, 0]
                D[:, 2, 0], D[:, 2, 1] = p0[:, 1], -p0[:, 0]
                D[:, :, 3:] = np.eye(3)
                js.append(np.einsum("nij,jk,nkl->nil", dudp, A, D).reshape(-1, 6))
        r = np.concatenate(rs)
        return r, (np.concatenate(js) if want_j else None)

    R = cv2.Rodrigues(np.asarray(guess[0], np.float64).reshape(3, 1))[0]
    t = np.asarray(guess[1], np.float64).flatten().copy()
    t_start = t.copy()
    r, J = residuals_and_jacobian(R, t, True)
    if r is None:                               # start pose puts a point behind a camera
        return None
    cost, w = _huber(r, robust_px)
    lam = 1e-3
    iters = 0
    stop = "cap"
    for iters in range(1, max_iter + 1):
        wr = np.repeat(w, 2)                    # rows of J and r come as (u, v) per corner
        H = J.T @ (J * wr[:, None])
        g = J.T @ (wr * r.ravel())
        try:
            delta = np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
        except np.linalg.LinAlgError:
            stop = "singular"
            break
        dR = cv2.Rodrigues(delta[:3].reshape(3, 1))[0]
        R_new, t_new = dR @ R, dR @ t + delta[3:]
        r_new, J_new = residuals_and_jacobian(R_new, t_new, True)
        if r_new is not None:
            cost_new, w_new = _huber(r_new, robust_px)
        if r_new is not None and cost_new < cost:
            R, t, r, J, cost, w = R_new, t_new, r_new, J_new, cost_new, w_new
            lam = max(lam / 10.0, 1e-7)
            if np.linalg.norm(delta) < min_step:
                stop = "small step"
                break
        else:
            lam *= 10.0
            if lam > 1e6:
                stop = "lambda"
                break

    rvec = cv2.Rodrigues(R)[0].flatten()
    e = np.linalg.norm(r, axis=1)
    err = float(np.median(e) if robust_px > 0 else e.mean())
    jump = float(np.linalg.norm(t - t_start))
    accepted = bool(np.all(np.isfinite(t)) and err <= DUAL_MAX_REPROJ_PX
                    and jump <= DUAL_MAX_JUMP_M)
    out = (rvec, t, err, accepted)
    return out + (iters, stop) if return_iters else out


def estimate_board_pose(board: BoardGeometry, corners, ids, camera_matrix,
                        guess=None):
    """Single rigid-body pose from all visible known markers.

    Returns (rvec (3,), tvec (3,), mean_reproj_px) or None if no known
    marker is visible / the solve fails.

    Solver choice:
      - guess available  -> ITERATIVE refined from the previous frame's pose.
        Works for any marker count and naturally rejects the mirrored
        tilt solution of a lone planar marker.
      - >= 2 markers, no guess -> SQPNP (robust for arbitrary 3D point sets).
      - 1 marker, no guess -> IPPE_SQUARE on that marker, composed with its
        stored board pose.
    """
    ids_flat = np.asarray(ids).flatten()
    obj_pts, img_pts, used_ids = [], [], []
    for corner, _id in zip(corners, ids_flat):
        _id = int(_id)
        if _id not in board.marker_poses:
            continue
        obj_pts.append(board.corners_in_board(_id))
        img_pts.append(np.asarray(corner).reshape(4, 2))
        used_ids.append(_id)
    if not used_ids:
        return None

    obj = np.concatenate(obj_pts).astype(np.float64)
    img = np.concatenate(img_pts).astype(np.float64)
    zero_dist = np.zeros(5)  # frames are undistorted upstream

    if guess is not None:
        rvec0 = np.asarray(guess[0], dtype=np.float64).reshape(3, 1).copy()
        tvec0 = np.asarray(guess[1], dtype=np.float64).reshape(3, 1).copy()
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera_matrix, zero_dist, rvec0, tvec0,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    elif len(used_ids) >= 2:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera_matrix, zero_dist, flags=cv2.SOLVEPNP_SQPNP,
        )
    else:
        _id = used_ids[0]
        # IPPE_SQUARE needs a square: the mean side (a start only — the next
        # frame refines from it with the true rectangle)
        side = board.marker_length * (1 + board.marker_aspect) / 2
        ok, r_m, t_m = cv2.solvePnP(
            marker_object_points(side).astype(np.float64), img,
            camera_matrix, zero_dist, flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        R_pnp = cv2.Rodrigues(r_m)[0]
        R_bm, t_bm = board.marker_poses[_id]
        R_cb = R_pnp @ R_bm.T                       # camera <- board
        t_cb = t_m.flatten() - R_cb @ t_bm
        rvec = cv2.Rodrigues(R_cb)[0]
        tvec = t_cb.reshape(3, 1)

    if not ok:
        return None
    err = reprojection_error(obj, img, rvec, tvec, camera_matrix)
    return rvec.flatten(), tvec.flatten(), err
