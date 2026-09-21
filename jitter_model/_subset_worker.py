"""One burst's worth of subset solving, importable by worker processes.

Kept in its own module because Windows spawns workers by re-importing, and
re-importing an analysis script would rerun the whole analysis in every worker.
"""

import itertools
import pickle
from collections import Counter

import cv2
import numpy as np

from jitter_model import common, geometry, jitter_stats

_SOLVER = None
_CACHE = None
_OPTIONS = None


def enumerate_subsets(frames0, frames1, max_tags, presence_fraction, camera_config):
    """Every subset up to `max_tags` of the tags this burst reliably shows.

    A tag qualifies only if it appears in nearly every frame of the burst, so
    the chosen sets keep most of the burst's frames usable. Stereo additionally
    requires the tag in both cameras, since the joint solve needs both views.
    """
    counts = Counter()
    for index, frame0 in enumerate(frames0):
        visible = set(frame0)
        if camera_config == "stereo":
            visible &= set(frames1[index]) if frames1[index] is not None else set()
        counts.update(visible)

    threshold = presence_fraction * len(frames0)
    eligible = sorted(
        marker_id for marker_id, count in counts.items() if count >= threshold
    )
    subsets = []
    for size in range(1, max_tags + 1):
        subsets.extend(itertools.combinations(eligible, size))
    return subsets


def solve_burst(
    burst,
    frames0,
    frames1,
    solver,
    *,
    camera_config,
    max_tags,
    presence_fraction,
    min_frames,
    fixed_point,
):
    """Per-subset jitter for one burst, with the subset frozen for the burst.

    Choosing the subset once per burst is deliberate: re-choosing per frame
    would let the estimator change inside the burst, and that switching would
    land in the standard deviation as if it were jitter.
    """
    rows = []
    for marker_ids in enumerate_subsets(
        frames0, frames1, max_tags, presence_fraction, camera_config
    ):
        rvecs = []
        tvecs = []
        reprojections = []
        solved_frames0 = []
        for index, frame0 in enumerate(frames0):
            if camera_config == "stereo":
                frame1 = frames1[index]
                if frame1 is None:
                    continue
                pose = solver.stereo_board_pose(frame0, frame1, marker_ids)
            else:
                pose = solver.mono_board_pose(frame0, marker_ids, camera_config)
            if pose is None:
                continue
            rvecs.append(pose["rvec"])
            tvecs.append(pose["tvec"])
            reprojections.append(pose["rmse_px"])
            solved_frames0.append(frame0)

        if len(rvecs) < min_frames:
            continue

        rvecs = np.asarray(rvecs)
        tvecs = np.asarray(tvecs)
        positions = jitter_stats.fixed_point_positions(rvecs, tvecs, fixed_point)

        # `median_index` indexes the SOLVED lists (rvecs/tvecs/reprojections),
        # which drop any frame that failed to solve for this subset. The
        # corner detections passed to pose_geometry must come from that same
        # filtered index space, not from `frames0`, or the median pose and
        # the "apparent size" corners silently come from different frames.
        median_index = int(np.argsort(reprojections)[len(reprojections) // 2])
        pose_terms = geometry.pose_geometry(
            solver.rig,
            marker_ids,
            rvecs[median_index],
            tvecs[median_index],
            solved_frames0[median_index],
        )

        row = {
            "burst": int(burst),
            "camera_config": camera_config,
            "marker_ids": "+".join(str(m) for m in marker_ids),
            "frames": len(rvecs),
            "reprojection_px": float(np.median(reprojections)),
        }
        row.update(geometry.subset_geometry(solver.rig, marker_ids, fixed_point))
        row.update(pose_terms)
        row.update(jitter_stats.position_jitter_mm(positions))
        row.update(jitter_stats.rotation_jitter_mdeg(rvecs))
        rows.append(row)
    return rows


def init_worker(payload):
    """Load the shared state once per worker, not once per task."""
    global _SOLVER, _CACHE, _OPTIONS
    # OpenCV's own threads would contend with the process pool and make the
    # whole run slower, so each worker stays single-threaded.
    cv2.setNumThreads(0)
    with open(payload["cache_path"], "rb") as stream:
        _CACHE = pickle.load(stream)
    rig = common.build_tag_rig(payload["rigidbody"])
    cameras = common.build_camera_models(payload["stereo"], payload["camera_names"])
    rotation, translation, _rvec = common.stereo_extrinsic(
        payload["stereo"], payload["rigidbody"]
    )
    _SOLVER = common.PoseSolver(rig, cameras, rotation, translation)
    _OPTIONS = payload["options"]


def worker_burst(task):
    """Solve one (burst, camera_config) task inside a pool worker."""
    burst, camera_config, frame_indices = task
    cam0 = _CACHE["cameras"]["cam0"]["detections"]
    cam1 = _CACHE["cameras"]["cam1"]["detections"]
    pairs = _OPTIONS["pairs"]
    frames0 = [cam0[i] for i in frame_indices]
    frames1 = [cam1[pairs[i]] if pairs[i] >= 0 else None for i in frame_indices]
    return solve_burst(
        burst,
        frames0,
        frames1,
        _SOLVER,
        camera_config=camera_config,
        max_tags=_OPTIONS["max_tags"],
        presence_fraction=_OPTIONS["presence_fraction"],
        min_frames=_OPTIONS["min_frames"],
        fixed_point=np.asarray(_OPTIONS["fixed_point"], dtype=np.float64),
    )
