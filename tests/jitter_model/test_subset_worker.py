import math
import pickle

import cv2
import numpy as np
import pytest

from jitter_model import _subset_worker, common
from tests.jitter_model.test_common_loading import _rigidbody_dict

TRUE_RVEC = np.array([0.03, -0.05, 0.01])
TRUE_TVEC = np.array([0.0, 0.0, 0.55])


def _mean_apparent_size(frame_detections, marker_ids):
    """The same "apparent tag size in pixels" formula `geometry.pose_geometry`
    uses internally, replicated here as an independent ground truth so the
    regression test below doesn't need to reach into geometry's internals.
    """
    sizes = []
    for marker_id in marker_ids:
        corners = np.asarray(frame_detections[marker_id], dtype=np.float64)
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        sizes.append(float(np.mean(sides)))
    return float(np.mean(sizes))


def _assert_rows_match(rows_a, rows_b):
    """Every field of every row must agree, matched by `marker_ids`.

    Both `solve_burst` and the pool-worker path are deterministic on
    identical inputs, so nothing short of full equality can catch a bug that
    scrambles values without changing which subsets appear. NaN needs its
    own check since `nan != nan`: single-tag rows carry NaN normal angles by
    design (`geometry.subset_geometry`), and that NaN in the same field of
    both rows should count as a match, not a mismatch.
    """
    assert len(rows_a) == len(rows_b)
    by_ids_a = {row["marker_ids"]: row for row in rows_a}
    by_ids_b = {row["marker_ids"]: row for row in rows_b}
    assert set(by_ids_a) == set(by_ids_b)
    for marker_ids, row_a in by_ids_a.items():
        row_b = by_ids_b[marker_ids]
        assert set(row_a) == set(row_b), marker_ids
        for key, value_a in row_a.items():
            value_b = row_b[key]
            if isinstance(value_a, float):
                if math.isnan(value_a) or math.isnan(value_b):
                    assert math.isnan(value_a) and math.isnan(value_b), (
                        marker_ids,
                        key,
                        value_a,
                        value_b,
                    )
                else:
                    assert math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-9), (
                        marker_ids,
                        key,
                        value_a,
                        value_b,
                    )
            else:
                assert value_a == value_b, (marker_ids, key, value_a, value_b)


@pytest.fixture
def solver(synthetic_rig_spec, synthetic_camera):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    cameras = {
        name: common.CameraModel(
            name=name,
            K=synthetic_camera["K"],
            D=synthetic_camera["D"],
            resolution=synthetic_camera["resolution"],
        )
        for name in ("cam0", "cam1")
    }
    return common.PoseSolver(rig, cameras, np.eye(3), np.array([0.0775, 0.0, 0.0]))


@pytest.fixture
def burst_frames(solver, project_corners):
    rng = np.random.default_rng(3)
    frames = []
    for _ in range(20):
        frames.append(
            {
                marker_id: project_corners(
                    solver.rig.corners_reference[marker_id], TRUE_RVEC, TRUE_TVEC
                )
                + rng.normal(0.0, 0.2, (4, 2))
                for marker_id in (1, 2, 3)
            }
        )
    return frames


def test_enumerate_subsets_respects_the_size_limit(burst_frames):
    subsets = _subset_worker.enumerate_subsets(
        burst_frames, None, max_tags=2, presence_fraction=0.9, camera_config="cam0"
    )
    assert all(len(subset) <= 2 for subset in subsets)
    assert (1,) in subsets
    assert (1, 2) in subsets
    assert len(subsets) == 6  # three singles plus three pairs


def test_enumerate_subsets_drops_rarely_visible_tags(burst_frames):
    sparse = [dict(frame) for frame in burst_frames]
    for frame in sparse[:15]:
        frame.pop(3)
    subsets = _subset_worker.enumerate_subsets(
        sparse, None, max_tags=3, presence_fraction=0.9, camera_config="cam0"
    )
    assert all(3 not in subset for subset in subsets)


def test_solve_burst_returns_one_row_per_subset(solver, burst_frames):
    rows = _subset_worker.solve_burst(
        7,
        burst_frames,
        None,
        solver,
        camera_config="cam0",
        max_tags=3,
        presence_fraction=0.9,
        min_frames=10,
        fixed_point=np.zeros(3),
    )
    assert len(rows) == 7  # 3 singles + 3 pairs + 1 triple
    assert {row["burst"] for row in rows} == {7}
    assert {row["camera_config"] for row in rows} == {"cam0"}


def test_solve_burst_rows_carry_geometry_and_both_jitters(solver, burst_frames):
    rows = _subset_worker.solve_burst(
        1,
        burst_frames,
        None,
        solver,
        camera_config="cam0",
        max_tags=3,
        presence_fraction=0.9,
        min_frames=10,
        fixed_point=np.zeros(3),
    )
    row = next(r for r in rows if r["marker_ids"] == "1+2+3")
    for key in (
        "max_baseline_mm",
        "min_singular_mm",
        "lever_mm",
        "distance_m",
        "mean_incidence_deg",
        "pos_jitter_mm",
        "rot_jitter_mdeg",
        "frames",
        "reprojection_px",
    ):
        assert key in row
    assert row["frames"] == 20
    assert row["pos_jitter_mm"] > 0.0


def test_solve_burst_skips_subsets_with_too_few_solved_frames(solver, burst_frames):
    rows = _subset_worker.solve_burst(
        1,
        burst_frames[:5],
        None,
        solver,
        camera_config="cam0",
        max_tags=3,
        presence_fraction=0.9,
        min_frames=10,
        fixed_point=np.zeros(3),
    )
    assert rows == []


def test_more_tags_give_less_jitter_in_a_burst(solver, burst_frames):
    rows = _subset_worker.solve_burst(
        1,
        burst_frames,
        None,
        solver,
        camera_config="cam0",
        max_tags=3,
        presence_fraction=0.9,
        min_frames=10,
        fixed_point=np.zeros(3),
    )
    by_ids = {row["marker_ids"]: row["pos_jitter_mm"] for row in rows}
    assert by_ids["1+2+3"] < by_ids["1"]


def test_solve_burst_uses_the_solved_frame_at_the_median_index(solver, project_corners):
    """Regression test for a prior bug: `pose_geometry` was given
    `frames0[median_index]` (an index into the RAW, unfiltered frame list),
    while `median_index` itself is computed from `argsort(reprojections)`,
    which only has one entry per SOLVED frame. The moment any frame fails to
    solve for a subset, those two index spaces diverge silently.

    This burst makes a handful of early frames fail to solve for tag 1 (by
    deleting its detections), and gives EVERY frame a distance that grows
    with its raw index, so apparent tag size is a unique, monotonic function
    of raw frame index. That way, if the wrong raw index is ever selected
    (whichever one it is), it gives a measurably different apparent size
    from the correct one -- discrimination doesn't depend on getting lucky
    that the picked wrong index happens to be one of the deleted frames.
    """
    rng = np.random.default_rng(11)
    marker_ids = (1,)
    skip_indices = {2, 5, 8}  # early, so the solved/raw index spaces diverge
    # well before the median position, and low presence_fraction still
    # keeps tag 1 eligible (17/20 = 0.85 frames still show it).
    frames0 = []
    for index in range(20):
        tvec = TRUE_TVEC * (1.0 + 0.15 * index)
        frame = {
            marker_id: project_corners(
                solver.rig.corners_reference[marker_id], TRUE_RVEC, tvec
            )
            + rng.normal(0.0, 0.2, (4, 2))
            for marker_id in (1, 2, 3)
        }
        if index in skip_indices:
            del frame[1]  # tag 1 is absent -> mono_board_pose returns None
        frames0.append(frame)

    rows = _subset_worker.solve_burst(
        1,
        frames0,
        None,
        solver,
        camera_config="cam0",
        max_tags=1,
        presence_fraction=0.8,
        min_frames=10,
        fixed_point=np.zeros(3),
    )
    row = next(r for r in rows if r["marker_ids"] == "1")

    # Ground truth: replicate solve_burst's own filtering, in the same order,
    # to find which frame the median index actually refers to.
    solved_frames0 = []
    reprojections = []
    for frame0 in frames0:
        pose = solver.mono_board_pose(frame0, marker_ids, "cam0")
        if pose is None:
            continue
        reprojections.append(pose["rmse_px"])
        solved_frames0.append(frame0)
    median_index = int(np.argsort(reprojections)[len(reprojections) // 2])

    # Sanity check that this burst actually exercises the bug: the raw and
    # solved index spaces must have diverged by the median position, or this
    # test would pass even with the old, broken indexing.
    assert solved_frames0[median_index] is not frames0[median_index]

    expected_size = _mean_apparent_size(solved_frames0[median_index], marker_ids)
    assert row["mean_apparent_size_px"] == pytest.approx(expected_size)

    # And the frame the buggy code would have picked instead gives a
    # distinctly different answer, so a regression would not pass by luck.
    try:
        wrong_size = _mean_apparent_size(frames0[median_index], marker_ids)
    except KeyError:
        wrong_size = None
    assert wrong_size is None or not math.isclose(
        wrong_size, expected_size, rel_tol=0.05
    )


def test_worker_globals_path_matches_solve_burst_directly(
    solver,
    burst_frames,
    synthetic_rig_spec,
    synthetic_camera,
    project_corners,
    tmp_path,
):
    """`init_worker`/`worker_burst` is the real Windows pool-worker entry point;
    this drives it end to end with synthetic data and checks it agrees with
    calling `solve_burst` directly on the same frames.

    Note: `init_worker` calls `cv2.setNumThreads(0)` as a deliberate
    production side effect (avoids OpenCV threads contending with the process
    pool). That call also affects this test process's OpenCV thread count
    for the remainder of the run; harmless for these tests but worth knowing.
    """
    n_frames = len(burst_frames)
    stereo_rotation = np.eye(3)
    stereo_translation = np.array([0.0775, 0.0, 0.0])
    stereo_rvec = cv2.Rodrigues(stereo_rotation)[0]

    # Build cam1 frames from the same true board pose, composed through the
    # stereo extrinsic, so the stereo smoke test uses a geometrically
    # consistent pair of views rather than two unrelated images.
    rvec1, tvec1 = cv2.composeRT(
        TRUE_RVEC.reshape(3, 1),
        TRUE_TVEC.reshape(3, 1),
        stereo_rvec,
        stereo_translation.reshape(3, 1),
    )[:2]
    rng1 = np.random.default_rng(4)
    cam1_frames = []
    for _ in range(n_frames):
        cam1_frames.append(
            {
                marker_id: project_corners(
                    solver.rig.corners_reference[marker_id], rvec1, tvec1
                )
                + rng1.normal(0.0, 0.2, (4, 2))
                for marker_id in (1, 2, 3)
            }
        )

    rigidbody = _rigidbody_dict(synthetic_rig_spec)
    rigidbody["stereo_refined"] = {
        "rotation_cam0_to_cam1": stereo_rotation.tolist(),
        "translation_cam0_to_cam1_m": stereo_translation.tolist(),
    }
    camera_toml = {
        "camera_matrix": synthetic_camera["K"].tolist(),
        "dist_coeffs": synthetic_camera["D"].tolist(),
        "resolution": list(synthetic_camera["resolution"]),
    }
    stereo = {
        "cam0": camera_toml,
        "cam1": camera_toml,
        "stereo": {
            "R": stereo_rotation.tolist(),
            "T": (stereo_translation * 1000.0).tolist(),
        },
    }

    cache_path = tmp_path / "cache.pkl"
    with open(cache_path, "wb") as stream:
        pickle.dump(
            {
                "cameras": {
                    "cam0": {"detections": burst_frames},
                    "cam1": {"detections": cam1_frames},
                }
            },
            stream,
        )

    payload = {
        "cache_path": str(cache_path),
        "rigidbody": rigidbody,
        "stereo": stereo,
        "camera_names": ["cam0", "cam1"],
        "options": {
            "pairs": list(range(n_frames)),
            "max_tags": 3,
            "presence_fraction": 0.9,
            "min_frames": 10,
            "fixed_point": [0.0, 0.0, 0.0],
        },
    }

    try:
        _subset_worker.init_worker(payload)
        pool_rows = _subset_worker.worker_burst((1, "cam0", list(range(n_frames))))

        direct_rows = _subset_worker.solve_burst(
            1,
            burst_frames,
            None,
            solver,
            camera_config="cam0",
            max_tags=3,
            presence_fraction=0.9,
            min_frames=10,
            fixed_point=np.zeros(3),
        )
        _assert_rows_match(pool_rows, direct_rows)

        stereo_rows = _subset_worker.worker_burst((2, "stereo", list(range(n_frames))))
        assert len(stereo_rows) > 0
    finally:
        _subset_worker._SOLVER = None
        _subset_worker._CACHE = None
        _subset_worker._OPTIONS = None
