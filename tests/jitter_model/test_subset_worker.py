import numpy as np
import pytest

from jitter_model import _subset_worker, common
from tests.jitter_model.test_common_loading import _rigidbody_dict

TRUE_RVEC = np.array([0.03, -0.05, 0.01])
TRUE_TVEC = np.array([0.0, 0.0, 0.55])


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
