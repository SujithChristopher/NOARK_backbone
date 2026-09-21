# Subset-Geometry Jitter Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how static jitter depends on AprilTag subset geometry (separation, orientation spread, coplanarity, lever arm) and build a validated simulator that can separate effects the dome itself confounds.

**Architecture:** Three layers. First, the pose estimators duplicated across `05_static_jitter.py` and `06_movement_error.py` move into a shared, importable `jitter_model/common.py` with explicit dependencies instead of module globals — this is what makes them testable and usable from worker processes. Second, `08_subset_geometry.py` enumerates every visible tag subset per burst, solves each with those estimators in a process pool, and fits a within-burst fixed-effects model of log jitter on geometry. Third, `09_layout_simulator.py` perturbs corners with calibrated noise, is gated on reproducing the real measurements, and only then sweeps synthetic layouts where baseline and curvature vary independently.

**Tech Stack:** Python 3.13, numpy, pandas, scipy, opencv-contrib-python (`cv2.fisheye`, `cv2.aruco`), toml, msgpack, tqdm, matplotlib, `concurrent.futures.ProcessPoolExecutor`. pytest added as the only new dependency, dev-only.

**Spec:** `docs/superpowers/specs/2026-09-21-subset-geometry-jitter-design.md`

## Global Constraints

- Python `>=3.13`; every command runs through `uv run`.
- No new runtime dependencies. `pytest` is added as a **dev** dependency only.
- Lint with `ruff check --extend-include="*.ipynb"` and `ruff format --extend-include="*.ipynb"` before each commit.
- **No temporal filtering anywhere.** Filtering trades away the jitter being measured.
- **Subsets are frozen per burst**, never re-selected per frame. Switching estimators inside a burst lands in the standard deviation as if it were jitter.
- Pose-estimator return contract is unchanged everywhere: `{"rvec": ndarray(3,), "tvec": ndarray(3,), "rmse_px": float}`, or `None` when the solve is impossible.
- `05_static_jitter.py` and `06_movement_error.py` must produce **identical** CSV output after refactoring. This is a hard gate, verified by byte comparison against baselines captured in Task 2.
- Dome facts, measured and fixed: tags on a sphere of R ≈ 115 mm, pair baselines 59–197 mm, pair normal angles 29.7–135.8 deg, `corr(baseline, normal_angle) = 0.98`.
- Detections are reused from the existing `jitter_detections.pkl`. Never rebuild them in this work.
- Default evaluation point `P_FIXED = np.zeros(3)` (the tag-1 board origin), a module constant.
- Scripts keep the repository's `# %%` cell style so they stay runnable cell-by-cell in a kernel.

---

### Task 1: Test harness and synthetic rig fixture

Everything downstream is tested against a synthetic rig whose true pose is known, so this comes first. No real recording data is needed by any unit test.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/jitter_model/__init__.py`
- Create: `tests/jitter_model/conftest.py`
- Test: `tests/jitter_model/test_conftest_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixtures `synthetic_camera` → `dict` with keys `K` (3x3 ndarray), `D` (4x1 ndarray), `resolution` (tuple); `synthetic_rig_spec` → `dict` mapping `marker_id: int` to `{"R": ndarray(3,3), "t": ndarray(3,)}` in board frame; `project_corners(object_points, rvec, tvec, camera)` → `ndarray(N,2)`.

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
uv add --dev pytest
```

- [ ] **Step 2: Write the fixtures**

Create `tests/__init__.py` and `tests/jitter_model/__init__.py` as empty files, then `tests/jitter_model/conftest.py`:

```python
"""Synthetic camera and tag rig, so the estimators can be tested without data.

The rig mimics the dome: one tag at the pole and four tilted onto facets of a
sphere of radius 115 mm, which is what the real dome measures.
"""

import cv2
import numpy as np
import pytest

SPHERE_RADIUS_M = 0.115
TAG_SIZE_M = 0.05


@pytest.fixture
def synthetic_camera():
    return {
        "K": np.array(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        # Zero distortion keeps the projection invertible to machine precision,
        # so a round-trip test failure means the estimator is wrong, not the lens.
        "D": np.zeros((4, 1), dtype=np.float64),
        "resolution": (640, 480),
    }


def _facet(azimuth_deg, tilt_deg):
    """One tag rotated onto a sphere facet, returned as board-frame R and t."""
    azimuth = np.radians(azimuth_deg)
    tilt = np.radians(tilt_deg)
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0]) * tilt
    rotation = cv2.Rodrigues(axis)[0]
    # The pole sits at the board origin; a facet centre is the pole swung
    # through `tilt` about the sphere centre, which lies at -z.
    centre = np.array([0.0, 0.0, -SPHERE_RADIUS_M])
    translation = centre + rotation @ np.array([0.0, 0.0, SPHERE_RADIUS_M])
    return rotation, translation


@pytest.fixture
def synthetic_rig_spec():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    for index, azimuth in enumerate((0.0, 90.0, 180.0, 270.0), start=2):
        rotation, translation = _facet(azimuth, 30.0)
        spec[index] = {"R": rotation, "t": translation}
    return spec


@pytest.fixture
def project_corners(synthetic_camera):
    def _project(object_points, rvec, tvec, camera=None):
        camera = camera or synthetic_camera
        projected, _ = cv2.fisheye.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera["K"],
            camera["D"],
        )
        return projected.reshape(-1, 2)

    return _project
```

- [ ] **Step 3: Write the failing test**

`tests/jitter_model/test_conftest_fixtures.py`:

```python
import numpy as np

from tests.jitter_model.conftest import SPHERE_RADIUS_M


def test_facet_tags_sit_on_the_sphere(synthetic_rig_spec):
    centre = np.array([0.0, 0.0, -SPHERE_RADIUS_M])
    for tag in synthetic_rig_spec.values():
        assert np.isclose(np.linalg.norm(tag["t"] - centre), SPHERE_RADIUS_M)


def test_facet_normals_are_tilted_thirty_degrees(synthetic_rig_spec):
    pole_normal = synthetic_rig_spec[1]["R"][:, 2]
    for marker_id in (2, 3, 4, 5):
        normal = synthetic_rig_spec[marker_id]["R"][:, 2]
        angle = np.degrees(np.arccos(np.clip(pole_normal @ normal, -1.0, 1.0)))
        assert np.isclose(angle, 30.0, atol=1e-6)


def test_projection_round_trips_a_known_point(project_corners):
    image = project_corners([[0.0, 0.0, 1.0]], np.zeros(3), np.zeros(3))
    assert np.allclose(image[0], [320.0, 240.0])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_conftest_fixtures.py -v`
Expected: 3 passed. A failure here means the fixture geometry is wrong and every later test would inherit it.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "test: add synthetic rig and camera fixtures for jitter estimators"
```

---

### Task 2: Capture output baselines for 05 and 06

The parity gate needs a reference recorded *before* anything moves. Committed outputs already exist on disk; this task freezes copies outside the recording tree so a rerun cannot overwrite them.

**Files:**
- Create: `tests/baselines/README.md`
- Create: `tests/baselines/static_jitter/*.csv` (copied)
- Create: `tests/baselines/movement_error/*.csv` (copied)
- Create: `tests/jitter_model/test_script_parity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/jitter_model/test_script_parity.py::test_static_jitter_matches_baseline` and `::test_movement_error_matches_baseline`, both marked `@pytest.mark.slow`.

- [ ] **Step 1: Copy the current CSVs to a baseline directory**

```bash
mkdir -p tests/baselines/static_jitter tests/baselines/movement_error
cp data/dome/sep18_26/dome_static_burst_sep18_26/static_jitter/*.csv tests/baselines/static_jitter/
cp data/dome/sep18_26/dome_random_movement_sep18_26/movement_error/*.csv tests/baselines/movement_error/
```

- [ ] **Step 2: Write the baseline README**

`tests/baselines/README.md`:

```markdown
# Output baselines

CSVs produced by `jitter_model/05_static_jitter.py` and
`jitter_model/06_movement_error.py` before the `common.py` extraction.

They exist to prove the refactor is behaviour-preserving. Regenerate them only
when a change is *intended* to alter the numbers, and say so in the commit
message that updates them.
```

- [ ] **Step 3: Write the parity test**

`tests/jitter_model/test_script_parity.py`:

```python
"""Byte-parity of the two analysis scripts against pre-refactor baselines.

Marked slow: these rerun the full analyses off the cached detections and take
minutes, so they are run deliberately rather than on every unit-test pass.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "tests" / "baselines"
STATIC_OUT = (
    REPO
    / "data/dome/sep18_26/dome_static_burst_sep18_26/static_jitter"
)
MOVEMENT_OUT = (
    REPO
    / "data/dome/sep18_26/dome_random_movement_sep18_26/movement_error"
)


def _run(script):
    result = subprocess.run(
        [sys.executable, str(REPO / "jitter_model" / script)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def _compare(baseline_dir, output_dir):
    baselines = sorted(baseline_dir.glob("*.csv"))
    assert baselines, f"no baselines in {baseline_dir}"
    for baseline in baselines:
        produced = output_dir / baseline.name
        assert produced.exists(), f"missing {produced}"
        assert produced.read_bytes() == baseline.read_bytes(), baseline.name


@pytest.mark.slow
def test_static_jitter_matches_baseline():
    _run("05_static_jitter.py")
    _compare(BASELINES / "static_jitter", STATIC_OUT)


@pytest.mark.slow
def test_movement_error_matches_baseline():
    _run("06_movement_error.py")
    _compare(BASELINES / "movement_error", MOVEMENT_OUT)
```

- [ ] **Step 4: Register the slow marker**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: reruns a full analysis script; minutes, not seconds"]
addopts = "-m 'not slow'"
```

- [ ] **Step 5: Run the parity tests against the unmodified scripts**

Run: `uv run pytest tests/jitter_model/test_script_parity.py -v -m slow`
Expected: 2 passed. This proves the scripts are deterministic *before* any refactor — if they fail now, stop and report it, because parity cannot be used as a gate.

- [ ] **Step 6: Commit**

```bash
git add tests/baselines tests/jitter_model/test_script_parity.py pyproject.toml
git commit -m "test: freeze 05 and 06 CSV baselines as a refactor parity gate"
```

---

### Task 3: `common.py` — calibration, rig, and camera loading

**Files:**
- Create: `jitter_model/common.py`
- Test: `tests/jitter_model/test_common_loading.py`

**Interfaces:**
- Consumes: fixtures from Task 1.
- Produces:
  - `CameraModel` frozen dataclass: `name: str`, `K: ndarray(3,3)`, `D: ndarray(-1,1)`, `resolution: tuple[int, int]`
  - `TagRig` frozen dataclass: `tag_size_m: float`, `reference_id: int`, `marker_ids: tuple[int, ...]`, `corners_reference: dict[int, ndarray(4,3)]`, `marker_to_reference: dict[int, tuple[ndarray(3,3), ndarray(3,)]]`, plus methods `centers() -> dict[int, ndarray(3,)]` and `normals() -> dict[int, ndarray(3,)]`
  - `tag_corners_local(tag_size_m) -> ndarray(4,3)`
  - `build_tag_rig(rigidbody: dict) -> TagRig`
  - `build_camera_models(stereo: dict, names: Sequence[str]) -> dict[str, CameraModel]`
  - `stereo_extrinsic(stereo: dict, rigidbody: dict) -> tuple[ndarray(3,3), ndarray(3,), ndarray(3,1)]` returning `(R, T, rvec)`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_common_loading.py`:

```python
import numpy as np
import pytest

from jitter_model import common


def _rigidbody_dict(spec, tag_size_m=0.05):
    return {
        "meta": {
            "tag_size_m": tag_size_m,
            "reference_id": 1,
            "marker_ids": sorted(spec),
        },
        "markers": {
            str(marker_id): {
                "rotation_marker_to_reference": tag["R"].tolist(),
                "translation_marker_to_reference_m": tag["t"].tolist(),
            }
            for marker_id, tag in spec.items()
        },
    }


def test_tag_corners_local_is_a_centred_square():
    corners = common.tag_corners_local(0.05)
    assert corners.shape == (4, 3)
    assert np.allclose(corners.mean(axis=0), 0.0)
    assert np.allclose(corners[:, 2], 0.0)
    side = np.linalg.norm(corners[1] - corners[0])
    assert np.isclose(side, 0.05)


def test_build_tag_rig_places_the_reference_tag_at_the_origin(synthetic_rig_spec):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    assert rig.reference_id == 1
    assert np.allclose(rig.centers()[1], 0.0)
    assert np.allclose(rig.corners_reference[1], common.tag_corners_local(0.05))


def test_tag_rig_normals_match_the_facet_tilt(synthetic_rig_spec):
    rig = common.build_tag_rig(_rigidbody_dict(synthetic_rig_spec))
    normals = rig.normals()
    angle = np.degrees(np.arccos(np.clip(normals[1] @ normals[2], -1.0, 1.0)))
    assert np.isclose(angle, 30.0, atol=1e-6)


def test_build_camera_models_reshapes_distortion(synthetic_camera):
    stereo = {
        "cam0": {
            "camera_matrix": synthetic_camera["K"].tolist(),
            "dist_coeffs": [0.1, 0.01, 0.0, 0.0],
            "resolution": [640, 480],
        }
    }
    models = common.build_camera_models(stereo, ["cam0"])
    assert models["cam0"].D.shape == (4, 1)
    assert models["cam0"].resolution == (640, 480)


def test_stereo_extrinsic_prefers_the_refined_block():
    identity = np.eye(3).tolist()
    stereo = {"stereo": {"R": identity, "T": [70.0, 0.0, 0.0]}}
    rigidbody = {
        "stereo_refined": {
            "rotation_cam0_to_cam1": identity,
            "translation_cam0_to_cam1_m": [0.0775, 0.0, 0.0],
        }
    }
    _R, T, _rvec = common.stereo_extrinsic(stereo, rigidbody)
    assert np.isclose(T[0], 0.0775)


def test_stereo_extrinsic_falls_back_and_converts_mm():
    stereo = {"stereo": {"R": np.eye(3).tolist(), "T": [70.0, 0.0, 0.0]}}
    with pytest.warns(UserWarning):
        _R, T, _rvec = common.stereo_extrinsic(stereo, {})
    assert np.isclose(T[0], 0.070)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_common_loading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model'`.

- [ ] **Step 3: Make `jitter_model` importable**

Create `jitter_model/__init__.py` as an empty file, and add the package to the wheel build so imports resolve the same way `support` does. In `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["support", "jitter_model"]
```

Then `uv sync` to re-install the project in editable mode.

- [ ] **Step 4: Write the implementation**

`jitter_model/common.py`:

```python
"""Shared calibration, rig, and pose machinery for the jitter analyses.

`05_static_jitter.py` and `06_movement_error.py` each carried their own copy of
this code. A third copy in `08_subset_geometry.py` is where copies start to
drift, so it lives here once, with its dependencies passed in rather than read
from module globals — which is also what lets worker processes and unit tests
use it.
"""

import warnings
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    D: np.ndarray
    resolution: tuple


@dataclass(frozen=True)
class TagRig:
    tag_size_m: float
    reference_id: int
    marker_ids: tuple
    corners_reference: dict
    marker_to_reference: dict

    def centers(self):
        """Each tag's centre in the reference tag's frame."""
        return {
            marker_id: corners.mean(axis=0)
            for marker_id, corners in self.corners_reference.items()
        }

    def normals(self):
        """Each tag's outward normal in the reference tag's frame."""
        return {
            marker_id: rotation[:, 2]
            for marker_id, (rotation, _t) in self.marker_to_reference.items()
        }


def tag_corners_local(tag_size_m):
    """One tag's four corners in its own frame, counter-clockwise from top-left."""
    half = tag_size_m / 2.0
    return np.asarray(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def build_tag_rig(rigidbody):
    """Every tag's corners in the reference tag's frame, as one rigid point cloud.

    Expressing all tags in one frame is what lets any subset be solved by a
    single PnP instead of averaging independent per-tag poses.
    """
    tag_size_m = float(rigidbody["meta"]["tag_size_m"])
    local = tag_corners_local(tag_size_m)
    corners_reference = {}
    marker_to_reference = {}
    for marker_id_text, marker_data in rigidbody["markers"].items():
        rotation = np.asarray(
            marker_data["rotation_marker_to_reference"], dtype=np.float64
        )
        translation = np.asarray(
            marker_data["translation_marker_to_reference_m"], dtype=np.float64
        )
        marker_id = int(marker_id_text)
        corners_reference[marker_id] = local @ rotation.T + translation
        marker_to_reference[marker_id] = (rotation, translation)
    return TagRig(
        tag_size_m=tag_size_m,
        reference_id=int(rigidbody["meta"]["reference_id"]),
        marker_ids=tuple(int(x) for x in rigidbody["meta"]["marker_ids"]),
        corners_reference=corners_reference,
        marker_to_reference=marker_to_reference,
    )


def build_camera_models(stereo, names):
    models = {}
    for name in names:
        camera_data = stereo[name]
        models[name] = CameraModel(
            name=name,
            K=np.asarray(camera_data["camera_matrix"], dtype=np.float64),
            D=np.asarray(camera_data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            resolution=tuple(camera_data["resolution"]),
        )
    return models


def stereo_extrinsic(stereo, rigidbody):
    """Rotation and translation cam0 -> cam1, preferring notebook 02's refit.

    The self-calibrated extrinsic was solved against this very rig, so it beats
    the checkerboard one; the calibration TOML is the fallback and states its
    translation in millimetres.
    """
    if "stereo_refined" in rigidbody:
        rotation = np.asarray(
            rigidbody["stereo_refined"]["rotation_cam0_to_cam1"], dtype=np.float64
        )
        translation = np.asarray(
            rigidbody["stereo_refined"]["translation_cam0_to_cam1_m"], dtype=np.float64
        )
    else:
        warnings.warn("No refined stereo extrinsic; falling back to calibration TOML")
        rotation = np.asarray(stereo["stereo"]["R"], dtype=np.float64)
        translation = np.asarray(stereo["stereo"]["T"], dtype=np.float64) / 1000.0
    return rotation, translation, cv2.Rodrigues(rotation)[0]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/jitter_model/test_common_loading.py -v`
Expected: 6 passed.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format jitter_model/common.py tests/jitter_model/
uv run ruff check jitter_model/ tests/
git add jitter_model/__init__.py jitter_model/common.py tests/jitter_model/test_common_loading.py pyproject.toml uv.lock
git commit -m "refactor: extract tag rig and camera loading into jitter_model.common"
```

---

### Task 4: `common.py` — the pose estimators

The three estimators move verbatim in behaviour; the only change is that `camera_models`, `marker_corners_reference`, `marker_to_reference`, `TAG_CORNERS_LOCAL`, `RVEC_STEREO` and `T_STEREO` become instance state on a `PoseSolver` instead of module globals.

**Files:**
- Modify: `jitter_model/common.py`
- Test: `tests/jitter_model/test_pose_solver.py`

**Interfaces:**
- Consumes: `TagRig`, `CameraModel` from Task 3.
- Produces: `PoseSolver(rig: TagRig, cameras: dict[str, CameraModel], stereo_rotation=None, stereo_translation=None)` with methods:
  - `stack_correspondences(frame_detections, marker_ids) -> tuple[ndarray(4N,3), ndarray(4N,2)] | None`
  - `raw_reprojection_rmse(object_points, image_points, rvec, tvec, camera_name) -> float`
  - `single_tag_pose(frame_detections, marker_id, camera_name) -> dict | None`
  - `seed_pose(frame_detections, marker_ids, camera_name) -> dict | None`
  - `mono_board_pose(frame_detections, marker_ids, camera_name="cam0") -> dict | None`
  - `stereo_board_pose(frame0, frame1, marker_ids) -> dict | None`
  - Every pose dict is `{"rvec": ndarray(3,), "tvec": ndarray(3,), "rmse_px": float}`.

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_pose_solver.py`:

```python
"""Round-trip tests: project a known pose, solve it back, expect it recovered.

With zero distortion the projection is exact, so any error here is the
estimator's, not the lens model's.
"""

import cv2
import numpy as np
import pytest

from jitter_model import common
from tests.jitter_model.test_common_loading import _rigidbody_dict

TRUE_RVEC = np.array([0.05, -0.10, 0.02])
TRUE_TVEC = np.array([0.01, -0.02, 0.60])


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
    stereo_rotation = np.eye(3)
    stereo_translation = np.array([0.0775, 0.0, 0.0])
    return common.PoseSolver(rig, cameras, stereo_rotation, stereo_translation)


def _detections(solver, marker_ids, project_corners, rvec=TRUE_RVEC, tvec=TRUE_TVEC):
    return {
        marker_id: project_corners(
            solver.rig.corners_reference[marker_id], rvec, tvec
        )
        for marker_id in marker_ids
    }


def test_single_tag_pose_recovers_the_board_pose(solver, project_corners):
    detections = _detections(solver, [1], project_corners)
    pose = solver.single_tag_pose(detections, 1, "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-4)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-3)
    assert pose["rmse_px"] < 1e-3


def test_single_tag_pose_of_an_offset_tag_still_reports_the_board(
    solver, project_corners
):
    detections = _detections(solver, [3], project_corners)
    pose = solver.single_tag_pose(detections, 3, "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-3)


def test_single_tag_pose_returns_none_when_the_tag_is_absent(solver):
    assert solver.single_tag_pose({}, 1, "cam0") is None


def test_mono_board_pose_recovers_the_pose_from_four_tags(solver, project_corners):
    detections = _detections(solver, [1, 2, 3, 4], project_corners)
    pose = solver.mono_board_pose(detections, (1, 2, 3, 4), "cam0")
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-5)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-5)


def test_mono_board_pose_requires_every_requested_tag(solver, project_corners):
    detections = _detections(solver, [1, 2], project_corners)
    assert solver.mono_board_pose(detections, (1, 2, 3), "cam0") is None


def test_stack_correspondences_orders_object_and_image_points_together(
    solver, project_corners
):
    detections = _detections(solver, [1, 2], project_corners)
    object_points, image_points = solver.stack_correspondences(detections, (1, 2))
    assert object_points.shape == (8, 3)
    assert image_points.shape == (8, 2)
    assert np.allclose(object_points[:4], solver.rig.corners_reference[1])


def test_stereo_board_pose_recovers_the_pose(solver, project_corners):
    frame0 = _detections(solver, [1, 2, 3], project_corners)
    rvec1, tvec1 = cv2.composeRT(
        TRUE_RVEC.reshape(3, 1),
        TRUE_TVEC.reshape(3, 1),
        cv2.Rodrigues(np.eye(3))[0],
        np.array([0.0775, 0.0, 0.0]).reshape(3, 1),
    )[:2]
    frame1 = {
        marker_id: project_corners(
            solver.rig.corners_reference[marker_id], rvec1, tvec1
        )
        for marker_id in (1, 2, 3)
    }
    pose = solver.stereo_board_pose(frame0, frame1, (1, 2, 3))
    assert np.allclose(pose["tvec"], TRUE_TVEC, atol=1e-4)
    assert np.allclose(pose["rvec"], TRUE_RVEC, atol=1e-4)


def test_more_tags_reduce_jitter_under_corner_noise(solver, project_corners):
    rng = np.random.default_rng(0)

    def spread(marker_ids):
        positions = []
        clean = _detections(solver, marker_ids, project_corners)
        for _ in range(60):
            noisy = {
                marker_id: corners + rng.normal(0.0, 0.3, corners.shape)
                for marker_id, corners in clean.items()
            }
            pose = solver.mono_board_pose(noisy, tuple(marker_ids), "cam0")
            positions.append(pose["tvec"])
        return np.linalg.norm(np.std(np.asarray(positions), axis=0, ddof=1))

    assert spread([1, 2, 3, 4]) < spread([1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_pose_solver.py -v`
Expected: FAIL with `AttributeError: module 'jitter_model.common' has no attribute 'PoseSolver'`.

- [ ] **Step 3: Implement `PoseSolver`**

Append to `jitter_model/common.py`. The bodies are the ones currently in `05_static_jitter.py:392-592`, with `camera_models[name]` becoming `self.cameras[name]`, `marker_corners_reference` becoming `self.rig.corners_reference`, `marker_to_reference` becoming `self.rig.marker_to_reference`, `TAG_CORNERS_LOCAL` becoming `self._local_corners`, and `RVEC_STEREO` / `T_STEREO` becoming `self._stereo_rvec` / `self._stereo_translation`. Keep every docstring and comment.

```python
from scipy.optimize import least_squares


class PoseSolver:
    """Joint board-PnP estimators over any subset of the rig's tags.

    One solve over every visible corner, never an average of per-tag `tvec`s:
    averaging independent poses throws away the constraint that the tags are
    one rigid body, which is most of what the extra tags are worth.
    """

    def __init__(self, rig, cameras, stereo_rotation=None, stereo_translation=None):
        self.rig = rig
        self.cameras = cameras
        self._local_corners = tag_corners_local(rig.tag_size_m)
        self._stereo_translation = (
            None if stereo_translation is None
            else np.asarray(stereo_translation, dtype=np.float64).reshape(3, 1)
        )
        self._stereo_rvec = (
            None if stereo_rotation is None else cv2.Rodrigues(stereo_rotation)[0]
        )

    def stack_correspondences(self, frame_detections, marker_ids):
        """Every requested tag's corners as one board, or None if any is missing.

        Requiring the whole set keeps a condition's geometry fixed: a 4-tag
        number is always four tags, never whichever two happened to be visible.
        """
        if not all(marker_id in frame_detections for marker_id in marker_ids):
            return None
        return (
            np.concatenate(
                [self.rig.corners_reference[m] for m in marker_ids]
            ),
            np.concatenate([frame_detections[m] for m in marker_ids]),
        )

    def raw_reprojection_rmse(
        self, object_points, image_points, rvec, tvec, camera_name
    ):
        camera = self.cameras[camera_name]
        projected, _ = cv2.fisheye.projectPoints(
            object_points.reshape(-1, 1, 3), rvec, tvec, camera.K, camera.D
        )
        residual = projected.reshape(-1, 2) - image_points
        return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    def _undistort(self, image_points_raw, camera):
        return cv2.fisheye.undistortPoints(
            image_points_raw.reshape(-1, 1, 2), camera.K, camera.D, P=camera.K
        ).reshape(-1, 2)

    def single_tag_pose(self, frame_detections, marker_id, camera_name):
        ...  # body of 05_static_jitter.py:415-467, globals replaced as above

    def seed_pose(self, frame_detections, marker_ids, camera_name):
        ...  # body of 05_static_jitter.py:469-481

    def mono_board_pose(self, frame_detections, marker_ids, camera_name="cam0"):
        ...  # body of 05_static_jitter.py:483-527

    def stereo_board_pose(self, frame0, frame1, marker_ids):
        ...  # body of 05_static_jitter.py:529-592, using self._stereo_rvec
        #     and self._stereo_translation in the cv2.composeRT call
```

Replace each `...` with the corresponding function body from `05_static_jitter.py`, indented as a method and with the global lookups substituted. Do not change any numerical step: the `IPPE_SQUARE` branch selection by `tvec[2] > 0` and lowest RMSE, the `SOLVEPNP_ITERATIVE` seeding, and the `least_squares` residual over both cameras all stay exactly as written.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_pose_solver.py -v`
Expected: 8 passed. If `test_stereo_board_pose_recovers_the_pose` fails on the sign of the stereo translation, the `composeRT` argument order was changed — restore it to `composeRT(rvec0, tvec0, stereo_rvec, stereo_translation)`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/common.py tests/jitter_model/test_pose_solver.py
uv run ruff check jitter_model/ tests/
git add jitter_model/common.py tests/jitter_model/test_pose_solver.py
git commit -m "refactor: move the board-PnP estimators into jitter_model.common.PoseSolver"
```

---

### Task 5: `common.py` — detection cache and camera pairing

**Files:**
- Modify: `jitter_model/common.py`
- Test: `tests/jitter_model/test_common_cache.py`

**Interfaces:**
- Consumes: `CameraModel`, `TagRig`.
- Produces:
  - `load_timestamp_records(path) -> dict` with keys `sync`, `monotonic_ns`, `burst` (zeros when the recording has no burst column)
  - `cache_is_compatible(cache, *, marker_ids, tag_size_m, recording_dir, camera_names) -> bool`
  - `load_detection_cache(cache_path, *, marker_ids, tag_size_m, recording_dir, camera_names) -> dict | None` — returns `None` when missing or stale, never rebuilds
  - `pair_cameras(cam0_monotonic_ns, cam1_monotonic_ns, max_fraction_of_frame) -> ndarray(int)` giving, per cam0 frame, the nearest cam1 index or `-1`

Detection *building* stays in `05`/`06`; this work never rebuilds detections, so only the read path moves.

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_common_cache.py`:

```python
import pickle

import numpy as np

from jitter_model import common


def _cache(tmp_path, marker_ids=(1, 2), tag_size_m=0.05):
    payload = {
        "version": 1,
        "recording_dir": str(tmp_path.resolve()),
        "marker_ids": marker_ids,
        "tag_size_m": tag_size_m,
        "cameras": {"cam0": {"detections": [], "metadata": {}},
                    "cam1": {"detections": [], "metadata": {}}},
    }
    path = tmp_path / "jitter_detections.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def test_load_detection_cache_returns_the_payload(tmp_path):
    path = _cache(tmp_path)
    cache = common.load_detection_cache(
        path,
        marker_ids=(1, 2),
        tag_size_m=0.05,
        recording_dir=tmp_path,
        camera_names=("cam0", "cam1"),
    )
    assert set(cache["cameras"]) == {"cam0", "cam1"}


def test_load_detection_cache_rejects_a_stale_tag_size(tmp_path):
    path = _cache(tmp_path)
    assert (
        common.load_detection_cache(
            path,
            marker_ids=(1, 2),
            tag_size_m=0.06,
            recording_dir=tmp_path,
            camera_names=("cam0", "cam1"),
        )
        is None
    )


def test_load_detection_cache_returns_none_when_missing(tmp_path):
    assert (
        common.load_detection_cache(
            tmp_path / "absent.pkl",
            marker_ids=(1,),
            tag_size_m=0.05,
            recording_dir=tmp_path,
            camera_names=("cam0",),
        )
        is None
    )


def test_pair_cameras_matches_the_nearest_frame():
    period = 33_000_000
    cam0 = np.arange(5, dtype=np.int64) * period
    cam1 = cam0 + 2_000_000
    pairs = common.pair_cameras(cam0, cam1, 0.55)
    assert pairs.tolist() == [0, 1, 2, 3, 4]


def test_pair_cameras_rejects_gaps_beyond_the_tolerance():
    period = 33_000_000
    cam0 = np.arange(3, dtype=np.int64) * period
    cam1 = cam0 + int(0.9 * period)
    pairs = common.pair_cameras(cam0, cam1, 0.55)
    assert (pairs == -1).all()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_common_cache.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_detection_cache'`.

- [ ] **Step 3: Implement**

Append to `jitter_model/common.py`, moving `load_timestamp_records` from `05_static_jitter.py:232-255` verbatim and turning `cache_is_compatible` and the cam0/cam1 pairing block (`05_static_jitter.py:304-390`) into the parameterised functions above:

```python
import pickle

import msgpack
import msgpack_numpy as mpn

CACHE_VERSION = 1


def load_timestamp_records(path):
    """Load the per-frame timestamp columns, including this take's burst index.

    The burst recorder appends a sixth column holding the 1-based burst number,
    which is what groups frames into repeats of one pose. Older recordings stop
    at five columns, so it is read only when present.
    """
    ...  # body of 05_static_jitter.py:232-255


def cache_is_compatible(cache, *, marker_ids, tag_size_m, recording_dir, camera_names):
    return (
        cache.get("version") == CACHE_VERSION
        and tuple(cache.get("marker_ids", ())) == tuple(marker_ids)
        and cache.get("tag_size_m") == tag_size_m
        and cache.get("recording_dir") == str(Path(recording_dir).resolve())
        and all(name in cache.get("cameras", {}) for name in camera_names)
    )


def load_detection_cache(
    cache_path, *, marker_ids, tag_size_m, recording_dir, camera_names
):
    """The cached detections, or None when absent or stale. Never rebuilds."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    if not cache_is_compatible(
        cache,
        marker_ids=marker_ids,
        tag_size_m=tag_size_m,
        recording_dir=recording_dir,
        camera_names=camera_names,
    ):
        return None
    return cache


def pair_cameras(cam0_monotonic_ns, cam1_monotonic_ns, max_fraction_of_frame):
    """Nearest cam1 frame for each cam0 frame, or -1 when none is close enough.

    The two cameras free-run, so they are paired on the shared host-monotonic
    clock rather than assumed to be in lockstep.
    """
    cam0_monotonic_ns = np.asarray(cam0_monotonic_ns, dtype=np.int64)
    cam1_monotonic_ns = np.asarray(cam1_monotonic_ns, dtype=np.int64)
    period_ns = float(np.median(np.diff(cam0_monotonic_ns)))
    tolerance_ns = max_fraction_of_frame * period_ns
    insert = np.searchsorted(cam1_monotonic_ns, cam0_monotonic_ns)
    pairs = np.full(len(cam0_monotonic_ns), -1, dtype=np.int64)
    for index, timestamp in enumerate(cam0_monotonic_ns):
        candidates = [
            candidate
            for candidate in (insert[index] - 1, insert[index])
            if 0 <= candidate < len(cam1_monotonic_ns)
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda c: abs(cam1_monotonic_ns[c] - timestamp))
        if abs(cam1_monotonic_ns[best] - timestamp) <= tolerance_ns:
            pairs[index] = best
    return pairs
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_common_cache.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/common.py tests/jitter_model/test_common_cache.py
uv run ruff check jitter_model/ tests/
git add jitter_model/common.py tests/jitter_model/test_common_cache.py
git commit -m "refactor: move detection-cache reading and camera pairing into common"
```

---

### Task 6: Migrate `05` and `06` onto `common.py`

**Files:**
- Modify: `jitter_model/05_static_jitter.py:141-592` (calibration block, cache block, estimator block)
- Modify: `jitter_model/06_movement_error.py:170-600` (same three blocks)

**Interfaces:**
- Consumes: everything from Tasks 3-5.
- Produces: no new interfaces. Both scripts keep their existing module-level names (`camera_models`, `mono_board_pose`, `stereo_board_pose`, …) as thin bindings, so the rest of each script is untouched and the diff stays small.

- [ ] **Step 1: Rewrite the calibration block in `05`**

Replace the body that builds `camera_models`, `R_STEREO`/`T_STEREO`/`RVEC_STEREO`, `TAG_CORNERS_LOCAL`, `marker_corners_reference` and `marker_to_reference` with:

```python
from jitter_model import common

camera_models = common.build_camera_models(stereo, CAMERA_NAMES)
R_STEREO, T_STEREO, RVEC_STEREO = common.stereo_extrinsic(stereo, rigidbody)
rig = common.build_tag_rig(rigidbody)
TAG_SIZE_M = rig.tag_size_m
REFERENCE_ID = rig.reference_id
DETECT_MARKER_IDS = rig.marker_ids
TAG_CORNERS_LOCAL = common.tag_corners_local(TAG_SIZE_M)
marker_corners_reference = rig.corners_reference
marker_to_reference = rig.marker_to_reference
```

`detect_camera` reads `camera_models[name]["resolution"]`; change that one access to `camera_models[name].resolution`, and the `K`/`D` accesses in the same function likewise.

- [ ] **Step 2: Replace the estimator definitions in `05` with bindings**

Delete the five function definitions at `05_static_jitter.py:392-592` and put in their place:

```python
solver = common.PoseSolver(rig, camera_models, R_STEREO, T_STEREO)
stack_correspondences = solver.stack_correspondences
raw_reprojection_rmse = solver.raw_reprojection_rmse
single_tag_pose = solver.single_tag_pose
seed_pose = solver.seed_pose
mono_board_pose = solver.mono_board_pose
stereo_board_pose = solver.stereo_board_pose
```

- [ ] **Step 3: Replace the cache and pairing blocks in `05`**

```python
detection_cache = None
if not REBUILD_DETECTION_CACHE:
    detection_cache = common.load_detection_cache(
        DETECTION_CACHE,
        marker_ids=DETECT_MARKER_IDS,
        tag_size_m=TAG_SIZE_M,
        recording_dir=RECORDING_DIR,
        camera_names=CAMERA_NAMES,
    )
```

Keep the existing rebuild branch exactly as it is — building detections stays in the script. Replace the hand-rolled `nearest_cam1_index` loop with `cam1_pair = common.pair_cameras(cam0_monotonic_ns, cam1_monotonic_ns, MAX_PAIR_FRACTION_OF_FRAME)`, keeping the existing print of the pairing statistics.

- [ ] **Step 4: Run the static-jitter parity gate**

Run: `uv run pytest tests/jitter_model/test_script_parity.py::test_static_jitter_matches_baseline -v -m slow`
Expected: PASS. A failure means the refactor changed a number — find it before going further; do not update the baseline.

- [ ] **Step 5: Apply the same three replacements to `06`**

`06_movement_error.py` has the same blocks with the same names, minus the burst column. Make the identical substitutions.

- [ ] **Step 6: Run the movement-error parity gate and the whole unit suite**

Run: `uv run pytest tests/jitter_model/test_script_parity.py -v -m slow && uv run pytest tests/ -v`
Expected: 2 slow passed, all unit tests passed.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format jitter_model/
uv run ruff check jitter_model/ tests/
git add jitter_model/05_static_jitter.py jitter_model/06_movement_error.py
git commit -m "refactor: point 05 and 06 at jitter_model.common, output unchanged"
```

---

### Task 7: `geometry.py` — subset geometry metrics

**Files:**
- Create: `jitter_model/geometry.py`
- Test: `tests/jitter_model/test_geometry.py`

**Interfaces:**
- Consumes: `TagRig` from Task 3.
- Produces:
  - `subset_geometry(rig, marker_ids, fixed_point) -> dict` with keys `max_baseline_mm`, `rms_radius_mm`, `min_singular_mm`, `max_normal_angle_deg`, `mean_normal_angle_deg`, `lever_mm`, `n_tags`, `n_corners`
  - `pose_geometry(rig, marker_ids, rvec, tvec, frame_detections) -> dict` with keys `distance_m`, `mean_incidence_deg`, `min_incidence_deg`, `mean_apparent_size_px`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_geometry.py`:

```python
import numpy as np

from jitter_model import common, geometry
from tests.jitter_model.test_common_loading import _rigidbody_dict


def _rig(spec):
    return common.build_tag_rig(_rigidbody_dict(spec))


def test_single_tag_has_no_baseline_and_no_thickness():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    result = geometry.subset_geometry(_rig(spec), (1,), np.zeros(3))
    assert result["max_baseline_mm"] == 0.0
    assert result["min_singular_mm"] < 1e-9
    assert result["n_corners"] == 4


def test_baseline_is_the_largest_pairwise_centre_distance():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
        3: {"R": np.eye(3), "t": np.array([0.04, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (1, 2, 3), np.zeros(3))
    assert np.isclose(result["max_baseline_mm"], 100.0)


def test_coplanar_tags_have_near_zero_min_singular():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (1, 2), np.zeros(3))
    assert result["min_singular_mm"] < 1e-6


def test_tilted_tags_have_a_real_min_singular(synthetic_rig_spec):
    result = geometry.subset_geometry(_rig(synthetic_rig_spec), (1, 2), np.zeros(3))
    assert result["min_singular_mm"] > 1.0


def test_normal_angle_matches_the_facet_tilt(synthetic_rig_spec):
    result = geometry.subset_geometry(_rig(synthetic_rig_spec), (1, 2), np.zeros(3))
    assert np.isclose(result["max_normal_angle_deg"], 30.0, atol=1e-6)


def test_lever_is_the_centroid_to_fixed_point_distance():
    spec = {
        1: {"R": np.eye(3), "t": np.zeros(3)},
        2: {"R": np.eye(3), "t": np.array([0.10, 0.0, 0.0])},
    }
    result = geometry.subset_geometry(_rig(spec), (2,), np.zeros(3))
    assert np.isclose(result["lever_mm"], 100.0)


def test_incidence_is_zero_for_a_tag_facing_the_camera():
    spec = {1: {"R": np.eye(3), "t": np.zeros(3)}}
    rig = _rig(spec)
    detections = {1: np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])}
    result = geometry.pose_geometry(
        rig, (1,), np.zeros(3), np.array([0.0, 0.0, 0.5]), detections
    )
    assert np.isclose(result["distance_m"], 0.5)
    assert result["mean_incidence_deg"] < 1e-6
    assert np.isclose(result["mean_apparent_size_px"], 10.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model.geometry'`.

- [ ] **Step 3: Implement**

`jitter_model/geometry.py`:

```python
"""Geometry descriptors for one tag subset.

Everything here is a candidate predictor in the jitter model, so each is
defined once and computed the same way for measured and simulated layouts.
"""

import itertools

import cv2
import numpy as np


def subset_geometry(rig, marker_ids, fixed_point):
    """Exact board-frame geometry of a subset, independent of any camera pose."""
    marker_ids = tuple(marker_ids)
    centers = rig.centers()
    normals = rig.normals()
    positions = np.asarray([centers[m] for m in marker_ids])
    corners = np.concatenate([rig.corners_reference[m] for m in marker_ids])

    if len(marker_ids) > 1:
        baselines = [
            np.linalg.norm(centers[a] - centers[b])
            for a, b in itertools.combinations(marker_ids, 2)
        ]
        angles = [
            np.degrees(np.arccos(np.clip(normals[a] @ normals[b], -1.0, 1.0)))
            for a, b in itertools.combinations(marker_ids, 2)
        ]
    else:
        baselines = [0.0]
        angles = [0.0]

    centroid = positions.mean(axis=0)
    # The smallest singular value of the mean-centred corner cloud is the
    # cloud's thickness out of its best-fit plane: near zero means the subset
    # sits on the planar-ambiguity ridge where PnP is worst conditioned.
    singular = np.linalg.svd(corners - corners.mean(axis=0), compute_uv=False)

    return {
        "n_tags": len(marker_ids),
        "n_corners": 4 * len(marker_ids),
        "max_baseline_mm": 1000.0 * float(np.max(baselines)),
        "rms_radius_mm": 1000.0
        * float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1)))),
        "min_singular_mm": 1000.0 * float(singular[-1]),
        "max_normal_angle_deg": float(np.max(angles)),
        "mean_normal_angle_deg": float(np.mean(angles)),
        "lever_mm": 1000.0 * float(np.linalg.norm(centroid - np.asarray(fixed_point))),
    }


def pose_geometry(rig, marker_ids, rvec, tvec, frame_detections):
    """Pose-dependent descriptors: how far away and how obliquely it was seen."""
    marker_ids = tuple(marker_ids)
    rotation = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    centers = rig.centers()
    normals = rig.normals()

    incidences = []
    sizes = []
    for marker_id in marker_ids:
        centre_cam = rotation @ centers[marker_id] + translation
        normal_cam = rotation @ normals[marker_id]
        viewing = centre_cam / np.linalg.norm(centre_cam)
        # The tag may be defined with either facing, so the obtuse case is
        # folded back: what matters is the tilt away from face-on, not the sign.
        cosine = abs(float(normal_cam @ viewing))
        incidences.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        corners = np.asarray(frame_detections[marker_id], dtype=np.float64)
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        sizes.append(float(np.mean(sides)))

    centroid_cam = (
        rotation @ np.asarray([centers[m] for m in marker_ids]).mean(axis=0)
        + translation
    )
    return {
        "distance_m": float(np.linalg.norm(centroid_cam)),
        "mean_incidence_deg": float(np.mean(incidences)),
        "min_incidence_deg": float(np.min(incidences)),
        "mean_apparent_size_px": float(np.mean(sizes)),
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_geometry.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/geometry.py tests/jitter_model/test_geometry.py
uv run ruff check jitter_model/ tests/
git add jitter_model/geometry.py tests/jitter_model/test_geometry.py
git commit -m "feat: add subset geometry descriptors for the jitter model"
```

---

### Task 8: `jitter_stats.py` — position and rotation jitter

**Files:**
- Create: `jitter_model/jitter_stats.py`
- Test: `tests/jitter_model/test_jitter_stats.py`

**Interfaces:**
- Consumes: nothing beyond numpy/cv2.
- Produces:
  - `fixed_point_positions(rvecs, tvecs, fixed_point) -> ndarray(N,3)`
  - `position_jitter_mm(positions) -> dict` with keys `pos_jitter_mm`, `pos_jitter_x_mm`, `pos_jitter_y_mm`, `pos_jitter_z_mm`
  - `chordal_mean_rotation(rotations) -> ndarray(3,3)`
  - `rotation_jitter_mdeg(rvecs) -> dict` with keys `rot_jitter_mdeg`, `rot_jitter_x_mdeg`, `rot_jitter_y_mdeg`, `rot_jitter_z_mdeg`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_jitter_stats.py`:

```python
import cv2
import numpy as np

from jitter_model import jitter_stats


def test_fixed_point_positions_applies_the_board_offset():
    rvecs = np.zeros((3, 3))
    tvecs = np.tile([0.0, 0.0, 0.5], (3, 1))
    positions = jitter_stats.fixed_point_positions(
        rvecs, tvecs, np.array([0.1, 0.0, 0.0])
    )
    assert np.allclose(positions, [[0.1, 0.0, 0.5]] * 3)


def test_fixed_point_positions_rotates_the_offset():
    rvec = np.array([0.0, 0.0, np.pi / 2])
    positions = jitter_stats.fixed_point_positions(
        rvec.reshape(1, 3), np.zeros((1, 3)), np.array([0.1, 0.0, 0.0])
    )
    assert np.allclose(positions[0], [0.0, 0.1, 0.0], atol=1e-9)


def test_position_jitter_is_the_norm_of_the_per_axis_std():
    rng = np.random.default_rng(1)
    positions = rng.normal(0.0, 0.001, (500, 3))
    result = jitter_stats.position_jitter_mm(positions)
    assert np.isclose(result["pos_jitter_mm"], np.sqrt(3.0), rtol=0.15)
    assert np.isclose(
        result["pos_jitter_mm"],
        np.linalg.norm(
            [result[f"pos_jitter_{a}_mm"] for a in "xyz"]
        ),
    )


def test_position_jitter_is_zero_for_a_still_estimate():
    result = jitter_stats.position_jitter_mm(np.tile([0.1, 0.2, 0.3], (10, 1)))
    assert result["pos_jitter_mm"] == 0.0


def test_chordal_mean_of_identical_rotations_is_that_rotation():
    rotation = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
    mean = jitter_stats.chordal_mean_rotation(np.tile(rotation, (5, 1, 1)))
    assert np.allclose(mean, rotation, atol=1e-9)


def test_rotation_jitter_is_zero_for_a_still_estimate():
    rvecs = np.tile([0.1, -0.2, 0.3], (8, 1))
    assert jitter_stats.rotation_jitter_mdeg(rvecs)["rot_jitter_mdeg"] == 0.0


def test_rotation_jitter_recovers_a_known_spread():
    rng = np.random.default_rng(2)
    base = np.array([0.4, -0.3, 1.2])
    sigma_rad = np.radians(0.01)
    rvecs = []
    for _ in range(800):
        perturb = cv2.Rodrigues(rng.normal(0.0, sigma_rad, 3))[0]
        rvecs.append(cv2.Rodrigues(cv2.Rodrigues(base)[0] @ perturb)[0].ravel())
    result = jitter_stats.rotation_jitter_mdeg(np.asarray(rvecs))
    expected_mdeg = 1000.0 * np.degrees(sigma_rad) * np.sqrt(3.0)
    assert np.isclose(result["rot_jitter_mdeg"], expected_mdeg, rtol=0.15)


def test_rotation_jitter_is_immune_to_rotvec_wrap():
    """Near pi the raw rvec can flip sign; the chordal residual must not care."""
    axis = np.array([0.0, 0.0, 1.0])
    rvecs = np.array(
        [(np.pi - 1e-6) * axis, -(np.pi - 1e-6) * axis, (np.pi - 1e-6) * axis]
    )
    result = jitter_stats.rotation_jitter_mdeg(rvecs)
    assert result["rot_jitter_mdeg"] < 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_jitter_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model.jitter_stats'`.

- [ ] **Step 3: Implement**

`jitter_model/jitter_stats.py`:

```python
"""Jitter of a repeated pose estimate, in position and in rotation.

The dome is still inside a burst, so the spread of the estimate across the
burst is the estimator's jitter with nothing else mixed in — no successive
differencing and no mocap residual, unlike the movement takes.
"""

import cv2
import numpy as np


def fixed_point_positions(rvecs, tvecs, fixed_point):
    """Where one fixed board point lands in camera coordinates, per frame.

    Every subset is reported at the same physical point, so jitter numbers
    stay comparable instead of each subset being quoted at its own origin.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    tvecs = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
    fixed_point = np.asarray(fixed_point, dtype=np.float64).reshape(3)
    positions = np.empty_like(tvecs)
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        positions[index] = cv2.Rodrigues(rvec.reshape(3, 1))[0] @ fixed_point + tvec
    return positions


def position_jitter_mm(positions):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(positions) < 2:
        raise ValueError("position jitter needs at least two samples")
    per_axis = 1000.0 * np.std(positions, axis=0, ddof=1)
    return {
        "pos_jitter_mm": float(np.linalg.norm(per_axis)),
        "pos_jitter_x_mm": float(per_axis[0]),
        "pos_jitter_y_mm": float(per_axis[1]),
        "pos_jitter_z_mm": float(per_axis[2]),
    }


def chordal_mean_rotation(rotations):
    """The rotation closest to a set of rotations in the Frobenius sense.

    Averaging rotation vectors component-wise is wrong near the pi wrap; this
    projects the arithmetic mean matrix back onto SO(3), which is not.
    """
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    u, _s, vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    return u @ correction @ vt


def rotation_jitter_mdeg(rvecs):
    """Spread of the rotation estimate, as residuals about the chordal mean.

    Working in the residual keeps every angle small, so nothing wraps and the
    standard deviation means what it looks like it means.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    if len(rvecs) < 2:
        raise ValueError("rotation jitter needs at least two samples")
    rotations = np.asarray([cv2.Rodrigues(r.reshape(3, 1))[0] for r in rvecs])
    mean = chordal_mean_rotation(rotations)
    residuals = np.asarray(
        [cv2.Rodrigues(mean.T @ rotation)[0].ravel() for rotation in rotations]
    )
    per_axis = 1000.0 * np.degrees(np.std(residuals, axis=0, ddof=1))
    return {
        "rot_jitter_mdeg": float(np.linalg.norm(per_axis)),
        "rot_jitter_x_mdeg": float(per_axis[0]),
        "rot_jitter_y_mdeg": float(per_axis[1]),
        "rot_jitter_z_mdeg": float(per_axis[2]),
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_jitter_stats.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/jitter_stats.py tests/jitter_model/test_jitter_stats.py
uv run ruff check jitter_model/ tests/
git add jitter_model/jitter_stats.py tests/jitter_model/test_jitter_stats.py
git commit -m "feat: add position and chordal rotation jitter statistics"
```

---

### Task 9: `_subset_worker.py` — one burst's subsets, solved

The worker is a module of its own precisely so Windows `spawn` re-imports something small instead of re-running a whole analysis script.

**Files:**
- Create: `jitter_model/_subset_worker.py`
- Test: `tests/jitter_model/test_subset_worker.py`

**Interfaces:**
- Consumes: `common.PoseSolver`, `geometry.subset_geometry`, `geometry.pose_geometry`, `jitter_stats.*`.
- Produces:
  - `enumerate_subsets(frames0, frames1, max_tags, presence_fraction, camera_config) -> list[tuple[int, ...]]`
  - `solve_burst(burst, frames0, frames1, solver, *, camera_config, max_tags, presence_fraction, min_frames, fixed_point) -> list[dict]` — one row per subset, with the schema from the spec. `frames1` is `None` for a mono config, and a list with `None` in the gaps for stereo.
  - `init_worker(payload: dict) -> None` setting module globals `_SOLVER`, `_CACHE`, `_OPTIONS`
  - `worker_burst(task: tuple[int, str, list[int]]) -> list[dict]` — `(burst, camera_config, frame_indices)`, using those globals

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_subset_worker.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_subset_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model._subset_worker'`.

- [ ] **Step 3: Implement**

`jitter_model/_subset_worker.py`:

```python
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

        if len(rvecs) < min_frames:
            continue

        rvecs = np.asarray(rvecs)
        tvecs = np.asarray(tvecs)
        positions = jitter_stats.fixed_point_positions(rvecs, tvecs, fixed_point)

        median_index = int(np.argsort(reprojections)[len(reprojections) // 2])
        pose_terms = geometry.pose_geometry(
            solver.rig,
            marker_ids,
            rvecs[median_index],
            tvecs[median_index],
            frames0[median_index],
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
    frames1 = [
        cam1[pairs[i]] if pairs[i] >= 0 else None for i in frame_indices
    ]
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_subset_worker.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/_subset_worker.py tests/jitter_model/test_subset_worker.py
uv run ruff check jitter_model/ tests/
git add jitter_model/_subset_worker.py tests/jitter_model/test_subset_worker.py
git commit -m "feat: add per-burst subset enumeration and solving worker"
```

---

### Task 10: `fixed_effects.py` — within-burst regression

**Files:**
- Create: `jitter_model/fixed_effects.py`
- Test: `tests/jitter_model/test_fixed_effects.py`

**Interfaces:**
- Consumes: numpy only.
- Produces:
  - `within_transform(values, groups, weights=None) -> ndarray` — group-demeaned copy
  - `variance_inflation(design) -> ndarray` — VIF per column
  - `fit_fixed_effects(design, response, groups, weights=None) -> FitResult`
  - `FitResult` frozen dataclass: `coefficients: ndarray`, `standard_errors: ndarray`, `ci_low: ndarray`, `ci_high: ndarray`, `vif: ndarray`, `r_squared: float`, `n_groups: int`, `n_observations: int`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_fixed_effects.py`:

```python
import numpy as np

from jitter_model import fixed_effects


def test_within_transform_removes_group_means():
    values = np.array([1.0, 3.0, 10.0, 14.0])
    groups = np.array([0, 0, 1, 1])
    demeaned = fixed_effects.within_transform(values, groups)
    assert np.allclose(demeaned, [-1.0, 1.0, -2.0, 2.0])


def test_fit_recovers_known_coefficients_despite_group_offsets():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(40), 12)
    offsets = rng.normal(0.0, 5.0, 40)[groups]
    x1 = rng.normal(0.0, 1.0, groups.size)
    x2 = rng.normal(0.0, 1.0, groups.size)
    y = offsets + 0.7 * x1 - 0.3 * x2 + rng.normal(0.0, 0.05, groups.size)
    result = fixed_effects.fit_fixed_effects(np.column_stack([x1, x2]), y, groups)
    assert np.allclose(result.coefficients, [0.7, -0.3], atol=0.02)
    assert result.n_groups == 40


def test_confidence_intervals_bracket_the_truth():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 15)
    x = rng.normal(0.0, 1.0, groups.size)
    y = rng.normal(0.0, 3.0, 30)[groups] + 0.5 * x + rng.normal(0.0, 0.2, groups.size)
    result = fixed_effects.fit_fixed_effects(x.reshape(-1, 1), y, groups)
    assert result.ci_low[0] < 0.5 < result.ci_high[0]


def test_clustered_errors_exceed_naive_errors_under_group_correlation():
    """Residuals correlated inside a group must widen the interval, not narrow it."""
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(25), 20)
    x = rng.normal(0.0, 1.0, groups.size)
    shock = rng.normal(0.0, 1.0, 25)[groups]
    y = 0.4 * x + shock * x + rng.normal(0.0, 0.05, groups.size)
    clustered = fixed_effects.fit_fixed_effects(x.reshape(-1, 1), y, groups)
    residual = y - fixed_effects.within_transform(x, groups) * clustered.coefficients[0]
    naive = np.sqrt(
        np.var(residual, ddof=1)
        / np.sum(fixed_effects.within_transform(x, groups) ** 2)
    )
    assert clustered.standard_errors[0] > naive


def test_variance_inflation_flags_a_collinear_pair():
    rng = np.random.default_rng(7)
    x1 = rng.normal(0.0, 1.0, 300)
    x2 = x1 + rng.normal(0.0, 0.02, 300)
    x3 = rng.normal(0.0, 1.0, 300)
    vif = fixed_effects.variance_inflation(np.column_stack([x1, x2, x3]))
    assert vif[0] > 50.0
    assert vif[1] > 50.0
    assert vif[2] < 2.0


def test_weights_favour_the_precise_observations():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(20), 10)
    x = rng.normal(0.0, 1.0, groups.size)
    y = 1.0 * x
    y[::2] += 5.0  # noisy half
    weights = np.where(np.arange(groups.size) % 2 == 0, 0.01, 100.0)
    result = fixed_effects.fit_fixed_effects(
        x.reshape(-1, 1), y, groups, weights=weights
    )
    assert np.isclose(result.coefficients[0], 1.0, atol=0.1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_fixed_effects.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model.fixed_effects'`.

- [ ] **Step 3: Implement**

`jitter_model/fixed_effects.py`:

```python
"""Within-group least squares with cluster-robust errors.

A per-burst intercept absorbs pose, distance and lighting exactly, so every
geometry coefficient is identified from subsets compared against each other
inside the same burst rather than across bursts that differ in other ways.
Implemented by demeaning rather than by dummy columns, which keeps it to one
`lstsq` and adds no dependency.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    vif: np.ndarray
    r_squared: float
    n_groups: int
    n_observations: int


def within_transform(values, groups, weights=None):
    """Subtract each group's (optionally weighted) mean."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.ones(len(groups)) if weights is None else np.asarray(weights, float)
    out = np.array(values, dtype=np.float64, copy=True)
    for group in np.unique(groups):
        mask = groups == group
        mean = np.average(values[mask], axis=0, weights=weights[mask])
        out[mask] = values[mask] - mean
    return out


def variance_inflation(design):
    """VIF per column: how much collinearity inflates that coefficient's variance."""
    design = np.asarray(design, dtype=np.float64)
    inflation = np.empty(design.shape[1])
    for column in range(design.shape[1]):
        others = np.delete(design, column, axis=1)
        others = np.column_stack([others, np.ones(len(others))])
        target = design[:, column]
        fitted = others @ np.linalg.lstsq(others, target, rcond=None)[0]
        residual_ss = float(np.sum((target - fitted) ** 2))
        total_ss = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
        inflation[column] = np.inf if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
    return inflation


def fit_fixed_effects(design, response, groups, weights=None, confidence=0.95):
    """Weighted within-group least squares with errors clustered by group."""
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.ones(len(response)) if weights is None else np.asarray(weights, float)

    x = within_transform(design, groups, weights)
    y = within_transform(response, groups, weights)
    root = np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(x * root, y * root.ravel(), rcond=None)[0]

    residual = y - x @ coefficients
    bread = np.linalg.pinv((x * weights[:, None]).T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        score = (x[mask] * weights[mask, None]).T @ residual[mask]
        meat += np.outer(score, score)
    n_groups = len(unique_groups)
    # Small-cluster correction, the usual one for a finite number of clusters.
    scale = n_groups / max(n_groups - 1, 1)
    covariance = scale * bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    critical = stats.t.ppf(0.5 + confidence / 2.0, max(n_groups - 1, 1))
    total_ss = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    residual_ss = float(np.sum(weights * residual**2))
    return FitResult(
        coefficients=coefficients,
        standard_errors=standard_errors,
        ci_low=coefficients - critical * standard_errors,
        ci_high=coefficients + critical * standard_errors,
        vif=variance_inflation(design),
        r_squared=1.0 - residual_ss / total_ss if total_ss > 0 else float("nan"),
        n_groups=n_groups,
        n_observations=len(response),
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_fixed_effects.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/fixed_effects.py tests/jitter_model/test_fixed_effects.py
uv run ruff check jitter_model/ tests/
git add jitter_model/fixed_effects.py tests/jitter_model/test_fixed_effects.py
git commit -m "feat: add within-group regression with cluster-robust errors"
```

---

### Task 11: `08_subset_geometry.py` — the sweep, parallelized

**Files:**
- Create: `jitter_model/08_subset_geometry.py`
- Test: `tests/jitter_model/test_subset_sweep_parallel.py`

**Interfaces:**
- Consumes: Tasks 3-10.
- Produces: `data/dome/sep18_26/dome_static_burst_sep18_26/subset_geometry/subset_geometry_cells.csv`, and the module-level function `run_sweep(tasks, payload, workers) -> list[dict]` used by both the script and the parity test.

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_subset_sweep_parallel.py`:

```python
"""Parallel and serial sweeps must agree exactly; parallelism is not a variable."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "jitter_model" / "08_subset_geometry.py"
RECORDING = REPO / "data/dome/sep18_26/dome_static_burst_sep18_26"


def _load_script():
    spec = importlib.util.spec_from_file_location("subset_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.skipif(not RECORDING.exists(), reason="dome recording not present")
def test_parallel_matches_serial_on_three_bursts():
    module = _load_script()
    tasks = module.build_tasks()[:6]  # three bursts x two camera configs
    payload = module.build_payload()
    serial = module.run_sweep(tasks, payload, workers=1)
    parallel = module.run_sweep(tasks, payload, workers=4)
    key = lambda row: (row["burst"], row["camera_config"], row["marker_ids"])
    assert sorted(serial, key=key) == sorted(parallel, key=key)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_subset_sweep_parallel.py -v -m slow`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the script**

`jitter_model/08_subset_geometry.py`, in the repository's `# %%` cell style. Header cells mirror `05_static_jitter.py:40-140` (imports, paths, settings) with these settings:

```python
# %% Paths and experiment settings
RECORDING_DIR = (
    PROJECT_ROOT / "data" / "dome" / "sep18_26" / "dome_static_burst_sep18_26"
)
OUTPUT_SUBDIR = "subset_geometry"
CAMERA_NAMES = ("cam0", "cam1")
CAMERA_CONFIGS = ("cam0", "stereo")
MAX_TAGS = 4
TAG_PRESENCE_FRACTION = 0.9
MIN_FRAMES_PER_BURST = 10
MAX_BURST_MOCAP_TRAVEL_M = 0.002
MAX_PAIR_FRACTION_OF_FRAME = 0.55
# Report every subset at one physical point so the numbers are comparable;
# zero is the reference tag's origin, which is what 05 reports.
P_FIXED = np.zeros(3)
WORKERS = None  # None -> os.cpu_count()
# A predictor above this VIF is collapsed into the composite spread term.
MAX_VIF = 10.0
```

Then the working cells:

```python
# %% Build the work list and the worker payload
def build_tasks():
    """One task per (burst, camera config), carrying that burst's frame indices."""
    return [
        (burst, camera_config, burst_frames[burst].tolist())
        for burst in sorted(static_bursts)
        for camera_config in CAMERA_CONFIGS
    ]


def build_payload():
    return {
        "cache_path": str(DETECTION_CACHE),
        "rigidbody": rigidbody,
        "stereo": stereo,
        "camera_names": list(CAMERA_NAMES),
        "options": {
            "pairs": cam1_pair,
            "max_tags": MAX_TAGS,
            "presence_fraction": TAG_PRESENCE_FRACTION,
            "min_frames": MIN_FRAMES_PER_BURST,
            "fixed_point": P_FIXED.tolist(),
        },
    }


def run_sweep(tasks, payload, workers=None):
    """Solve every task, in a process pool unless `workers` is 1.

    The serial path exists so the parallel result can be checked against it and
    so the script still runs cell-by-cell in a kernel, where a pool would try
    to re-import __main__.
    """
    if workers == 1:
        _subset_worker.init_worker(payload)
        return [
            row
            for task in tqdm(tasks, desc="Solving subsets")
            for row in _subset_worker.worker_burst(task)
        ]

    rows = []
    with ProcessPoolExecutor(
        max_workers=workers or os.cpu_count(),
        initializer=_subset_worker.init_worker,
        initargs=(payload,),
    ) as pool:
        futures = [pool.submit(_subset_worker.worker_burst, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Solving subsets"
        ):
            rows.extend(future.result())
    return rows


# %% Run it
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    cells = run_sweep(build_tasks(), build_payload(), WORKERS)
else:
    cells = run_sweep(build_tasks(), build_payload(), workers=1)

cell_table = pd.DataFrame(cells).sort_values(
    ["camera_config", "burst", "n_tags", "marker_ids"]
).reset_index(drop=True)
cell_table.to_csv(CELLS_CSV, index=False)
print(f"{len(cell_table)} cells -> {CELLS_CSV}")
```

Reuse `05_static_jitter.py`'s burst grouping cell (`05_static_jitter.py:594-612`) and its mocap stillness cell (`05_static_jitter.py:794-932`) verbatim to produce `burst_frames` and `static_bursts`; they read the same recording and the same `bursts.json`.

Imports this script adds beyond `05`'s: `import os`, `import sys`, `from concurrent.futures import ProcessPoolExecutor, as_completed`, `from jitter_model import _subset_worker, common, fixed_effects, geometry, jitter_stats`.

- [ ] **Step 4: Run the sweep end to end**

Run: `uv run python jitter_model/08_subset_geometry.py`
Expected: a progress bar, then a cells CSV of roughly 9000 rows. Sanity-check the print: cell count in the thousands, and no burst with zero subsets.

- [ ] **Step 5: Run the parallel-parity test**

Run: `uv run pytest tests/jitter_model/test_subset_sweep_parallel.py -v -m slow`
Expected: PASS. Any mismatch means worker state leaked between tasks — check that `init_worker` sets all three globals and that nothing mutates `_CACHE`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format jitter_model/08_subset_geometry.py tests/jitter_model/test_subset_sweep_parallel.py
uv run ruff check jitter_model/ tests/
git add jitter_model/08_subset_geometry.py tests/jitter_model/test_subset_sweep_parallel.py
git commit -m "feat: sweep every visible tag subset per burst, in a process pool"
```

---

### Task 12: `08` — the model, the collinearity report, and the figures

**Files:**
- Modify: `jitter_model/08_subset_geometry.py` (append cells)
- Test: `tests/jitter_model/test_model_assembly.py`

**Interfaces:**
- Consumes: `fixed_effects.fit_fixed_effects`, the cells table from Task 11.
- Produces:
  - `build_design(cells, predictors) -> tuple[ndarray, list[str]]` — log-transformed design matrix and its column names
  - `collapse_collinear(design, names, max_vif) -> tuple[ndarray, list[str], list[str]]` — returns the reduced design, its names, and the names folded into `composite_spread`
  - `subset_geometry_model.csv`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_model_assembly.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "jitter_model" / "08_subset_geometry.py"


def _module():
    spec = importlib.util.spec_from_file_location("subset_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_design_log_transforms_positive_predictors():
    module = _module()
    cells = pd.DataFrame(
        {"max_baseline_mm": [10.0, 100.0], "lever_mm": [1.0, 10.0]}
    )
    design, names = module.build_design(cells, ["max_baseline_mm", "lever_mm"])
    assert names == ["log_max_baseline_mm", "log_lever_mm"]
    assert np.allclose(design[:, 0], np.log([10.0, 100.0]))


def test_build_design_is_finite_when_a_predictor_is_zero():
    module = _module()
    cells = pd.DataFrame({"max_baseline_mm": [0.0, 50.0]})
    design, _names = module.build_design(cells, ["max_baseline_mm"])
    assert np.isfinite(design).all()


def test_collapse_collinear_folds_a_redundant_pair():
    module = _module()
    rng = np.random.default_rng(9)
    x1 = rng.normal(0.0, 1.0, 400)
    design = np.column_stack([x1, x1 + rng.normal(0.0, 0.01, 400),
                              rng.normal(0.0, 1.0, 400)])
    reduced, names, folded = module.collapse_collinear(
        design, ["a", "b", "c"], max_vif=10.0
    )
    assert "composite_spread" in names
    assert set(folded) == {"a", "b"}
    assert reduced.shape[1] == 2


def test_collapse_collinear_leaves_independent_predictors_alone():
    module = _module()
    rng = np.random.default_rng(10)
    design = rng.normal(0.0, 1.0, (400, 3))
    reduced, names, folded = module.collapse_collinear(
        design, ["a", "b", "c"], max_vif=10.0
    )
    assert folded == []
    assert names == ["a", "b", "c"]
    assert reduced.shape == design.shape
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_model_assembly.py -v`
Expected: FAIL — `AttributeError: module 'subset_geometry' has no attribute 'build_design'`.

- [ ] **Step 3: Implement the model cells**

Append to `jitter_model/08_subset_geometry.py`:

```python
# %% Design matrix
GEOMETRY_PREDICTORS = [
    "max_baseline_mm",
    "rms_radius_mm",
    "min_singular_mm",
    "max_normal_angle_deg",
    "lever_mm",
    "n_corners",
]
POSE_PREDICTORS = ["distance_m", "mean_incidence_deg", "mean_apparent_size_px"]
# Added to every predictor before the log so a legitimately zero baseline (one
# tag) stays finite instead of becoming -inf and dropping the row.
LOG_FLOOR = 1e-3


def build_design(cells, predictors):
    """Log-transformed predictors, so coefficients read as elasticities."""
    columns = []
    names = []
    for predictor in predictors:
        columns.append(np.log(cells[predictor].to_numpy(dtype=float) + LOG_FLOOR))
        names.append(f"log_{predictor}")
    return np.column_stack(columns), names


def collapse_collinear(design, names, max_vif):
    """Fold predictors that cannot be separated into one composite spread term.

    On a dome, tag separation and orientation spread are the same variable
    (chord = 2 R sin(dtheta/2)), so fitting them as if they were independent
    would hand one of them the other's effect. Rather than silently dropping
    one, the inseparable group is standardised and averaged into a single term
    whose members are named in the output.
    """
    design = np.asarray(design, dtype=np.float64)
    names = list(names)
    inflation = fixed_effects.variance_inflation(design)
    offenders = [i for i, value in enumerate(inflation) if value > max_vif]
    if len(offenders) < 2:
        return design, names, []

    block = design[:, offenders]
    standardised = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
    composite = standardised.mean(axis=1)
    keep = [i for i in range(design.shape[1]) if i not in offenders]
    reduced = np.column_stack([design[:, keep], composite])
    reduced_names = [names[i] for i in keep] + ["composite_spread"]
    return reduced, reduced_names, [names[i] for i in offenders]


# %% Fit, within burst
def fit_response(cells, response_column):
    design, names = build_design(cells, GEOMETRY_PREDICTORS)
    design, names, folded = collapse_collinear(design, names, MAX_VIF)
    result = fixed_effects.fit_fixed_effects(
        design,
        np.log(cells[response_column].to_numpy(dtype=float)),
        cells["burst"].to_numpy(),
        weights=2.0 * (cells["frames"].to_numpy(dtype=float) - 1.0),
    )
    return result, names, folded
```

Then a cell that runs `fit_response` for `pos_jitter_mm` and `rot_jitter_mdeg`, per camera config, writes `subset_geometry_model.csv` with columns `camera_config, response, predictor, coefficient, standard_error, ci_low, ci_high, vif`, and prints:

- the predictor correlation matrix,
- `corr(log_max_baseline_mm, log_max_normal_angle_deg)` over the surviving cells,
- the names folded into `composite_spread`, if any,
- the two caveats verbatim:

```python
print(
    "Caveat: a subset exists only when its tags are visible, and visibility "
    "correlates with incidence angle. Burst fixed effects do not remove this."
)
print(
    "Caveat: lever_mm is constant per subset across bursts and is identified "
    "only relative to P_FIXED. Dropping it would attribute 'far from the "
    "reported point' to 'widely separated'."
)
```

Add the between-burst stage: average the stage-1 residuals per burst, regress those burst means on `log(distance_m)` and `mean_incidence_deg`, append the coefficients to the same CSV with `stage="between"`.

- [ ] **Step 4: Run the tests and the script**

Run: `uv run pytest tests/jitter_model/test_model_assembly.py -v && uv run python jitter_model/08_subset_geometry.py`
Expected: 4 passed, then a model CSV plus the printed correlation matrix and caveats.

- [ ] **Step 5: Add the figures**

Append a figures cell producing, into `OUTPUT_DIR`:

- `subset_jitter_vs_baseline.png` — `pos_jitter_mm` against `max_baseline_mm`, log y, coloured by `n_tags`, one panel per camera config
- `subset_jitter_vs_lever.png` — the same against `lever_mm`
- `subset_predictor_correlations.png` — the correlation matrix as a heatmap with the values annotated
- `subset_partial_residuals.png` — one panel per surviving predictor, partial residual against that predictor
- `subset_position_vs_rotation.png` — `pos_jitter_mm` against `rot_jitter_mdeg`, coloured by `lever_mm`, which is where the lever mechanism shows itself

Follow `05_static_jitter.py`'s figure conventions: `turbo` colormap, `dpi=200`, titles naming the take.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format jitter_model/08_subset_geometry.py tests/jitter_model/test_model_assembly.py
uv run ruff check jitter_model/ tests/
git add jitter_model/08_subset_geometry.py tests/jitter_model/test_model_assembly.py
git commit -m "feat: fit within-burst geometry model and report the collinearity"
```

---

### Task 13: `simulate.py` — the Monte-Carlo engine

**Files:**
- Create: `jitter_model/simulate.py`
- Test: `tests/jitter_model/test_simulate.py`

**Interfaces:**
- Consumes: `common.PoseSolver`, `jitter_stats`.
- Produces:
  - `simulate_detections(solver, marker_ids, rvec, tvec, camera_name, sigma_px, rng) -> dict[int, ndarray(4,2)]`
  - `simulate_jitter(solver, marker_ids, rvec, tvec, *, sigma_px, trials, fixed_point, camera_config="cam0", rng=None) -> dict` with the same `pos_jitter_*` / `rot_jitter_*` keys as `jitter_stats`, plus `trials_solved`
  - `sigma_from_apparent_size(size_px, coefficients) -> ndarray` where `coefficients = (a, b)` and `sigma = a * (size_px / 100.0) ** b`
  - `fit_sigma_model(cells) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_simulate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from jitter_model import common, simulate
from tests.jitter_model.test_common_loading import _rigidbody_dict

RVEC = np.array([0.02, -0.03, 0.01])
TVEC = np.array([0.0, 0.0, 0.6])


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


def test_zero_noise_gives_zero_jitter(solver):
    result = simulate.simulate_jitter(
        solver, (1, 2, 3), RVEC, TVEC,
        sigma_px=0.0, trials=20, fixed_point=np.zeros(3),
        rng=np.random.default_rng(11),
    )
    assert result["pos_jitter_mm"] < 1e-6


def test_jitter_scales_linearly_with_sigma(solver):
    kwargs = dict(
        marker_ids=(1, 2, 3), rvec=RVEC, tvec=TVEC, trials=400,
        fixed_point=np.zeros(3),
    )
    low = simulate.simulate_jitter(
        solver, sigma_px=0.2, rng=np.random.default_rng(12), **kwargs
    )["pos_jitter_mm"]
    high = simulate.simulate_jitter(
        solver, sigma_px=0.4, rng=np.random.default_rng(12), **kwargs
    )["pos_jitter_mm"]
    assert np.isclose(high / low, 2.0, rtol=0.15)


def test_more_tags_reduce_simulated_jitter(solver):
    kwargs = dict(
        rvec=RVEC, tvec=TVEC, sigma_px=0.3, trials=300, fixed_point=np.zeros(3)
    )
    one = simulate.simulate_jitter(
        solver, marker_ids=(1,), rng=np.random.default_rng(13), **kwargs
    )["pos_jitter_mm"]
    four = simulate.simulate_jitter(
        solver, marker_ids=(1, 2, 3, 4), rng=np.random.default_rng(13), **kwargs
    )["pos_jitter_mm"]
    assert four < one


def test_simulation_is_reproducible_for_a_seed(solver):
    kwargs = dict(
        marker_ids=(1, 2), rvec=RVEC, tvec=TVEC, sigma_px=0.3, trials=50,
        fixed_point=np.zeros(3),
    )
    first = simulate.simulate_jitter(solver, rng=np.random.default_rng(14), **kwargs)
    second = simulate.simulate_jitter(solver, rng=np.random.default_rng(14), **kwargs)
    assert first == second


def test_sigma_model_recovers_a_known_power_law():
    sizes = np.array([20.0, 40.0, 80.0, 160.0])
    cells = pd.DataFrame(
        {
            "mean_apparent_size_px": sizes,
            "reprojection_px": 0.5 * (sizes / 100.0) ** -0.5,
            "frames": np.full(4, 50),
        }
    )
    a, b = simulate.fit_sigma_model(cells)
    assert np.isclose(a, 0.5, rtol=0.05)
    assert np.isclose(b, -0.5, rtol=0.05)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_simulate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jitter_model.simulate'`.

- [ ] **Step 3: Implement**

`jitter_model/simulate.py`:

```python
"""Corner-noise Monte Carlo for a tag layout.

Real corners are replaced by projected ones plus Gaussian noise and re-solved
with the same estimators used on real data, so simulated and measured jitter
are directly comparable. This is the only way to vary baseline and curvature
independently: on a real dome they are the same variable.
"""

import cv2
import numpy as np

from jitter_model import jitter_stats


def simulate_detections(solver, marker_ids, rvec, tvec, camera_name, sigma_px, rng):
    camera = solver.cameras[camera_name]
    detections = {}
    for marker_id in marker_ids:
        projected, _ = cv2.fisheye.projectPoints(
            solver.rig.corners_reference[marker_id].reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera.K,
            camera.D,
        )
        corners = projected.reshape(-1, 2)
        if sigma_px > 0:
            corners = corners + rng.normal(0.0, sigma_px, corners.shape)
        detections[marker_id] = corners
    return detections


def simulate_jitter(
    solver,
    marker_ids,
    rvec,
    tvec,
    *,
    sigma_px,
    trials,
    fixed_point,
    camera_config="cam0",
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    rvecs = []
    tvecs = []
    for _ in range(trials):
        frame0 = simulate_detections(
            solver, marker_ids, rvec, tvec, "cam0", sigma_px, rng
        )
        if camera_config == "stereo":
            rvec1, tvec1 = cv2.composeRT(
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                solver._stereo_rvec,
                solver._stereo_translation,
            )[:2]
            frame1 = simulate_detections(
                solver, marker_ids, rvec1, tvec1, "cam1", sigma_px, rng
            )
            pose = solver.stereo_board_pose(frame0, frame1, marker_ids)
        else:
            pose = solver.mono_board_pose(frame0, marker_ids, camera_config)
        if pose is None:
            continue
        rvecs.append(pose["rvec"])
        tvecs.append(pose["tvec"])

    if len(rvecs) < 2:
        return {"trials_solved": len(rvecs), "pos_jitter_mm": float("nan")}

    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)
    positions = jitter_stats.fixed_point_positions(rvecs, tvecs, fixed_point)
    result = {"trials_solved": len(rvecs)}
    result.update(jitter_stats.position_jitter_mm(positions))
    result.update(jitter_stats.rotation_jitter_mdeg(rvecs))
    return result


def sigma_from_apparent_size(size_px, coefficients):
    """Corner noise as a power law in apparent tag size.

    A single global sigma would understate the near bursts and overstate the
    far ones, because corner localisation degrades as the tag shrinks.
    """
    a, b = coefficients
    return a * (np.asarray(size_px, dtype=float) / 100.0) ** b


def fit_sigma_model(cells):
    """Fit (a, b) of the power law to measured reprojection RMSE per cell."""
    size = cells["mean_apparent_size_px"].to_numpy(dtype=float)
    sigma = cells["reprojection_px"].to_numpy(dtype=float)
    finite = np.isfinite(size) & np.isfinite(sigma) & (size > 0) & (sigma > 0)
    design = np.column_stack(
        [np.ones(finite.sum()), np.log(size[finite] / 100.0)]
    )
    solution = np.linalg.lstsq(design, np.log(sigma[finite]), rcond=None)[0]
    return float(np.exp(solution[0])), float(solution[1])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_simulate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/simulate.py tests/jitter_model/test_simulate.py
uv run ruff check jitter_model/ tests/
git add jitter_model/simulate.py tests/jitter_model/test_simulate.py
git commit -m "feat: add corner-noise Monte Carlo engine for tag layouts"
```

---

### Task 14: Validation gate

**Files:**
- Modify: `jitter_model/simulate.py`
- Test: `tests/jitter_model/test_validation_gate.py`

**Interfaces:**
- Consumes: `simulate_jitter`, `fit_sigma_model`.
- Produces: `validate_against_measured(simulated, measured, *, slope_tolerance=0.15, max_scatter=0.35) -> dict` with keys `slope`, `intercept`, `log_scatter`, `n`, `passed`.

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_validation_gate.py`:

```python
import numpy as np

from jitter_model import simulate


def test_gate_passes_when_simulation_matches_measurement():
    rng = np.random.default_rng(15)
    measured = np.exp(rng.normal(0.0, 0.8, 300))
    simulated = measured * np.exp(rng.normal(0.0, 0.10, 300))
    report = simulate.validate_against_measured(simulated, measured)
    assert report["passed"]
    assert np.isclose(report["slope"], 1.0, atol=0.15)


def test_gate_fails_on_a_systematic_slope_error():
    rng = np.random.default_rng(16)
    measured = np.exp(rng.normal(0.0, 0.8, 300))
    simulated = measured**0.5
    report = simulate.validate_against_measured(simulated, measured)
    assert not report["passed"]


def test_gate_fails_on_excess_scatter():
    rng = np.random.default_rng(17)
    measured = np.exp(rng.normal(0.0, 0.8, 300))
    simulated = measured * np.exp(rng.normal(0.0, 1.2, 300))
    report = simulate.validate_against_measured(simulated, measured)
    assert not report["passed"]


def test_gate_ignores_non_finite_pairs():
    measured = np.array([1.0, 2.0, np.nan, 4.0])
    simulated = np.array([1.0, 2.0, 3.0, np.inf])
    report = simulate.validate_against_measured(simulated, measured)
    assert report["n"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_validation_gate.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'validate_against_measured'`.

- [ ] **Step 3: Implement**

Append to `jitter_model/simulate.py`:

```python
def validate_against_measured(
    simulated, measured, *, slope_tolerance=0.15, max_scatter=0.35
):
    """Whether the simulator reproduces the real measurements well enough to trust.

    A slope away from 1 in log-log means the simulator scales wrongly with
    geometry, not merely that sigma is off; scatter beyond the ~10% sampling
    noise of a 50-frame standard deviation means something other than corner
    noise is driving the real jitter. Either way the simulator does not get
    used as a design tool.
    """
    simulated = np.asarray(simulated, dtype=float)
    measured = np.asarray(measured, dtype=float)
    finite = (
        np.isfinite(simulated) & np.isfinite(measured)
        & (simulated > 0) & (measured > 0)
    )
    x = np.log(measured[finite])
    y = np.log(simulated[finite])
    if len(x) < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "log_scatter": float("nan"),
            "n": int(len(x)),
            "passed": False,
        }
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    scatter = float(np.std(y - design @ np.array([intercept, slope]), ddof=2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "log_scatter": scatter,
        "n": int(len(x)),
        "passed": bool(
            abs(slope - 1.0) <= slope_tolerance and scatter <= max_scatter
        ),
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_validation_gate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format jitter_model/simulate.py tests/jitter_model/test_validation_gate.py
uv run ruff check jitter_model/ tests/
git add jitter_model/simulate.py tests/jitter_model/test_validation_gate.py
git commit -m "feat: add the simulator validation gate against measured jitter"
```

---

### Task 15: `09_layout_simulator.py` — validation run and design sweep

**Files:**
- Create: `jitter_model/09_layout_simulator.py`
- Test: `tests/jitter_model/test_layout_sweep.py`

**Interfaces:**
- Consumes: Tasks 3, 8, 13, 14, plus `subset_geometry_cells.csv` from Task 11.
- Produces:
  - `sphere_layout(n_tags, baseline_mm, radius_mm, tag_size_m) -> dict[int, dict]` in the `synthetic_rig_spec` shape
  - `simulated_validation.csv`, `simulated_layout_sweep.csv`, and figures, all prefixed `simulated_`

- [ ] **Step 1: Write the failing test**

`tests/jitter_model/test_layout_sweep.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "jitter_model" / "09_layout_simulator.py"


def _module():
    spec = importlib.util.spec_from_file_location("layout_simulator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sphere_layout_hits_the_requested_baseline():
    module = _module()
    layout = module.sphere_layout(2, baseline_mm=120.0, radius_mm=115.0,
                                  tag_size_m=0.05)
    centres = np.asarray([tag["t"] for tag in layout.values()])
    assert np.isclose(1000.0 * np.linalg.norm(centres[0] - centres[1]), 120.0,
                      atol=1e-6)


def test_a_flatter_sphere_gives_a_smaller_normal_angle():
    module = _module()
    curved = module.sphere_layout(2, 120.0, 115.0, 0.05)
    flat = module.sphere_layout(2, 120.0, 5000.0, 0.05)

    def angle(layout):
        normals = [tag["R"][:, 2] for tag in layout.values()]
        return np.degrees(np.arccos(np.clip(normals[0] @ normals[1], -1.0, 1.0)))

    assert angle(flat) < angle(curved)


def test_baseline_is_held_while_curvature_varies():
    """The whole point of the sweep: one variable moves at a time."""
    module = _module()
    for radius in (115.0, 250.0, 1000.0):
        layout = module.sphere_layout(2, 100.0, radius, 0.05)
        centres = np.asarray([tag["t"] for tag in layout.values()])
        assert np.isclose(1000.0 * np.linalg.norm(centres[0] - centres[1]), 100.0,
                          atol=1e-6)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/jitter_model/test_layout_sweep.py -v`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the script**

`jitter_model/09_layout_simulator.py`, same cell style and the same header/paths pattern as `08`, with:

```python
# %% Settings
CELLS_CSV = RECORDING_DIR / "subset_geometry" / "subset_geometry_cells.csv"
OUTPUT_SUBDIR = "layout_simulation"
TRIALS = 400
VALIDATION_SAMPLE = 600  # cells replayed through the simulator
RADII_MM = (115.0, 160.0, 250.0, 500.0, 5000.0)
BASELINES_MM = (60.0, 90.0, 120.0, 160.0, 200.0)
DISTANCES_M = (0.3, 0.5, 0.7, 0.9)
TAG_COUNTS = (1, 2, 3, 4)
P_FIXED = np.zeros(3)
```

```python
# %% Synthetic layouts
def sphere_layout(n_tags, baseline_mm, radius_mm, tag_size_m):
    """`n_tags` evenly spaced on a spherical cap, at an exact chord baseline.

    Baseline is the controlled variable and curvature is swept around it, which
    is the construction the real dome cannot provide: there, chord and facet
    tilt move together and the two cannot be told apart.
    """
    radius = radius_mm / 1000.0
    baseline = baseline_mm / 1000.0
    if n_tags == 1:
        return {1: {"R": np.eye(3), "t": np.zeros(3)}}

    # Place tags on a ring whose chord between neighbours is exactly `baseline`
    # for two tags, and whose diameter gives that chord for more.
    ring_radius = (
        baseline / 2.0
        if n_tags == 2
        else baseline / (2.0 * np.sin(np.pi / n_tags))
    )
    centre = np.array([0.0, 0.0, -radius])
    layout = {}
    for index in range(n_tags):
        azimuth = 2.0 * np.pi * index / n_tags
        # Tilt needed to walk `ring_radius` across the sphere's surface.
        tilt = 2.0 * np.arcsin(np.clip(ring_radius / (2.0 * radius), -1.0, 1.0)) * 2.0
        axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0]) * tilt
        rotation = cv2.Rodrigues(axis)[0]
        layout[index + 1] = {
            "R": rotation,
            "t": centre + rotation @ np.array([0.0, 0.0, radius]),
        }
    return layout
```

Verify in Step 4 that the produced chord matches `baseline_mm`; if the closed form above is off for `n_tags > 2`, solve `tilt` numerically with `scipy.optimize.brentq` on `chord(tilt) - baseline` rather than adjusting the test.

Then the working cells:

1. **Validation.** Load `subset_geometry_cells.csv`, fit `(a, b)` with `fit_sigma_model`, take a `VALIDATION_SAMPLE` stratified over `n_tags` and `distance_m`, and for each cell rebuild the real rig, simulate at that cell's own pose with `sigma_from_apparent_size(cell.mean_apparent_size_px, (a, b))`, and collect simulated vs measured. Write `simulated_validation.csv`, plot `simulated_validation.png` log-log with the fitted line, and print the report.

2. **The gate.**

```python
report = simulate.validate_against_measured(
    validation["simulated_pos_jitter_mm"], validation["pos_jitter_mm"]
)
print(
    f"Validation: slope {report['slope']:.3f}, log scatter "
    f"{report['log_scatter']:.3f} over {report['n']} cells"
)
if not report["passed"]:
    raise SystemExit(
        "Simulator did not reproduce the measured jitter, so it is not used as "
        "a design tool. A slope away from 1 or scatter beyond the sampling "
        "noise means corner noise is not the dominant error source here — "
        "calibration error or detector bias carries the rest."
    )
```

3. **The sweep**, only reached when the gate passes: cross `TAG_COUNTS` x `BASELINES_MM` x `RADII_MM` x `DISTANCES_M`, build each layout with `sphere_layout`, wrap it in a `TagRig` via `common.build_tag_rig`, simulate, and write `simulated_layout_sweep.csv`. Figures: `simulated_jitter_vs_curvature.png` (jitter against radius at fixed baseline, one line per baseline) and `simulated_jitter_vs_baseline.png` (the converse), both faceted by tag count, every title prefixed "Simulated".

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/jitter_model/test_layout_sweep.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the script**

Run: `uv run python jitter_model/09_layout_simulator.py`
Expected: either the validation report followed by the sweep CSVs and figures, or the `SystemExit` with its message. **Both are acceptable outcomes** — report which one happened and the slope/scatter numbers.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format jitter_model/09_layout_simulator.py tests/jitter_model/test_layout_sweep.py
uv run ruff check jitter_model/ tests/
git add jitter_model/09_layout_simulator.py tests/jitter_model/test_layout_sweep.py
git commit -m "feat: validate the layout simulator and sweep baseline against curvature"
```

---

### Task 16: Findings write-up

**Files:**
- Create: `jitter_model/subset_geometry_findings.md`
- Modify: `CLAUDE.md` (jitter_model section)

**Interfaces:**
- Consumes: every output of Tasks 11-15.
- Produces: documentation only.

- [ ] **Step 1: Write the findings document**

`jitter_model/subset_geometry_findings.md`, following the style of `jitter_model/jitter_metrics.md`, containing:

- what was measured and on which take;
- the fitted coefficients with their cluster-robust CIs, for position and rotation, per camera config;
- the measured baseline/normal-angle correlation and what was collapsed into `composite_spread`;
- the lever-arm result: how much of the apparent baseline effect is lever;
- the validation outcome with slope and scatter;
- the design recommendation if the gate passed, or the reason it did not;
- the two caveats printed by `08`, restated.

Fill in the actual numbers produced by the runs. If a number is not available because the gate failed, say that explicitly rather than leaving a gap.

- [ ] **Step 2: Update `CLAUDE.md`**

Add `08_subset_geometry.py` and `09_layout_simulator.py` to the project documentation alongside the existing jitter scripts, with their one-line purposes and run commands.

- [ ] **Step 3: Run the full suite one last time**

Run: `uv run pytest tests/ -v && uv run pytest tests/ -v -m slow`
Expected: everything passes, including both parity gates.

- [ ] **Step 4: Commit**

```bash
git add jitter_model/subset_geometry_findings.md CLAUDE.md
git commit -m "docs: record the subset-geometry jitter findings"
```

---

## Notes for the implementer

- **The parity gate is not negotiable.** If `05` or `06` changes by one digit after the `common.py` extraction, something moved that should not have. Find it; do not regenerate the baseline.
- **`solver._stereo_rvec` and `solver._stereo_translation` are private** but are used by `simulate.py`. That is deliberate — the simulator is part of the same subsystem. Do not widen them into the public API for other callers.
- **When a test fails, read what it is asserting before changing it.** Several tests encode physics (jitter falls with more tags, rises linearly with sigma, is immune to rotvec wrap). A failure there is a bug in the implementation, not an over-strict test.
- **`sphere_layout` for more than two tags** is the one place where the closed form may need replacing with a numeric solve. The test pins the requirement — an exact chord — not the method.
