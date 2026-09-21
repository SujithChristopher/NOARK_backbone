"""Parallel and serial sweeps must agree exactly; parallelism is not a variable."""

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "jitter_model" / "09_subset_geometry.py"
RECORDING = REPO / "data/dome/sep18_26/dome_static_burst_sep18_26"


def _load_script():
    spec = importlib.util.spec_from_file_location("subset_geometry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_rows_match(rows_a, rows_b):
    """Every field of every row must agree, matched by (burst, camera_config, marker_ids).

    A plain `sorted(...) == sorted(...)` -- as briefed -- would fail on any
    genuinely-identical pair of runs here, because single-tag subsets carry a
    `nan` `max_normal_angle_deg`/`mean_normal_angle_deg` by design (decision 4;
    `geometry.subset_geometry`), and `nan != nan` even when both sides agree.
    `tests/jitter_model/test_subset_worker.py::_assert_rows_match` establishes
    the same NaN-aware comparison for the same reason; this mirrors it.
    """
    key = lambda row: (row["burst"], row["camera_config"], row["marker_ids"])
    by_key_a = {key(row): row for row in rows_a}
    by_key_b = {key(row): row for row in rows_b}
    assert set(by_key_a) == set(by_key_b)
    assert len(rows_a) == len(rows_b)
    for row_key, row_a in by_key_a.items():
        row_b = by_key_b[row_key]
        assert set(row_a) == set(row_b), row_key
        for field, value_a in row_a.items():
            value_b = row_b[field]
            if isinstance(value_a, float):
                if math.isnan(value_a) or math.isnan(value_b):
                    assert math.isnan(value_a) and math.isnan(value_b), (
                        row_key,
                        field,
                        value_a,
                        value_b,
                    )
                else:
                    assert math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-9), (
                        row_key,
                        field,
                        value_a,
                        value_b,
                    )
            else:
                assert value_a == value_b, (row_key, field, value_a, value_b)


@pytest.mark.slow
@pytest.mark.skipif(not RECORDING.exists(), reason="dome recording not present")
def test_parallel_matches_serial_on_three_bursts():
    module = _load_script()
    tasks = module.build_tasks()[:6]  # three bursts x two camera configs
    payload = module.build_payload()
    serial = module.run_sweep(tasks, payload, workers=1)
    parallel = module.run_sweep(tasks, payload, workers=4)
    _assert_rows_match(serial, parallel)
