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
STATIC_OUT = REPO / "data/dome/sep18_26/dome_static_burst_sep18_26/static_jitter"
MOVEMENT_OUT = REPO / "data/dome/sep18_26/dome_random_movement_sep18_26/movement_error"


def _run(script):
    result = subprocess.run(
        [sys.executable, str(REPO / "jitter_model" / script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
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
