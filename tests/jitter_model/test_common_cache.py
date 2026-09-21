import pickle

import numpy as np

from jitter_model import common


def _cache(tmp_path, marker_ids=(1, 2), tag_size_m=0.05):
    payload = {
        "version": 1,
        "recording_dir": str(tmp_path.resolve()),
        "marker_ids": marker_ids,
        "tag_size_m": tag_size_m,
        "cameras": {
            "cam0": {"detections": [], "metadata": {}},
            "cam1": {"detections": [], "metadata": {}},
        },
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
    # cam0[0]=0 is too far from cam1[0]=29.7ms (gap > 18.15ms)
    # but cam0[1]=33ms pairs with cam1[0]=29.7ms (gap 3.3ms < 18.15ms)
    # and cam0[2]=66ms pairs with cam1[1]=62.7ms (gap 3.3ms < 18.15ms)
    assert pairs.tolist() == [-1, 0, 1]
