import pickle
import warnings

import numpy as np
import pytest

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


def test_load_detection_cache_warns_when_stale(tmp_path):
    """An existing but incompatible cache warns so a silent invalidation is visible."""
    path = _cache(tmp_path)
    with pytest.warns(UserWarning, match="stale"):
        cache = common.load_detection_cache(
            path,
            marker_ids=(1, 2),
            tag_size_m=0.06,  # deliberately mismatched, as in the stale test above
            recording_dir=tmp_path,
            camera_names=("cam0", "cam1"),
        )
    assert cache is None


def test_load_detection_cache_does_not_warn_when_missing(tmp_path):
    """A first run with no cache file yet is normal and must stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache = common.load_detection_cache(
            tmp_path / "absent.pkl",
            marker_ids=(1,),
            tag_size_m=0.05,
            recording_dir=tmp_path,
            camera_names=("cam0",),
        )
    assert cache is None


def test_pair_cameras_matches_the_nearest_frame():
    period = 33_000_000
    cam0 = np.arange(5, dtype=np.int64) * period
    cam1 = cam0 + 2_000_000
    pairs = common.pair_cameras(cam0, cam1, 0.55)
    assert pairs.tolist() == [0, 1, 2, 3, 4]


def test_pair_cameras_matches_the_nearest_frame_even_when_it_is_earlier():
    # cam1 is offset by 0.9 of a frame period from cam0. With period=33ms and
    # tolerance=0.55*33ms=18.15ms, each cam0 frame pairs with its nearest cam1
    # frame within tolerance, even when that frame is EARLIER in time.
    # cam0[0]=0ms has no match (nearest cam1[0]=29.7ms is 29.7ms away > 18.15ms).
    # cam0[1]=33ms pairs with cam1[0]=29.7ms (3.3ms earlier, within tolerance).
    # cam0[2]=66ms pairs with cam1[1]=62.7ms (3.3ms earlier, within tolerance).
    period = 33_000_000
    cam0 = np.arange(3, dtype=np.int64) * period
    cam1 = cam0 + int(0.9 * period)
    pairs = common.pair_cameras(cam0, cam1, 0.55)
    assert pairs.tolist() == [-1, 0, 1]


def test_pair_cameras_rejects_when_nothing_is_close_enough():
    # When cam1 is offset by many frame periods, no cam0 frame has a candidate
    # within tolerance. With period=33ms and tolerance=0.55*33ms=18.15ms,
    # an offset of 100*period leaves every gap >> 18.15ms.
    period = 33_000_000
    cam0 = np.arange(3, dtype=np.int64) * period
    cam1 = cam0 + 100 * period  # Offset by 3300ms, far beyond tolerance
    pairs = common.pair_cameras(cam0, cam1, 0.55)
    assert (pairs == -1).all()
