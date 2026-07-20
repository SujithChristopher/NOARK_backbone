"""Benchmark depth pipeline stages for one frame.

Stages timed:
  left_match   — StereoSGBM left disparity
  right_match  — right disparity (for WLS)
  wls_filter   — Weighted Least Squares filter
  ema_update   — temporal EMA smoother
  total        — end-to-end (left+right+wls+ema)

Usage:
  python bench_depth.py          # uses default data paths from upperlimb_3d_analysis
  python bench_depth.py -n 50    # number of frames to average over
"""

import argparse
import time
import numpy as np
import cv2

# Reuse paths and calibration from the analysis script
from upperlimb_3d_analysis import (
    CALIB_TOML, CAM0_VIDEO, CAM1_VIDEO, CAM_SIZE,
    NUM_DISP, BLOCK_SIZE, WLS_LAMBDA, WLS_SIGMA, DEPTH_EMA_ALPHA,
    load_calib, build_rectify_maps, rectify, load_all_frames,
    TemporalDepthSmoother,
)


def build_matchers():
    left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISP,
        blockSize=BLOCK_SIZE,
        P1=8  * 3 * BLOCK_SIZE ** 2,
        P2=32 * 3 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right = cv2.ximgproc.createRightMatcher(left)
    wls   = cv2.ximgproc.createDisparityWLSFilter(left)
    wls.setLambda(WLS_LAMBDA)
    wls.setSigmaColor(WLS_SIGMA)
    return left, right, wls


def bench(n_frames):
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    maps0, maps1, _ = build_rectify_maps(K0, D0, K1, D1, R, T, CAM_SIZE)

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1), n_frames)
    print(f"  benchmarking {n} frames at {CAM_SIZE[0]}x{CAM_SIZE[1]}")

    left_m, right_m, wls_f = build_matchers()
    smoother = TemporalDepthSmoother()

    times = {k: [] for k in ("left_match", "right_match", "wls_filter", "ema_update", "total")}

    for i in range(n):
        f0 = frames0[i]
        f1 = frames1[i]
        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1
        r0_gray = cv2.cvtColor(rectify(f0_bgr, maps0), cv2.COLOR_BGR2GRAY)
        r1_gray = cv2.cvtColor(rectify(f1_bgr, maps1), cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()

        t_lm = time.perf_counter()
        disp_left  = left_m.compute(r0_gray, r1_gray)
        times["left_match"].append((time.perf_counter() - t_lm) * 1000)

        t_rm = time.perf_counter()
        disp_right = right_m.compute(r1_gray, r0_gray)
        times["right_match"].append((time.perf_counter() - t_rm) * 1000)

        t_wls = time.perf_counter()
        disp_wls = wls_f.filter(disp_left, r0_gray, disparity_map_right=disp_right)
        disp = disp_wls.astype(np.float32) / 16.0
        disp[disp <= 0] = 0.0
        times["wls_filter"].append((time.perf_counter() - t_wls) * 1000)

        t_ema = time.perf_counter()
        _ = smoother.update(disp)
        times["ema_update"].append((time.perf_counter() - t_ema) * 1000)

        times["total"].append((time.perf_counter() - t0) * 1000)

        if i % 10 == 0:
            print(f"  frame {i:3d}/{n}  total {times['total'][-1]:.1f} ms")

    print()
    print(f"{'stage':<14} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  (ms)")
    print("-" * 54)
    for stage, vals in times.items():
        a = np.array(vals[1:])  # skip first frame (JIT warm-up)
        print(f"{stage:<14} {a.mean():>8.1f} {a.std():>8.1f} {a.min():>8.1f} {a.max():>8.1f}")

    total = np.array(times["total"][1:])
    print()
    print(f"Effective throughput: {1000/total.mean():.1f} fps  (depth pipeline only)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--frames", type=int, default=30, help="frames to benchmark")
    args = p.parse_args()
    bench(args.frames)
