"""
check_sampling_rate.py
======================
Finds the true ADC sampling rate from loadcell CSVs.

USB serial batches samples into bursts, so the naive (total rows / total time)
underestimates the real rate.  This script measures the rate *inside* bursts
by discarding inter-burst gaps (intervals > GAP_THRESHOLD_MS).
"""

import os
import numpy as np
import pandas as pd

CSV_ROOT       = "/home/sujith/Documents/NOARK_backbone/csv_data"
GAP_THRESHOLD_MS = 20.0   # intervals longer than this are USB gaps, not ADC gaps


def analyse_session(session_dir: str) -> dict | None:
    path = os.path.join(session_dir, "loadcell.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 10:
        return None

    dt_ms = df["timestamp"].diff().dt.total_seconds().dropna() * 1000  # ms

    # ── burst-rate: intervals that are actual ADC ticks (not USB gaps)
    burst_dt = dt_ms[dt_ms < GAP_THRESHOLD_MS]
    n_gaps   = (dt_ms >= GAP_THRESHOLD_MS).sum()

    total_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()

    return {
        "rows"         : len(df),
        "duration_s"   : round(total_s, 1),
        "mean_hz"      : round(len(df) / total_s, 1),          # naive (wrong)
        "median_hz"    : round(1000 / dt_ms.median(), 1),
        "burst_hz"     : round(1000 / burst_dt.median(), 1),   # true ADC rate
        "n_usb_gaps"   : int(n_gaps),
        "max_gap_ms"   : round(dt_ms.max(), 1),
        "min_dt_ms"    : round(dt_ms.min(), 3),
    }


def main():
    sessions = sorted([
        d for d in os.listdir(CSV_ROOT)
        if os.path.isdir(os.path.join(CSV_ROOT, d))
    ])

    print(f"{'Session':<30} {'Rows':>6} {'Dur(s)':>7} {'Mean Hz':>8} "
          f"{'Median Hz':>10} {'Burst Hz':>9} {'USB gaps':>9} {'MaxGap(ms)':>11}")
    print("-" * 100)

    for sess in sessions:
        r = analyse_session(os.path.join(CSV_ROOT, sess))
        if r is None:
            print(f"{sess:<30}  (no loadcell.csv)")
            continue
        print(f"{sess:<30} {r['rows']:>6} {r['duration_s']:>7} "
              f"{r['mean_hz']:>8} {r['median_hz']:>10} {r['burst_hz']:>9} "
              f"{r['n_usb_gaps']:>9} {r['max_gap_ms']:>11}")


if __name__ == "__main__":
    main()
