"""
check_sampling_rate.py
======================
Finds the true ADC sampling rate from a loadcell CSV.

USB serial batches samples into bursts, so the naive (total rows / total time)
underestimates the real rate.  This script measures the rate *inside* bursts
by discarding inter-burst gaps (intervals > GAP_THRESHOLD_MS).
"""

import os
import numpy as np
import pandas as pd

CSV_PATH         = "/home/sujith/Documents/NOARK_backbone/csv_data/day1/loadcell.csv"
GAP_THRESHOLD_MS = 20.0   # intervals longer than this are USB gaps, not ADC gaps


def analyse(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"[error] file not found: {path}")
        return None

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 10:
        print(f"[error] too few rows ({len(df)}) in {path}")
        return None

    dt_ms = df["timestamp"].diff().dt.total_seconds().dropna() * 1000  # ms

    burst_dt = dt_ms[dt_ms < GAP_THRESHOLD_MS]
    n_gaps   = (dt_ms >= GAP_THRESHOLD_MS).sum()

    total_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()

    return {
        "rows"       : len(df),
        "duration_s" : round(total_s, 1),
        "mean_hz"    : round(len(df) / total_s, 1),        # naive (underestimates if gaps)
        "median_hz"  : round(1000 / dt_ms.median(), 1),
        "burst_hz"   : round(1000 / burst_dt.median(), 1), # true ADC rate inside bursts
        "n_usb_gaps" : int(n_gaps),
        "max_gap_ms" : round(dt_ms.max(), 1),
        "min_dt_ms"  : round(dt_ms.min(), 3),
    }


def main():
    r = analyse(CSV_PATH)
    if r is None:
        return

    print(f"\nFile        : {CSV_PATH}")
    print(f"Rows        : {r['rows']}")
    print(f"Duration    : {r['duration_s']} s")
    print(f"Mean Hz     : {r['mean_hz']}  (naive — affected by USB gaps)")
    print(f"Median Hz   : {r['median_hz']}")
    print(f"Burst Hz    : {r['burst_hz']}  ← true ADC rate between gaps")
    print(f"USB gaps    : {r['n_usb_gaps']}  (intervals ≥ {GAP_THRESHOLD_MS} ms)")
    print(f"Max gap     : {r['max_gap_ms']} ms")
    print(f"Min interval: {r['min_dt_ms']} ms")


if __name__ == "__main__":
    main()
