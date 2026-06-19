"""
check_loadcell_rate.py
======================
Live sampling-rate monitor for the Seeeduino load cell.

Connects via SeeduinoPort, counts samples in rolling windows, and
prints burst-aware rate estimates every REPORT_INTERVAL seconds.

Usage:
    python check_loadcell_rate.py              # auto-detect port
    python check_loadcell_rate.py /dev/ttyACM1 # explicit port
"""

import sys
import time
import signal
import threading
import collections

from teensy_seed_fv import SeeduinoPort

# ── tuning ────────────────────────────────────────────────────────────────────
REPORT_INTERVAL  = 2.0   # seconds between rate reports
WINDOW_SIZE      = 500   # rolling window of inter-sample intervals (samples)
GAP_THRESHOLD_MS = 20.0  # USB burst gap threshold (ms), same as offline analyser
# ─────────────────────────────────────────────────────────────────────────────

_lock      = threading.Lock()
_intervals = collections.deque(maxlen=WINDOW_SIZE)  # ms between consecutive samples
_count     = 0
_t_last    = None
_running   = True


def _on_sample(fx, fy):
    global _count, _t_last
    now = time.perf_counter()
    with _lock:
        if _t_last is not None:
            dt_ms = (now - _t_last) * 1000.0
            _intervals.append(dt_ms)
        _t_last = now
        _count += 1


def _report_loop():
    global _running
    prev_count = 0
    prev_time  = time.perf_counter()

    while _running:
        time.sleep(REPORT_INTERVAL)
        now = time.perf_counter()

        with _lock:
            cur_count = _count
            ivs       = list(_intervals)

        elapsed   = now - prev_time
        window_hz = (cur_count - prev_count) / elapsed if elapsed > 0 else 0.0

        if ivs:
            import statistics
            median_dt = statistics.median(ivs)
            burst_ivs = [v for v in ivs if v < GAP_THRESHOLD_MS]
            usb_gaps  = sum(1 for v in ivs if v >= GAP_THRESHOLD_MS)

            median_hz = 1000.0 / median_dt if median_dt > 0 else 0.0
            burst_hz  = (1000.0 / statistics.median(burst_ivs)
                         if burst_ivs else float("nan"))
        else:
            median_hz = burst_hz = usb_gaps = float("nan")

        print(
            f"  samples={cur_count:6d}"
            f"  window={window_hz:6.1f} Hz"
            f"  median={median_hz:6.1f} Hz"
            f"  burst={burst_hz:6.1f} Hz  (USB gaps: {usb_gaps})"
        )

        prev_count = cur_count
        prev_time  = now


def main():
    global _running

    port = sys.argv[1] if len(sys.argv) > 1 else None

    seeed = SeeduinoPort(port=port)
    seeed.on_update = _on_sample

    def _stop(sig=None, frame=None):
        global _running
        _running = False
        seeed.stop()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Connecting to Seeeduino (port={port or 'auto'}) …")
    seeed.start()
    print("Logging — press Ctrl-C to stop.\n")
    print(f"  {'samples':>8}  {'window Hz':>10}  {'median Hz':>10}  {'burst Hz':>9}  USB gaps")

    reporter = threading.Thread(target=_report_loop, daemon=True)
    reporter.start()

    while _running:
        time.sleep(0.1)

    print("\nDone.")


if __name__ == "__main__":
    main()