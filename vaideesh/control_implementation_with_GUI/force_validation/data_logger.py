"""
data_logger.py  –  NOARK multi-stream CSV logger
=================================================
Four files, each logged at its own natural sampling rate:

    camera_YYYYMMDD_HHMMSS.csv   → timestamp, pos_x, pos_z
    encoder_YYYYMMDD_HHMMSS.csv  → timestamp, enc1, enc2
    loadcell_YYYYMMDD_HHMMSS.csv → timestamp, Fx, Fy
    gui_YYYYMMDD_HHMMSS.csv      → timestamp, magnitude, direction,
                                    T1_left, T3_right, tau1, tau2

Timestamps are time.time() 6-decimal float (Unix seconds).

Usage
-----
    logger = DataLogger(session_dir="logs")
    logger.start()

    # from camera thread:
    logger.log_camera(pos_x, pos_z)

    # from teensy thread:
    logger.log_encoder(enc1, enc2)

    # from seeeduino thread (every new parse):
    logger.log_loadcell(fx, fy)

    # from GUI _refresh (every timer tick):
    logger.log_gui(magnitude, direction, T1, T3, tau1, tau2)

    logger.stop()
"""

import csv
import os
import queue
import threading
from datetime import datetime


# ── one background writer per CSV stream ─────────────────────────────────────

class _StreamWriter:
    """
    Owns one CSV file.  log() is non-blocking (puts to a queue).
    A daemon thread drains the queue and writes rows.
    """
    def __init__(self, filepath: str, header: list[str]):
        self._q: queue.Queue = queue.Queue()
        self._fh = open(filepath, "w", newline="", buffering=1)   # line-buffered
        self._writer = csv.writer(self._fh)
        self._writer.writerow(header)
        self._running = True
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def log(self, row: list):
        """Non-blocking.  Call from any thread at any rate."""
        self._q.put(row)

    def _drain(self):
        while self._running or not self._q.empty():
            try:
                row = self._q.get(timeout=0.05)
                self._writer.writerow(row)
            except queue.Empty:
                pass

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._fh.flush()
        self._fh.close()


# ── public logger ─────────────────────────────────────────────────────────────

class DataLogger:
    def __init__(self, base_dir: str = "csv_data", session_name: str = "session"):
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(base_dir, f"{session_name}_{tag}")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir

        self._cam  = _StreamWriter(
            os.path.join(session_dir, "camera.csv"),
            ["timestamp", "pos_x", "pos_z"]
        )
        # self._enc  = _StreamWriter(
        #     os.path.join(session_dir, "encoder.csv"),
        #     ["timestamp", "enc1", "enc2"]
        # )
        self._lc   = _StreamWriter(
            os.path.join(session_dir, "loadcell.csv"),
            ["timestamp", "Fx", "Fy"]
        )
        self._gui  = _StreamWriter(
            os.path.join(session_dir, "gui.csv"),
            ["timestamp", "magnitude", "direction", "Fx", "Fz",
             "T1_left", "T3_right", "tau1", "tau2"]
        )
        self._active = False
        self._cam_logged = False
        print(f"[logger] session → {session_dir}/")

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # ── control ──────────────────────────────────────────────────────────────

    def start(self):
        self._active = True
        print("[logger] recording started")

    def stop(self):
        self._active = False
        self._cam.stop()
        # self._enc.stop()
        self._lc.stop()
        self._gui.stop()
        print("[logger] recording stopped, files closed")

    # ── log calls (no-ops if not started) ────────────────────────────────────

    def log_camera(self, pos_x: float, pos_z: float):
        """Writes exactly one row — the reference position."""
        if not self._active or self._cam_logged:
            return
        self._cam.log([self._ts(), f"{pos_x:.6f}", f"{pos_z:.6f}"])
        self._cam_logged = True

    def log_encoder(self, enc1: float, enc2: float):
        """Teensy encoder logging — disabled for now."""
        # if not self._active:
        #     return
        # self._enc.log([self._ts(), f"{enc1:.6f}", f"{enc2:.6f}"])
        pass

    def log_loadcell(self, fx: float, fy: float):
        """Call inside SeeduinoReceiver._parse_line after each successful parse."""
        if not self._active:
            return
        self._lc.log([self._ts(), f"{fx:.6f}", f"{fy:.6f}"])

    def log_gui(self, magnitude: float, direction: float,
                Fx: float, Fz: float,
                T1: float, T3: float, tau1: float, tau2: float):
        if not self._active:
            return
        self._gui.log([
            self._ts(),
            f"{magnitude:.6f}", f"{direction:.6f}",
            f"{Fx:.6f}",        f"{Fz:.6f}",
            f"{T1:.6f}",        f"{T3:.6f}",
            f"{tau1:.6f}",      f"{tau2:.6f}",
        ])
