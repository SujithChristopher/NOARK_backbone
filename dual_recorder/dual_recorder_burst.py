"""Burst recorder for static poses: one keypress captures a fixed run of frames.

Built on dual_recorder_rcam.py, which it imports rather than copies - cameras,
flicker-safe exposure, phase alignment and the GPIO sync line all behave
identically here. The difference is what reaches the disk.

The continuous recorder writes every frame between 's' and 'q', so a session
spent moving an object between poses is mostly frames of a hand in motion. For
static analysis that is all waste: what you want is a short stationary burst per
pose. So here the cameras free-run for the preview, but nothing is written until
you press 'a', which captures exactly --frames frames on each camera and stops.
Move the object, press 'a' again, and that is burst 2.

Every frame carries its burst index, so a pose is one `burst == n` filter away
in analysis and the file layout stays the same as the continuous recorder's:

    cam0_frame.msgpack      frames, in capture order across all bursts
    cam0_timestamp.msgpack  one record per frame, [sync, wall, mono,
                            sensor_ns, seq, burst] - burst is the added field
    sync_events.msgpack     every mocap trigger edge, [timestamp_ns, value,
                            line_seqno], on CLOCK_MONOTONIC like sensor_ns
    phase_log.msgpack       one row per phase measurement, [monotonic_ns,
                            phase_us, jitter_us, drift_ppm, n, nudged]
    bursts.json             per-burst manifest (counts, wall times, phase,
                            mono_start_ns/mono_end_ns bounds, sync edge count)

The per-frame `sync` column only says whether the line read high when a frame
was taken, once per frame - too coarse to say when a pulse began, and blind
between bursts. sync_events.msgpack is the answer to "when did it start": the
kernel timestamps each edge in its interrupt handler, on the same clock as
sensor_ns, so pulses and frames compare directly.

Pairing across the two cameras is still done on sensor_ns, exactly as before -
never on frame index, and now never on burst index either, because a burst holds
the same *count* on both cameras but not necessarily the same instants.

The manifest keeps one phase figure per burst, which is what you want when
judging a pose; phase_log.msgpack keeps the once-a-second series behind it,
which is what you want when judging the session. That series matters more here
than in the continuous recorder: this script aligns the sensors once at startup
and never tops the alignment up, so across a long session the ~50 ppm crystal
difference walks the phase out at ~3 ms/minute. Rows carry no burst index - a
row belongs to whichever burst's mono_start_ns/mono_end_ns bracket its
monotonic_ns - and `nudged` is always 0, since nothing nudges mid-session.

Usage:
    uv run python dual_recorder/dual_recorder_burst.py -f static -n trial1 -c True
    uv run python dual_recorder/dual_recorder_burst.py -f static -n trial1 -c True \
        --frames 100 --settle-ms 300
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import queue
import sys
import threading
import time

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_recorder_rcam import (  # noqa: E402 - needs the sys.path line above
    DRIVER_MODULE,
    MEDIA_NODE,
    KeyReader,
    PhaseLog,
    SyncLine,
    align_cameras,
    driver_loaded,
    ensure_driver,
    list_cameras,
    make_frame_sync,
    open_camera,
    phase_from_timestamps,
    resolve_exposure,
)


class BurstWorker(threading.Thread):
    """Captures continuously, but only writes while a burst is armed.

    Capture never stops, because the preview and the phase measurement both
    need a live stream, and because a sensor restarted per burst would come
    back with a fresh unknown phase. Arming is therefore just a counter: the
    next ``remaining`` frames go to the writer queue tagged with the burst
    index, and then the worker goes quiet again.
    """

    QUEUE_DEPTH = 64  # a burst is bounded, so let it queue rather than stall
    PHASE_WINDOW = 90

    def __init__(self, cam, index, out_dir, record, sync, stop_evt):
        super().__init__(name=f"cam{index}", daemon=True)
        self.cam = cam
        self.index = index
        self.out_dir = out_dir
        self.record = record
        self.sync = sync
        self.stop_evt = stop_evt

        self.latest = None
        self.frames = 0  # captured since start, burst or not
        self.written = 0  # handed to the writer, i.e. inside a burst
        self.dropped = 0
        self.error: BaseException | None = None
        self._last_seq: int | None = None
        self.recent_ns: collections.deque[int] = collections.deque(
            maxlen=self.PHASE_WINDOW
        )
        # Guards the (index, remaining) pair against the main thread arming a
        # burst while this thread is mid-decrement.
        self._lock = threading.Lock()
        self._burst_index = 0
        self._remaining = 0
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_DEPTH)
        self._writer: threading.Thread | None = None

    # -- burst control (main thread) ---------------------------------------
    def arm(self, burst_index: int, n: int) -> None:
        with self._lock:
            self._burst_index = burst_index
            self._remaining = n

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    # -- capture -----------------------------------------------------------
    def _write_loop(self):
        frame_path = os.path.join(self.out_dir, f"cam{self.index}_frame.msgpack")
        stamp_path = os.path.join(self.out_dir, f"cam{self.index}_timestamp.msgpack")
        with open(frame_path, "wb") as fh_frame, open(stamp_path, "wb") as fh_stamp:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                frame, sync_val, wall, mono, sensor_ns, seq, burst = item
                fh_frame.write(mp.packb(frame, default=mpn.encode))
                fh_stamp.write(mp.packb([sync_val, wall, mono, sensor_ns, seq, burst]))
                self.written += 1

    def run(self):
        if self.record:
            self._writer = threading.Thread(
                target=self._write_loop, name=f"cam{self.index}-writer", daemon=True
            )
            self._writer.start()
        try:
            while not self.stop_evt.is_set():
                frame, sensor_ns, seq = self.cam.capture_array_meta()
                mono = time.monotonic_ns()
                self.latest = frame
                self.frames += 1
                if seq is not None:
                    if self._last_seq is not None and seq - self._last_seq > 1:
                        self.dropped += seq - self._last_seq - 1
                    self._last_seq = seq
                if sensor_ns is not None:
                    self.recent_ns.append(sensor_ns)
                if not self.record:
                    continue
                with self._lock:
                    if self._remaining <= 0:
                        continue
                    self._remaining -= 1
                    burst = self._burst_index
                # Naive local time on purpose: the notebook loaders feed this
                # straight into np.datetime64, which rejects tz-aware values.
                wall = datetime.datetime.now().isoformat(sep=" ")  # noqa: DTZ005
                self._queue.put(
                    (frame, self.sync.get_value(), wall, mono, sensor_ns, seq, burst)
                )
        except BaseException as exc:  # noqa: BLE001 - surfaced by the main loop
            self.error = exc
            self.stop_evt.set()
        finally:
            if self._writer is not None:
                self._queue.put(None)
                self._writer.join()

    def backlog(self) -> int:
        return self._queue.qsize()


class BurstRecorder:
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        frames_per_burst=50,
        settle_ms=0,
        fps_value=30,
        display=True,
        flicker_hz=50,
        gain=3.0,
        exposure_us=None,
        vflip=False,
        hflip=False,
        labels=None,
        preview_width=480,
        sync_chip="gpiochip4",
        sync_pin="PIN_11",
        auto_modprobe=True,
        frame_sync=True,
        phase_tol_us=200.0,
    ):
        if auto_modprobe:
            ensure_driver()
        elif not driver_loaded():
            print(f"{DRIVER_MODULE} not loaded and --no-modprobe given")
        if not os.path.exists(MEDIA_NODE):
            sys.exit(
                f"{MEDIA_NODE} missing - the camera pipeline is not up. "
                f"Try: sudo modprobe {DRIVER_MODULE}"
            )

        detected = list_cameras()
        print(f"Detected {len(detected)} cameras: {detected or 'none'}")
        if len(detected) < 2:
            print(f"Error: this script requires 2 cameras, found {len(detected)}.")
            sys.exit(1)
        self.labels = list(labels) if labels else detected[:2]
        for label in self.labels:
            if label not in detected:
                sys.exit(f"camera {label!r} not in {detected}")

        exposure_us = resolve_exposure(exposure_us, flicker_hz, fps_value)

        print("Opening cameras:")
        self.cams = [
            open_camera(label, fps_value, exposure_us, gain, vflip, hflip)
            for label in self.labels
        ]
        self.sync = SyncLine(sync_chip, sync_pin)
        self._pth = _pth
        self.record_camera = record_camera
        self.frames_per_burst = frames_per_burst
        self.settle_s = settle_ms / 1000.0
        self.fps_value = fps_value
        self.display = display
        self.preview_width = preview_width
        self.frame_sync = frame_sync
        self.phase_tol_us = phase_tol_us
        self.bursts: list[dict] = []
        self.phase_log = PhaseLog()

    def _phase_now(self, phase_sync, workers):
        if phase_sync is None:
            return None
        stamps = [list(w.recent_ns) for w in workers]
        if min(len(x) for x in stamps) < 10:
            return None
        return phase_from_timestamps(stamps[0], stamps[1], phase_sync.period_us)

    def _draw(self, workers, burst_no, active, last_drawn):
        counts = tuple(w.frames for w in workers)
        if counts == last_drawn:
            return last_drawn
        tiles = []
        for w in workers:
            frame = w.latest
            if frame is None:
                continue
            h = self.preview_width * frame.shape[0] // frame.shape[1]
            tiles.append(cv2.resize(frame, (self.preview_width, h)))
        if tiles:
            canvas = np.hstack(tiles)
            if canvas.ndim == 2:  # mono sensor - colourise so the overlay reads
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if active:
                done = self.frames_per_burst - max(w.remaining for w in workers)
                text = f"BURST {burst_no}  {done}/{self.frames_per_burst}"
                colour = (0, 0, 255)
            else:
                text = f"ready - press 'a' for burst {burst_no + 1}"
                colour = (0, 200, 0)
            cv2.putText(
                canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2
            )
            cv2.imshow(" | ".join(self.labels), canvas)
        return counts

    def capture(self):
        for cam in self.cams:
            cam.flush(4)

        phase_sync = make_frame_sync(self.cams)
        if phase_sync is None:
            print("Recording free-running, on host arrival times only.")
        elif self.frame_sync:
            align_cameras(phase_sync, self.labels[0], self.labels[1], self.phase_tol_us)
        else:
            print("Frame sync disabled (--no-frame-sync); sensors free-run.")
        print("Pair frames in post on the sensor_ns column, not on frame index.")
        print(
            f"Press 'a' to capture a {self.frames_per_burst}-frame burst, "
            f"'q' to finish (in this terminal or the preview window)."
        )
        if not self.record_camera:
            print("! -c True not given: 'a' will preview bursts but write nothing.")

        stop_evt = threading.Event()
        workers = [
            BurstWorker(cam, i, self._pth, self.record_camera, self.sync, stop_evt)
            for i, cam in enumerate(self.cams)
        ]
        for w in workers:
            w.start()

        keys = KeyReader(self.display)
        burst_no = 0
        active = None  # the in-flight burst's bookkeeping, or None when idle
        t_status = time.monotonic()
        last_counts = [0, 0]
        last_drawn = ()
        try:
            while not stop_evt.is_set():
                if self.display:
                    last_drawn = self._draw(
                        workers, burst_no, active is not None, last_drawn
                    )
                else:
                    time.sleep(0.005)

                key = keys.get()
                if key in ("a", "A"):
                    if active is not None:
                        print("\n  burst still running - wait for it to finish.")
                    else:
                        burst_no += 1
                        if self.settle_s:
                            time.sleep(self.settle_s)
                        started = [w.written for w in workers]
                        for w in workers:
                            w.arm(burst_no, self.frames_per_burst)
                        active = {
                            "index": burst_no,
                            "requested": self.frames_per_burst,
                            "started_at": datetime.datetime.now().isoformat(  # noqa: DTZ005
                                sep=" "
                            ),
                            "_written_before": started,
                            "_t0": time.monotonic(),
                            # CLOCK_MONOTONIC, same as the sync edges and
                            # sensor_ns, so a pulse maps onto a burst by
                            # comparison alone.
                            "mono_start_ns": time.monotonic_ns(),
                        }
                        print(f"\n  burst {burst_no}: capturing...", flush=True)
                elif key in ("q", "Q"):
                    if active is not None:
                        print("\n  finishing the burst in flight before quitting...")
                        while any(w.remaining for w in workers) and not stop_evt.is_set():
                            time.sleep(0.01)
                    stop_evt.set()

                if active is not None and not any(w.remaining for w in workers):
                    # `remaining` drops to zero while the last frame is still on
                    # its way to the writer, so wait on the writes themselves.
                    # Trusting the counter here undercounts the burst by one in
                    # the manifest even though the files got every frame.
                    before = active.pop("_written_before")
                    want = active["requested"]
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and not stop_evt.is_set():
                        if all(w.written - b >= want for w, b in zip(workers, before)):
                            break
                        time.sleep(0.005)
                    got = [w.written - b for w, b in zip(workers, before)]
                    report = self._phase_now(phase_sync, workers)
                    self.phase_log.add(report)
                    active["seconds"] = round(time.monotonic() - active.pop("_t0"), 3)
                    active["mono_end_ns"] = time.monotonic_ns()
                    inside = [
                        e
                        for e in self.sync.events()
                        if active["mono_start_ns"] <= e[0] <= active["mono_end_ns"]
                    ]
                    active["sync_edges"] = len(inside)
                    active["sync_first_rise_ns"] = next(
                        (ts for ts, value, _ in inside if value), None
                    )
                    active["frames"] = dict(zip(self.labels, got))
                    active["phase_us"] = (
                        round(report.phase_us, 1) if report is not None else None
                    )
                    self.bursts.append(active)
                    phase = (
                        f", phase {report.phase_us:+.0f}us"
                        if report is not None
                        else ""
                    )
                    print(
                        f"  burst {active['index']}: "
                        + " + ".join(f"{lbl} {n}" for lbl, n in zip(self.labels, got))
                        + f" frames in {active['seconds']:.2f}s{phase}"
                        + (
                            f", {active['sync_edges']} sync edges"
                            if active["sync_edges"]
                            else ""
                        ),
                        flush=True,
                    )
                    active = None

                now = time.monotonic()
                if now - t_status >= 1.0:
                    fps = [
                        (w.frames - c) / (now - t_status)
                        for w, c in zip(workers, last_counts)
                    ]
                    last_counts = [w.frames for w in workers]
                    t_status = now
                    state = f"BURST {burst_no}" if active is not None else "idle "
                    report = self._phase_now(phase_sync, workers)
                    self.phase_log.add(report)
                    phase = (
                        f"  phase={report.phase_us:+6.0f}us"
                        if report is not None
                        else ""
                    )
                    print(
                        f"\r{state}  "
                        + "  ".join(
                            f"{lbl} {f:5.1f}fps" for lbl, f in zip(self.labels, fps)
                        )
                        + phase
                        + f"  bursts={len(self.bursts)}"
                        + f"  written={[w.written for w in workers]}"
                        + f"  dropped={[w.dropped for w in workers]}"
                        + f"  sync={self.sync.get_value()}   ",
                        end="",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            stop_evt.set()
            for w in workers:
                w.join(timeout=10)
            keys.close()
            if self.display:
                cv2.destroyAllWindows()
            for cam in self.cams:
                cam.stop()
            print()
            for label, w in zip(self.labels, workers):
                print(
                    f"{label}: captured {w.frames}, written {w.written}, "
                    f"sensor frames lost {w.dropped}"
                )
                if w.error is not None:
                    print(f"{label} stopped on error: {w.error!r}")
            self.phase_log.add(self._phase_now(phase_sync, workers))
            self.sync.close()
            if self.record_camera and self.bursts:
                events_path = self.sync.write_events(self._pth)
                if self.sync.edges_available:
                    print(
                        f"sync: {self.sync.event_count} edges latched"
                        + (f" -> {events_path}" if events_path else " (none to write)")
                    )
                else:
                    print("sync: no edge timestamps (edge detection unavailable)")
                phase_path = self.phase_log.write(self._pth)
                if phase_path is not None:
                    print(f"phase: {self.phase_log.count} measurements -> {phase_path}")
                else:
                    print("phase: nothing measured (no frame sync on this backend)")
                manifest = os.path.join(self._pth, "bursts.json")
                with open(manifest, "w") as fh:
                    json.dump(
                        {
                            "cameras": self.labels,
                            "frames_per_burst": self.frames_per_burst,
                            "fps": self.fps_value,
                            "bursts": self.bursts,
                        },
                        fh,
                        indent=2,
                    )
                print(f"{len(self.bursts)} bursts -> {self._pth}")
                print(f"manifest: {manifest}")
            elif self.record_camera:
                print("No bursts captured - nothing written.")

    def run(self):
        self.capture()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dual OV9281 burst recorder (rcam)",
        description="Captures a fixed burst of frames per keypress, for static poses",
        epilog="Each frame is tagged with its burst index; see bursts.json",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument(
        "--frames", help="frames per burst", type=int, default=50
    )
    parser.add_argument(
        "--settle-ms",
        help="wait this long after the keypress before capturing, to let the "
        "rig stop wobbling (default 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-z",
        "--hz",
        help="mains frequency for flicker compensation (50 or 60)",
        type=int,
        default=50,
    )
    parser.add_argument("--fps", help="target frame rate", type=float, default=30)
    parser.add_argument(
        "--gain", help="analogue gain, 1.0-16.0", type=float, default=2.0
    )
    parser.add_argument(
        "--exposure",
        help="exposure in us (overrides --hz; default: one --hz half-period). "
        "Set brightness with --gain instead.",
        type=float,
        default=None,
    )
    parser.add_argument("--vflip", action="store_true", help="flip both vertically")
    parser.add_argument("--hflip", action="store_true", help="flip both horizontally")
    parser.add_argument(
        "--cams", nargs=2, metavar=("CAM_A", "CAM_B"), help="camera labels to use"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="run without a preview window"
    )
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--sync-chip", default="gpiochip4")
    parser.add_argument("--sync-pin", default="PIN_11")
    parser.add_argument("--no-modprobe", action="store_true")
    parser.add_argument(
        "--no-frame-sync", action="store_true", help="skip phase alignment"
    )
    parser.add_argument("--phase-tol", type=float, default=200.0)

    args = parser.parse_args()

    if args.frames < 1:
        sys.exit("--frames must be at least 1")

    if args.folder is None and args.name is None and args.camera is None:
        print("No arguments passed, please enter manually")
        record_camera = True
        _name = input("Enter the name of the recording: ")
        _folder_name = "static"
    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera == "True"

    _pth = None
    if record_camera:
        _pth = os.path.join(
            os.path.dirname(__file__), "..", "data", _folder_name, _name
        )
        _pth = _pth.replace("\n", "")
        if not os.path.exists(_pth):
            os.makedirs(_pth)

    BurstRecorder(
        _pth=_pth,
        record_camera=record_camera,
        frames_per_burst=args.frames,
        settle_ms=args.settle_ms,
        fps_value=args.fps,
        display=not args.no_display,
        flicker_hz=args.hz,
        gain=args.gain,
        exposure_us=args.exposure,
        vflip=args.vflip,
        hflip=args.hflip,
        labels=args.cams,
        preview_width=args.preview_width,
        sync_chip=args.sync_chip,
        sync_pin=args.sync_pin,
        auto_modprobe=not args.no_modprobe,
        frame_sync=not args.no_frame_sync,
        phase_tol_us=args.phase_tol,
    ).run()
