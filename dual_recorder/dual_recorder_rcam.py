"""Dual OV9281 recorder for the Radxa Dragon Q6A, using the `rcam` driver.

Same job (and same on-disk format) as dual_recorder.py, but that one talks to
picamera2/libcamera, which this board has no pipeline handler for. Here frames
come from rcam, which drives the CAMSS RDI over V4L2 directly:

    cam{0,1}_frame.msgpack      raw HxW uint8 mono frames, msgpack-numpy packed
    cam{0,1}_timestamp.msgpack  one row per frame:
        [sync, wall_clock_iso, monotonic_ns, sensor_ns, sequence]

The notebooks in trunkpose/dual_notebooks read column 0 as the GPIO sync bit and
column 1 as the frame time, so both stay compatible; columns 3 and 4 are new.

Use column 3, not column 2, to pair frames across the two cameras. Column 2 is a
host arrival time (time.monotonic_ns() once the frame is unpacked) and carries
1.6-1.9 ms of scheduling jitter. Column 3 is the timestamp CAMSS stamps in its
frame-done interrupt - same CLOCK_MONOTONIC, ~100 us of jitter. Column 4 is the
driver's frame counter: a gap in it means the sensor produced a frame that never
reached us, which a gap in the timestamps alone cannot tell apart from a capture
thread being descheduled.

Frame sync: the two sensors self-clock off separate 24 MHz oscillators with no
FSIN wiring between them, so they free-run at an arbitrary phase - measured cold
at -14.9 ms of a 33.3 ms frame period, near anti-phase. Before recording starts,
rcam.FrameSync walks one sensor's phase onto the other by briefly stretching its
vertical blanking, which brings them to ~100 us. The crystals still differ by
~50 ppm (~3 ms/minute), so the alignment is topped up during the recording from
the timestamps as they arrive. Pass --no-frame-sync to record free-running.

The out-of-tree ov9282 driver is not autoloaded at boot, so the recorder
modprobes it itself (via sudo when not already root) before looking for cameras.
Pass --no-modprobe to skip that and manage the driver yourself.

Usage:
    uv run python dual_recorder/dual_recorder_rcam.py -f recordings -n test -c True
    # then 's' to start recording, 'q' to stop and exit
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np

# rcam lives in its own project (editable install in its own venv), so it is not
# necessarily importable from this project's environment.
RCAM_SRC = os.environ.get("RCAM_PATH", "/home/radxa/Documents/rcam/src")
try:
    from rcam import Camera, FrameSync, list_cameras, phase_from_timestamps
except ImportError:
    sys.path.insert(0, RCAM_SRC)
    try:
        from rcam import Camera, FrameSync, list_cameras, phase_from_timestamps
    except ImportError as exc:
        raise SystemExit(
            f"cannot import rcam (looked in {RCAM_SRC}); set RCAM_PATH to its src/"
        ) from exc


DRIVER_MODULE = "ov9282"  # driver for the OV9281; out-of-tree, no autoload
MEDIA_NODE = "/dev/media0"  # CAMSS media graph rcam walks to find the sensors


def driver_loaded(module: str = DRIVER_MODULE) -> bool:
    return os.path.exists(f"/sys/module/{module}")


def _have_terminal() -> bool:
    """True if sudo has somewhere to ask for a password (it reads /dev/tty)."""
    try:
        os.close(os.open("/dev/tty", os.O_RDWR))
        return True
    except OSError:
        return False


def _sensor_in_device_tree() -> bool:
    """True if the booted DTB declares an OV9281, i.e. the overlay is applied."""
    for root, _dirs, files in os.walk("/proc/device-tree"):
        if "compatible" in files:
            try:
                with open(os.path.join(root, "compatible"), "rb") as fh:
                    if b"ovti,ov9281" in fh.read():
                        return True
            except OSError:
                continue
    return False


def _modprobe(module: str) -> bool:
    """Load `module`, escalating with sudo only as far as the situation allows."""
    base = ["modprobe", module]

    def run(cmd, capture=True):
        print(f"-> {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=capture, text=True, check=False)
        except FileNotFoundError:
            print(f"! {cmd[0]} not found - load it manually: sudo modprobe {module}")
            return False
        if proc.returncode == 0:
            return True
        detail = (proc.stderr or "").strip() if capture else f"exit {proc.returncode}"
        print(f"! modprobe failed: {detail}")
        return False

    if os.geteuid() == 0:
        return run(base)
    # sudo -n first: succeeds outright when sudo is passwordless or the ticket is
    # still cached, and tells us whether a password prompt is needed at all.
    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False)
    if probe.returncode == 0:
        return run(["sudo", "-n", *base])
    if _have_terminal():
        return run(["sudo", *base], capture=False)
    # Non-interactive (IDE task, systemd unit, ssh without a tty): sudo cannot
    # prompt, and its own error message is misleading, so say what to do.
    print(
        f"! sudo needs a password and there is no terminal to ask on.\n"
        f"  Load it first:  sudo modprobe {module}\n"
        f"  Or persist it:  echo {module} | sudo tee /etc/modules-load.d/{module}.conf"
    )
    return False


def ensure_driver(
    module: str = DRIVER_MODULE, media: str = MEDIA_NODE, timeout: float = 5.0
) -> bool:
    """Load the sensor driver if it isn't loaded, so a fresh boot just works.

    ov9282 lives in /lib/modules/.../updates and is not autoloaded, so every run
    after a reboot would otherwise fail with "no cameras" until someone ran
    modprobe by hand.
    """
    if not driver_loaded(module):
        print(f"{module} not loaded")
        if not _modprobe(module):
            return False
    # udev creates the media/video nodes a moment after the driver binds.
    deadline = time.monotonic() + timeout
    while not os.path.exists(media) and time.monotonic() < deadline:
        time.sleep(0.1)
    if os.path.exists(media):
        return True
    if not _sensor_in_device_tree():
        print(
            f"! {module} is loaded but the booted device tree declares no ov9281, "
            f"so nothing binds to it and {media} never appears.\n"
            "  The EFI boot DTB is stock - a kernel/image update reverts it. Fix:\n"
            "  rcam/ov9281/scripts/deploy_efi_dtb.sh   (then reboot)"
        )
    else:
        print(
            f"! {module} loaded and the DTB declares the sensor, but {media} did not "
            f"appear within {timeout:.0f}s - check the camera cables and power."
        )
    return False


class SyncLine:
    """GPIO sync input from the mocap trigger, read once per frame.

    The 40-pin header is gpiochip4 (the SoC TLMM) and header pin N maps to a
    TLMM line that is *not* N - the device tree names them, so a line can be
    given either by number or by its header name ("PIN_11"). The default,
    PIN_11 = line 29, is the same physical hole the Raspberry Pi used for
    GPIO17, so existing trigger wiring does not have to move. Header GPIO is
    3.3V (3.63V tolerant) per the Q6A product brief, same as the Pi.

    Note the TLMM ignored bias requests through the character device in
    testing: PIN_11 idles low on its own, but a floating input on another pin
    cannot be pulled down in software - wire an external pull-down if the
    trigger output is open-drain or tri-state.

    Tolerates libgpiod v1 and v2, and falls back to a constant 0 when gpiod is
    missing so a recording without the trigger box still runs.
    """

    def __init__(self, chip: str = "gpiochip4", line: int | str = "PIN_11"):
        self.available = False
        self._read = lambda: 0
        try:
            import gpiod
        except ImportError:
            print("gpiod not installed -> sync flag stays 0")
            return
        requested = line
        try:
            line = self._resolve(gpiod, chip, line)
            if hasattr(gpiod, "LINE_REQ_DIR_IN"):  # libgpiod v1
                self._chip = gpiod.Chip(chip)
                gline = self._chip.get_line(line)
                gline.request(consumer="dual_recorder", type=gpiod.LINE_REQ_DIR_IN)
                self._read = gline.get_value
            else:  # libgpiod v2
                from gpiod.line import Direction, Value

                self._req = gpiod.request_lines(
                    f"/dev/{chip}",
                    consumer="dual_recorder",
                    config={line: gpiod.LineSettings(direction=Direction.INPUT)},
                )
                self._read = lambda: (
                    1 if self._req.get_value(line) == Value.ACTIVE else 0
                )
            self.available = True
            named = "" if str(requested) == str(line) else f" ({requested})"
            print(f"GPIO sync on {chip} line {line}{named}")
        except Exception as exc:  # noqa: BLE001 - any GPIO failure degrades to 0
            print(f"GPIO sync unavailable on {chip} line {line}: {exc}")
            detail = self._describe(gpiod, chip, line)
            if detail:
                print(detail)
            print(
                "  -> sync flag stays 0. Point --sync-chip/--sync-pin at the line "
                "the trigger box is actually wired to."
            )

    @staticmethod
    def _resolve(gpiod, chip: str, line: int | str) -> int:
        """Accept a line number or a device-tree line name such as "PIN_11"."""
        text = str(line)
        if text.lstrip("-").isdigit():
            return int(text)
        if hasattr(gpiod, "LINE_REQ_DIR_IN"):  # libgpiod v1
            found = gpiod.Chip(chip).find_line(text)
            if found is None:
                raise ValueError(f"no line named {text!r} on {chip}")
            return found.offset()
        with gpiod.Chip(f"/dev/{chip}") as c:
            return c.line_offset_from_id(text)

    @staticmethod
    def _describe(gpiod, chip: str, line: int) -> str:
        """Best-effort reason a line request failed (libgpiod v2 introspection).

        The usual cause is a line already claimed by a kernel driver - GPIO17 is
        the Raspberry Pi wiring and is taken on the Q6A - which the raw EINVAL
        from the request does not say.
        """
        try:
            with gpiod.Chip(f"/dev/{chip}") as c:
                info = c.get_line_info(line)
                label = c.get_info().label
                if info.used:
                    who = info.consumer or "a kernel driver"
                    return f"  line {line} on {label} is already claimed by {who}"
                return f"  line {line} on {label} is free, so the request itself failed"
        except Exception:  # noqa: BLE001 - introspection is a nicety, not a must
            return ""

    def get_value(self) -> int:
        return int(self._read())


class CameraWorker(threading.Thread):
    """Captures from one camera as fast as the sensor delivers frames.

    rcam's native backend releases the GIL inside DQBUF/unpack, so one thread per
    camera really does run in parallel. Disk writes are handed to a separate
    writer thread through a bounded queue, so a slow flush stalls the queue
    rather than the capture loop (2 x 1280x800 x 30 fps is ~61 MB/s).
    """

    QUEUE_DEPTH = 32
    PHASE_WINDOW = 90  # frames of history the drift resync fits over (~3 s)

    def __init__(self, cam, index, out_dir, record, sync, start_evt, stop_evt):
        super().__init__(name=f"cam{index}", daemon=True)
        self.cam = cam
        self.index = index
        self.out_dir = out_dir
        self.record = record
        self.sync = sync
        self.start_evt = start_evt
        self.stop_evt = stop_evt

        self.latest = None  # newest frame, for the preview window
        self.frames = 0  # frames captured since start
        self.written = 0  # frames written to disk
        self.dropped = 0  # frames the sensor made that never reached us
        self.error: BaseException | None = None
        self._last_seq: int | None = None
        # Recent sensor timestamps, for the mid-recording phase check. A deque
        # with maxlen is the whole synchronisation: append from this thread,
        # snapshot from the main thread, no lock needed.
        self.recent_ns: collections.deque[int] = collections.deque(
            maxlen=self.PHASE_WINDOW
        )
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_DEPTH)
        self._writer: threading.Thread | None = None

    def _write_loop(self):
        frame_path = os.path.join(self.out_dir, f"cam{self.index}_frame.msgpack")
        stamp_path = os.path.join(self.out_dir, f"cam{self.index}_timestamp.msgpack")
        with open(frame_path, "wb") as fh_frame, open(stamp_path, "wb") as fh_stamp:
            while True:
                item = self._queue.get()
                if item is None:  # sentinel: capture finished
                    break
                frame, sync_val, wall, mono, sensor_ns, seq = item
                fh_frame.write(mp.packb(frame, default=mpn.encode))
                fh_stamp.write(mp.packb([sync_val, wall, mono, sensor_ns, seq]))
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
                if self.record and self.start_evt.is_set():
                    # Naive local time on purpose: the notebook loaders feed
                    # this straight into np.datetime64, which rejects tz-aware
                    # values, and the picamera2 recorder writes the same format.
                    wall = datetime.datetime.now().isoformat(sep=" ")  # noqa: DTZ005
                    self._queue.put(
                        (frame, self.sync.get_value(), wall, mono, sensor_ns, seq)
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


def open_camera(label, fps, exposure_us, gain, vflip, hflip):
    """Configure one OV9281 and report the exposure the sensor actually took."""
    cam = Camera(label)
    cam.configure(size=(1280, 800), bit_depth=8)
    # Frame rate first: it sets vertical_blanking, which bounds the exposure
    # range the driver will accept.
    cam.set_controls({"FrameRate": fps}, settle=False)
    cam.set_controls(
        {
            "ExposureTime": exposure_us,
            "AnalogueGain": gain,
            "VFlip": vflip,
            "HFlip": hflip,
        },
        settle=False,
    )
    lines = cam.get_control("exposure")
    actual_us = lines * cam._line_time_us()
    print(
        f"  {label}: exposure={lines} lines ({actual_us:.0f} us), gain={gain}, "
        f"fps target={fps}"
    )
    if abs(actual_us - exposure_us) > 0.02 * exposure_us:
        print(
            f"  ! {label} exposure clipped to {actual_us:.0f} us "
            f"(asked {exposure_us} us) - flicker banding may return"
        )
    return cam.start()


def make_frame_sync(cams):
    """Build the FrameSync for a camera pair, or None if they cannot be synced.

    The one way it fails is the v4l2-ctl fallback backend, which exposes no
    buffer timestamps. Recording still works there - the sensors just stay
    free-running and the sensor_ns column is None, leaving post-processing only
    the host arrival times, which carry milliseconds of jitter.
    """
    try:
        return FrameSync(cams[0], cams[1])
    except RuntimeError as exc:
        print(f"! frame sync unavailable: {exc}")
        return None


def align_cameras(sync, ref_label, adj_label, tol_us, verbose=True):
    """Bring the second sensor's frame phase onto the first's."""
    print(
        f"Aligning frame phase ({adj_label} -> {ref_label}, "
        f"{sync.period_us / 1000:.2f} ms period, {sync.line_time_us:.2f} us/line):"
    )
    return sync.align(tol_us=tol_us, verbose=verbose)


class KeyReader:
    """'s'/'q' from the preview window, or from a raw terminal when headless."""

    def __init__(self, display: bool):
        self.display = display
        self._fd = None
        self._old = None
        if not display and sys.stdin.isatty():
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def get(self) -> str | None:
        if self.display:
            key = cv2.waitKey(1) & 0xFF
            return chr(key) if key != 255 else None
        if self._fd is None:
            return None
        import select

        if select.select([sys.stdin], [], [], 0.01)[0]:
            return sys.stdin.read(1)
        return None

    def close(self):
        if self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


class RecordData:
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        fps_value=30,
        display=True,
        flicker_hz=50,
        gain=4.0,
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
        resync_every_s=5.0,
        resync_threshold_us=1000.0,
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
            print(
                f"Error: this script requires 2 cameras ({DRIVER_MODULE} is "
                f"loaded but the media graph has {len(detected)}). Check the "
                "camera cables and rcam/ov9281/README.md."
            )
            sys.exit(1)

        self.labels = list(labels) if labels else detected[:2]
        for label in self.labels:
            if label not in detected:
                sys.exit(f"camera {label!r} not in {detected}")

        # Exposure must be an integer multiple of the AC half-period to avoid
        # banding under fluorescent/tube lights: 10000us for 50Hz, 8333us for 60Hz.
        if exposure_us is None:
            exposure_us = 1_000_000 // (flicker_hz * 2)
            print(
                f"Flicker compensation: {flicker_hz}Hz -> ExposureTime={exposure_us}us"
            )
        else:
            # An explicit exposure still has to be a whole number of light
            # periods or the frames pulse in brightness: measured on this board,
            # 5000us gave 9.5% frame-to-frame modulation against 0.1% at 10000us.
            half_period = 1_000_000 / (flicker_hz * 2)
            periods = round(exposure_us / half_period)
            print(f"ExposureTime={exposure_us:.0f}us (explicit)")
            if (
                periods < 1
                or abs(exposure_us - periods * half_period) > 0.02 * half_period
            ):
                nearest = max(1, periods) * half_period
                print(
                    f"! {exposure_us:.0f}us is not a multiple of the {half_period:.0f}us "
                    f"light period at {flicker_hz}Hz - expect the image to pulse in "
                    f"brightness. Flicker-free values near it: {nearest:.0f}us or "
                    f"{nearest + half_period:.0f}us (use --gain to set brightness "
                    f"instead of exposure)."
                )
        frame_period_us = 1_000_000 / fps_value
        if exposure_us > frame_period_us:
            print(
                f"! exposure {exposure_us}us exceeds the {frame_period_us:.0f}us "
                f"frame period at {fps_value} fps - the sensor will slow down"
            )

        print("Opening cameras:")
        self.cams = [
            open_camera(label, fps_value, exposure_us, gain, vflip, hflip)
            for label in self.labels
        ]

        self.record_camera = record_camera
        self._pth = _pth
        self.display = display
        self.fps_value = fps_value
        self.preview_width = preview_width
        self.frame_sync = frame_sync
        self.phase_tol_us = phase_tol_us
        self.resync_every_s = resync_every_s
        self.resync_threshold_us = resync_threshold_us
        self.sync = SyncLine(sync_chip, sync_pin)

    def _phase_now(self, phase_sync, workers):
        """Current phase from the timestamps the workers have already recorded.

        Pure arithmetic on two deques - it takes no frames, so it is safe to
        call from the status loop while the workers own the cameras.
        """
        if phase_sync is None:
            return None
        stamps = [list(w.recent_ns) for w in workers]
        if min(len(x) for x in stamps) < 10:
            return None
        return phase_from_timestamps(stamps[0], stamps[1], phase_sync.period_us)

    def _resync(self, phase_sync, workers, busy):
        """Top up the alignment against the ~50 ppm drift between the crystals.

        Runs on its own thread: a nudge is two v4l2-ctl calls around a sleep of
        roughly a frame period, and doing that inline would stall the preview
        and swallow keypresses.
        """
        if phase_sync is None or busy.is_set():
            return
        stamps = [list(w.recent_ns) for w in workers]
        if min(len(x) for x in stamps) < 30:
            return
        busy.set()

        def work():
            try:
                report = phase_sync.resync_if_needed(
                    stamps[0], stamps[1], self.resync_threshold_us
                )
                if report is not None:
                    print(f"\n  resync: phase was {report.phase_us:+.0f} us, nudged")
            except Exception as exc:  # noqa: BLE001 - a failed nudge must not end the take
                print(f"\n! resync failed: {exc!r}")
            finally:
                busy.clear()

        threading.Thread(target=work, name="resync", daemon=True).start()

    def capture_webcam(self):
        """Capture from both cameras until 'q'; record between 's' and 'q'."""
        for cam in self.cams:
            cam.flush(4)  # drop the queued warm-up frames

        # Built either way: even with alignment disabled it is what measures and
        # reports the phase, which is how you tell whether syncing helped.
        phase_sync = make_frame_sync(self.cams)
        if phase_sync is None:
            print("Recording free-running, on host arrival times only.")
        elif self.frame_sync:
            align_cameras(phase_sync, self.labels[0], self.labels[1], self.phase_tol_us)
        else:
            print("Frame sync disabled (--no-frame-sync); sensors free-run.")
        print("Pair frames in post on the sensor_ns column, not on frame index.")
        print("Press 's' to start recording, 'q' to quit.")

        start_evt = threading.Event()
        stop_evt = threading.Event()
        workers = [
            CameraWorker(
                cam, i, self._pth, self.record_camera, self.sync, start_evt, stop_evt
            )
            for i, cam in enumerate(self.cams)
        ]
        for w in workers:
            w.start()

        keys = KeyReader(self.display)
        window = " | ".join(self.labels)
        t_status = time.monotonic()
        t_resync = t_status
        resync_busy = threading.Event()
        last_counts = [0, 0]
        last_drawn = ()
        try:
            while not stop_evt.is_set():
                if self.display:
                    # Only redraw when a new frame actually arrived: waitKey(1)
                    # spins far faster than the sensors, and resizing the same
                    # two frames 200x a second steals CPU from capture.
                    counts = tuple(w.frames for w in workers)
                    if counts != last_drawn:
                        last_drawn = counts
                        tiles = []
                        for w in workers:
                            frame = w.latest
                            if frame is None:
                                continue
                            h = self.preview_width * frame.shape[0] // frame.shape[1]
                            tiles.append(cv2.resize(frame, (self.preview_width, h)))
                        if tiles:
                            cv2.imshow(window, np.hstack(tiles))
                else:
                    time.sleep(0.01)

                key = keys.get()
                if key == "s" and not start_evt.is_set():
                    if not self.record_camera:
                        print("Not recording (-c True to enable).")
                    else:
                        print("Started recording.")
                        start_evt.set()
                elif key == "q":
                    stop_evt.set()

                now = time.monotonic()
                if now - t_status >= 1.0:
                    fps = [
                        (w.frames - c) / (now - t_status)
                        for w, c in zip(workers, last_counts)
                    ]
                    last_counts = [w.frames for w in workers]
                    t_status = now
                    state = "REC" if start_evt.is_set() else "live"
                    report = self._phase_now(phase_sync, workers)
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
                        + f"  written={[w.written for w in workers]}"
                        + f"  backlog={[w.backlog() for w in workers]}"
                        + f"  dropped={[w.dropped for w in workers]}"
                        + f"  sync={self.sync.get_value()}   ",
                        end="",
                        flush=True,
                    )

                if (
                    phase_sync is not None
                    and self.frame_sync
                    and self.resync_every_s > 0
                    and now - t_resync >= self.resync_every_s
                ):
                    t_resync = now
                    self._resync(phase_sync, workers, resync_busy)
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
            report = self._phase_now(phase_sync, workers)
            if report is not None:
                print(
                    f"final phase {report.phase_us:+.0f} us "
                    f"({abs(report.phase_us) / phase_sync.period_us * 100:.2f}% of a "
                    f"frame), {phase_sync.nudges} nudges applied"
                )
            if self.record_camera and start_evt.is_set():
                print(f"Saved to {self._pth}")

    def run(self):
        """run the program"""
        self.capture_webcam()


if __name__ == "__main__":
    """get parameter from external program"""

    parser = argparse.ArgumentParser(
        prog="Dual OV9281 recorder (rcam)",
        description="Records data from two OV9281 cameras on the Radxa Dragon Q6A",
        epilog="Captures to cam0_frame.msgpack and cam1_frame.msgpack",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument("-s", "--sensors", help="record sensors", required=False)
    parser.add_argument(
        "-z",
        "--hz",
        help="mains frequency for flicker compensation (50 or 60)",
        required=False,
        type=int,
        default=50,
    )
    parser.add_argument("--fps", help="target frame rate", type=float, default=30)
    parser.add_argument(
        "--gain", help="analogue gain, 1.0-16.0", type=float, default=3.0
    )
    parser.add_argument(
        "--exposure", help="exposure in us (overrides --hz)", type=float, default=10000
    )
    parser.add_argument(
        "--vflip", action="store_true", help="flip both cameras vertically"
    )
    parser.add_argument(
        "--hflip", action="store_true", help="flip both cameras horizontally"
    )
    parser.add_argument(
        "--cams",
        nargs=2,
        metavar=("CAM_A", "CAM_B"),
        help="camera labels to use, e.g. --cams CAM2 CAM3",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="run without a preview window"
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=480,
        help="width of each preview tile in px",
    )
    parser.add_argument(
        "--sync-chip", default="gpiochip4", help="gpiochip for the sync input"
    )
    parser.add_argument(
        "--sync-pin",
        default="PIN_11",
        help="sync input line: a TLMM line number, or a header name from the "
        "device tree such as PIN_11 (the default, = line 29, the same header "
        "hole as the Pi's GPIO17). List them with: gpioinfo gpiochip4",
    )
    parser.add_argument(
        "--no-modprobe",
        action="store_true",
        help=f"do not try to load the {DRIVER_MODULE} driver automatically",
    )
    parser.add_argument(
        "--no-frame-sync",
        action="store_true",
        help="skip aligning the two sensors' frame phase and let them free-run "
        "(they start up to half a frame period apart)",
    )
    parser.add_argument(
        "--phase-tol",
        type=float,
        default=200.0,
        help="stop aligning once the two sensors are within this many us "
        "(default 200; the measurement floor is around 100)",
    )
    parser.add_argument(
        "--resync-every",
        type=float,
        default=5.0,
        help="seconds between mid-recording phase checks (0 disables)",
    )
    parser.add_argument(
        "--resync-threshold",
        type=float,
        default=1000.0,
        help="only re-nudge once the phase has drifted past this many us. Each "
        "nudge puts one long frame interval into the recording, so this trades "
        "residual skew against disturbance (default 1000)",
    )

    args = parser.parse_args()

    display = not args.no_display

    # if your not passing any arguments then the default values will be used
    if args.folder is None and args.name is None and args.camera is None:
        print("No arguments passed, please enter manually")

        record_camera = True
        record_sensors = False

        if record_camera or record_sensors:
            _name = input("Enter the name of the recording: ")
        _pth = None
        _folder_name = "recordings"

    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera == "True"
        record_sensors = args.sensors

    if record_camera or record_sensors:
        _pth = os.path.join(
            os.path.dirname(__file__), "..", "data", _folder_name, _name
        )

        if "\n" in _pth:
            _pth = _pth.replace("\n", "")

        if not os.path.exists(_pth):
            os.makedirs(_pth)
    time.sleep(1)

    record_data = RecordData(
        _pth=_pth,
        record_camera=record_camera,
        fps_value=args.fps,
        display=display,
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
        resync_every_s=args.resync_every,
        resync_threshold_us=args.resync_threshold,
    )
    record_data.run()
