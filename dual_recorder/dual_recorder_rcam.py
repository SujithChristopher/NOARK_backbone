"""Dual OV9281 recorder for the Radxa Dragon Q6A, using the `rcam` driver.

Same job (and same on-disk format) as dual_recorder.py, but that one talks to
picamera2/libcamera, which this board has no pipeline handler for. Here frames
come from rcam, which drives the CAMSS RDI over V4L2 directly:

    cam{0,1}_frame.msgpack      raw HxW uint8 mono frames, msgpack-numpy packed
    cam{0,1}_timestamp.msgpack  one [sync, wall_clock_iso, monotonic_ns] per frame

The notebooks in trunkpose/dual_notebooks read column 0 as the GPIO sync bit and
column 1 as the frame time, so both stay compatible.

Timestamps differ from the picamera2 version: rcam exposes no SensorTimestamp,
so column 2 is a host arrival time (time.monotonic_ns() right after the frame is
unpacked), not a sensor exposure time. It carries scheduling jitter of a few
hundred microseconds. If exposure-accurate stamps are ever needed, the V4L2
buffer timestamp is already in rcam's Rust reader (rust/lib.rs v4l2_buffer) and
just needs plumbing out through next_u8().

The out-of-tree ov9282 driver is not autoloaded at boot, so the recorder
modprobes it itself (via sudo when not already root) before looking for cameras.
Pass --no-modprobe to skip that and manage the driver yourself.

Usage:
    uv run python dual_recorder/dual_recorder_rcam.py -f recordings -n test -c True
    # then 's' to start recording, 'q' to stop and exit
"""

from __future__ import annotations

import argparse
import datetime
import math
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
    from rcam import Camera, list_cameras
except ImportError:
    sys.path.insert(0, RCAM_SRC)
    try:
        from rcam import Camera, list_cameras
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

    Tolerates libgpiod v1 and v2, and falls back to a constant 0 when gpiod is
    missing so a recording without the trigger box still runs.
    """

    def __init__(self, chip: str = "gpiochip4", line: int = 17):
        self.available = False
        self._read = lambda: 0
        try:
            import gpiod
        except ImportError:
            print("gpiod not installed -> sync flag stays 0")
            return
        try:
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
            print(f"GPIO sync on {chip} line {line}")
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
        self.error: BaseException | None = None
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
                frame, sync_val, wall, mono = item
                fh_frame.write(mp.packb(frame, default=mpn.encode))
                fh_stamp.write(mp.packb([sync_val, wall, mono]))
                self.written += 1

    def run(self):
        if self.record:
            self._writer = threading.Thread(
                target=self._write_loop, name=f"cam{self.index}-writer", daemon=True
            )
            self._writer.start()
        try:
            while not self.stop_evt.is_set():
                frame = self.cam.capture_array()
                mono = time.monotonic_ns()
                self.latest = frame
                self.frames += 1
                if self.record and self.start_evt.is_set():
                    # Naive local time on purpose: the notebook loaders feed
                    # this straight into np.datetime64, which rejects tz-aware
                    # values, and the picamera2 recorder writes the same format.
                    wall = datetime.datetime.now().isoformat(sep=" ")  # noqa: DTZ005
                    self._queue.put((frame, self.sync.get_value(), wall, mono))
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


def measure_skew(cams, n_frames, frame_period_us):
    """Estimate the free-running phase offset between the two sensors.

    Host-side only: rcam gives no sensor timestamp, so this pairs the arrival
    times of frames grabbed in parallel and includes thread scheduling jitter.

    The two sensors free-run, so pairing frame i of one with frame i of the
    other is arbitrary up to whole frame periods - a raw difference of 30 ms at
    33 ms/frame is really -3 ms with the pairing off by one. So the offsets are
    reduced modulo the frame period and averaged as angles (circular mean),
    which gives the phase difference in (-T/2, +T/2] regardless of pairing.
    """
    print(f"Measuring arrival skew over {n_frames} frame pairs...")
    stamps: list[list[int]] = [[], []]

    def grab(i):
        cams[i].capture_array()
        stamps[i].append(time.monotonic_ns())

    for _ in range(n_frames):
        threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    skews = [(b - a) / 1000.0 for a, b in zip(stamps[0], stamps[1])]
    angles = [2 * math.pi * s / frame_period_us for s in skews]
    cos_mean = sum(math.cos(a) for a in angles) / len(angles)
    sin_mean = sum(math.sin(a) for a in angles) / len(angles)
    scale = frame_period_us / (2 * math.pi)
    mean = math.atan2(sin_mean, cos_mean) * scale
    # Circular ("angular") standard deviation; R -> 1 means tightly clustered.
    r = math.hypot(cos_mean, sin_mean)
    std = scale * math.sqrt(-2 * math.log(r)) if r > 0 else float("inf")
    return mean, std


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
        sync_pin=17,
        auto_modprobe=True,
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
            print(f"ExposureTime={exposure_us}us (explicit, flicker setting ignored)")
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
        self.sync = SyncLine(sync_chip, sync_pin)

    def capture_webcam(self):
        """Capture from both cameras until 'q'; record between 's' and 'q'."""
        frame_period_us = 1_000_000 / self.fps_value

        for cam in self.cams:
            cam.flush(4)  # drop the queued warm-up frames
        mean_skew_us, std_skew_us = measure_skew(
            self.cams, int(self.fps_value * 2), frame_period_us
        )
        print(
            f"Sensor phase offset: {mean_skew_us:+.0f} us +/- {std_skew_us:.0f} us  "
            f"({abs(mean_skew_us) / frame_period_us * 100:.1f}% of the "
            f"{frame_period_us:.0f} us frame period)"
        )
        print("The sensors free-run; correct in post using the monotonic_ns column.")
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
                    print(
                        f"\r{state}  "
                        + "  ".join(
                            f"{lbl} {f:5.1f}fps" for lbl, f in zip(self.labels, fps)
                        )
                        + f"  written={[w.written for w in workers]}"
                        + f"  backlog={[w.backlog() for w in workers]}"
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
                print(f"{label}: captured {w.frames}, written {w.written}")
                if w.error is not None:
                    print(f"{label} stopped on error: {w.error!r}")
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
        "--gain", help="analogue gain, 1.0-16.0", type=float, default=4.0
    )
    parser.add_argument(
        "--exposure", help="exposure in us (overrides --hz)", type=float, default=None
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
        type=int,
        default=17,
        help="line for the sync input (17 is the old RPi pin; on the Q6A many "
        "TLMM lines are taken - the program says so if the request fails)",
    )
    parser.add_argument(
        "--no-modprobe",
        action="store_true",
        help=f"do not try to load the {DRIVER_MODULE} driver automatically",
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
    )
    record_data.run()
