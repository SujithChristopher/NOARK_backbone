# %% [markdown]
# # Reference preview videos from the recorded frames
#
# The takes are stored as raw `cam{0,1}_frame.msgpack`, three to eight gigabytes
# each, which is excellent for analysis and useless for looking at. This script
# turns every take of a session into one small side-by-side mp4 so the movement
# can be eyeballed: what the dome actually did, when it left the frame, which
# tags were facing away.
#
# The video is a reference, not data. It is lossy, downscaled to half in each
# axis, and played at **3x** wall clock.
#
# Speed comes from decimation rather than from a high frame rate: the output is
# always 30 fps and every `round(capture_fps * SPEED / 30)`-th frame is kept.
# Writing all frames into a 90 or 300 fps container would give the same 3x on
# paper, but most players either clamp the rate or stutter, and the file would
# be several times larger for a clip nobody measures anything from.
#
# The two sensors free-run against each other, so the halves are paired on the
# shared host-monotonic clock exactly as `05_static_jitter.py` does, not by
# index. A cam0 frame with no cam1 frame close enough keeps a black right half
# rather than being dropped, so the timeline stays linear.
#
# Run from the repository root:
#
# ```powershell
# uv run python jitter_model/00_video_convert.py
# uv run python jitter_model/00_video_convert.py --take dome_random_movement_sep18_26
# ```

# %% Imports
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy as mpn
import numpy as np
import toml
from tqdm.auto import tqdm

# %% Paths and settings
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:  # running cells in an interactive kernel
    NOTEBOOK_DIR = Path.cwd()
    if NOTEBOOK_DIR.name != "jitter_model":
        NOTEBOOK_DIR = NOTEBOOK_DIR / "jitter_model"

PROJECT_ROOT = NOTEBOOK_DIR.parent

SESSION_DIR = PROJECT_ROOT / "data" / "dome" / "sep18_26"
SESSION_TOML = SESSION_DIR / "session.toml"

# Which takes to convert. None means every take named in session.toml; set a
# list of directory names to restrict it when working in a kernel.
TAKES = None

CAMERA_NAMES = ("cam0", "cam1")

# Playback speed and the rate the file is written at. Frames are dropped to
# reach the speed, so these two together fix the decimation step.
SPEED = 3.0
OUTPUT_FPS = 30

# Half of the 1280x800 sensor in each axis, so the pair lands at 1280x400.
PANEL_SIZE = (640, 400)

# x264 quality. 28 is visibly lossy on the tag edges and perfectly adequate for
# watching the dome move; nothing is ever measured off this file.
CRF = 28
PRESET = "veryfast"

# Cap the frames read per take, for a quick smoke test. None reads the take.
MAX_FRAMES = None

# A pair straddling different exposures is worse than no pair, so reject beyond
# this fraction of a frame period. Same rule as 05_static_jitter.py.
MAX_PAIR_FRACTION_OF_FRAME = 0.55


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Write side-by-side preview videos for a session's takes."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Session directory holding session.toml (default: data/dome/sep18_26).",
    )
    parser.add_argument(
        "--take",
        action="append",
        default=None,
        help="Take directory name to convert; repeatable. Default: every take "
        "named in session.toml.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many cam0 frames per take, for a smoke test.",
    )
    return parser.parse_args(argv)


# argparse only when actually run as a script; in a kernel sys.argv is the
# kernel's own command line and would fail to parse.
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    _args = _parse_args(sys.argv[1:])
    if _args.session is not None:
        SESSION_DIR = _args.session
        SESSION_TOML = SESSION_DIR / "session.toml"
    if _args.take is not None:
        TAKES = _args.take
    if _args.max_frames is not None:
        MAX_FRAMES = _args.max_frames

SESSION_DIR = Path(SESSION_DIR)
if not SESSION_DIR.is_absolute():
    SESSION_DIR = PROJECT_ROOT / SESSION_DIR
SESSION_TOML = Path(SESSION_TOML)
if not SESSION_TOML.is_absolute():
    SESSION_TOML = PROJECT_ROOT / SESSION_TOML

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError(
        "ffmpeg is not on PATH. It does the encoding; cv2.VideoWriter's mp4v "
        "is several times larger at the same legibility."
    )

session = toml.load(SESSION_TOML)
take_names = list(TAKES) if TAKES else list(session["takes"].values())
print(f"Session : {SESSION_DIR}")
print(f"Takes   : {', '.join(take_names)}")


# %% Timestamps and pairing
def load_frame_times(path):
    """Per-frame timing for one camera, plus the burst index when recorded.

    Record layout written by the dual recorder:
        [sync, wall_clock_iso, monotonic_ns, sensor_ns, sequence]

    The burst recorder appends a sixth column holding the 1-based burst number.
    Older takes stop at five, so it is read only when present.

    `monotonic_ns` is the right clock here, not the tighter `sensor_ns`: the two
    sensors stamp `sensor_ns` off their own free-running counters, which share
    no origin, whereas `monotonic_ns` is stamped on the host and is directly
    comparable between the cameras. Its 1.6-1.9 ms of scheduling jitter matters
    for a phase measurement and not at all for choosing which frame to show.
    """
    with path.open("rb") as stream:
        records = list(msgpack.Unpacker(stream, object_hook=mpn.decode))
    if not records:
        raise ValueError(f"No timestamp records in {path}")
    monotonic_ns = np.asarray([int(record[2]) for record in records], dtype=np.int64)
    if len(records[0]) > 5:
        burst = np.asarray([int(record[5]) for record in records], dtype=np.int32)
    else:
        burst = np.zeros(len(records), dtype=np.int32)
    return monotonic_ns, burst


def pair_on_monotonic_clock(cam0_ns, cam1_ns, max_gap_ns):
    """For each cam0 frame, the nearest cam1 frame, or -1 if none is close."""
    insertion = np.searchsorted(cam1_ns, cam0_ns)
    before = np.clip(insertion - 1, 0, len(cam1_ns) - 1)
    after = np.clip(insertion, 0, len(cam1_ns) - 1)
    take_after = np.abs(cam1_ns[after] - cam0_ns) < np.abs(cam1_ns[before] - cam0_ns)
    nearest = np.where(take_after, after, before)
    return np.where(np.abs(cam1_ns[nearest] - cam0_ns) <= max_gap_ns, nearest, -1)


# %% Frame streaming
class FrameStream:
    """Forward-only reader over a `*_frame.msgpack`, decoding only what is kept.

    The files have no index, so a frame can only be reached by walking every
    record before it. `Unpacker.skip` walks a record without building its numpy
    array, which is what makes decimation actually cheaper than reading the
    whole take: at a step of 3 or 10, most records are never decoded.
    """

    def __init__(self, path):
        self._stream = path.open("rb")
        self._unpacker = msgpack.Unpacker(self._stream, object_hook=mpn.decode)
        self._position = 0
        self._last_index = None
        self._last_frame = None

    def at(self, index):
        """The frame at ``index``, which must not be behind the cursor.

        The most recent frame is held, because where one camera dropped a frame
        two consecutive frames of the other pair onto the same index of this
        one, and that is a repeat rather than a seek backwards.
        """
        if index == self._last_index:
            return self._last_frame
        if index < self._position:
            raise ValueError(
                f"{index} is behind the cursor at {self._position}; this reader "
                "only moves forward"
            )
        while self._position < index:
            self._unpacker.skip()
            self._position += 1
        self._last_frame = self._unpacker.unpack()
        self._last_index = index
        self._position += 1
        return self._last_frame

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()


def to_panel(frame, label):
    """One camera's frame as a labelled half of the output, in grayscale."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    panel = cv2.resize(frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    # Black outline under white text, so the label survives both a blown-out
    # ceiling light and the dark half of the room.
    for colour, thickness in ((0, 3), (255, 1)):
        cv2.putText(
            panel,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            thickness,
            cv2.LINE_AA,
        )
    return panel


def blank_panel(label):
    return to_panel(np.zeros(PANEL_SIZE[::-1], dtype=np.uint8), label)


def open_encoder(output_path, width, height):
    command = [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(OUTPUT_FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        str(CRF),
        "-preset",
        PRESET,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


# %% Convert one take
def convert_take(take_dir):
    cam0_ns, cam0_burst = load_frame_times(take_dir / "cam0_timestamp.msgpack")
    cam1_ns, _cam1_burst = load_frame_times(take_dir / "cam1_timestamp.msgpack")

    capture_fps = 1e9 / float(np.median(np.diff(cam0_ns)))
    max_gap_ns = MAX_PAIR_FRACTION_OF_FRAME * float(np.median(np.diff(cam0_ns)))
    pair = pair_on_monotonic_clock(cam0_ns, cam1_ns, max_gap_ns)

    frame_total = len(cam0_ns) if MAX_FRAMES is None else min(len(cam0_ns), MAX_FRAMES)
    step = max(1, round(capture_fps * SPEED / OUTPUT_FPS))
    kept = np.arange(0, frame_total, step)
    has_burst = bool(cam0_burst.any())

    output_path = take_dir / f"{take_dir.name}_{SPEED:g}x.mp4"
    print(
        f"\n{take_dir.name}: {frame_total} frames at {capture_fps:.1f} fps, "
        f"1 frame in {step} kept -> {len(kept)} frames, "
        f"{len(kept) / OUTPUT_FPS:.1f} s at {SPEED:g}x "
        f"({100.0 * (pair[:frame_total] >= 0).mean():.1f}% paired)"
    )

    encoder = open_encoder(output_path, 2 * PANEL_SIZE[0], PANEL_SIZE[1])
    try:
        with (
            FrameStream(take_dir / "cam0_frame.msgpack") as cam0_frames,
            FrameStream(take_dir / "cam1_frame.msgpack") as cam1_frames,
        ):
            for index in tqdm(kept, desc=f"Encoding {take_dir.name}"):
                index = int(index)
                seconds = (cam0_ns[index] - cam0_ns[0]) / 1e9
                stamp = f"{index}  {seconds:6.2f}s"
                if has_burst:
                    stamp += f"  burst {cam0_burst[index]}"

                left = to_panel(cam0_frames.at(index), f"cam0  {stamp}")
                match = int(pair[index])
                # An unpaired frame still gets its slot: dropping it would make
                # the clip jump in time wherever cam1 missed a frame.
                right = (
                    to_panel(cam1_frames.at(match), f"cam1  {stamp}")
                    if match >= 0
                    else blank_panel("cam1  no paired frame")
                )
                encoder.stdin.write(np.hstack((left, right)).tobytes())
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {take_dir.name}")

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {output_path} ({size_mb:.1f} MB)")
    return output_path


# %% Every take in the session
for take_name in take_names:
    take_dir = SESSION_DIR / take_name
    missing = [
        name
        for name in CAMERA_NAMES
        if not (take_dir / f"{name}_frame.msgpack").exists()
    ]
    if missing:
        print(f"\n{take_name}: skipped, no {', '.join(missing)} frame data")
        continue
    convert_take(take_dir)
