# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NOARK is a Raspberry Pi-based research platform combining two subsystems:

1. **Motion tracking** (`GameStream/`, `recorder/`) — ArUco marker detection, pose estimation, and real-time UDP streaming to Godot for AR/VR applications.
2. **Force control** (`vaideesh/`) — Cable-driven parallel robot with dual Maxon motors, real-time kinematics, 3-axis load cell feedback, and a PySide6 GUI for force validation experiments.

## Key Commands

### Code Quality
```bash
ruff check --extend-include="*.ipynb"   # lint Python + notebooks
ruff format --extend-include="*.ipynb"  # format
```

### Motion Tracking
```bash
# Production tracking → UDP to Godot
python GameStream/stream_optimize_v2.py --calib calibration_toml/current/settings.toml --udp

# Desktop webcam testing
python GameStream/webcamera3.py --calib calibration_toml/current/settings_webcam.toml --udp

# Recording (msgpack, GPIO-synced)
python recorder/recorder.py -f recordings_160fov -n test_recording -c True

# Calibration capture
python recorder/calibration_capture.py
```

### Force Validation GUI
```bash
# Main force control experiment GUI (requires RPi + Teensy + Seeeduino)
python vaideesh/control_implementation_with_GUI/force_validation/force_validation.py
```

## Architecture

### Motion Tracking Subsystem

**Core pipeline**: Picamera2 (RPi) or cv2.VideoCapture (desktop) → ArUco detection (`DICT_APRILTAG_36h11`, marker IDs [4, 8, 12, 14, 20]) → 3D pose with per-marker offsets → ExponentialMovingAverageFilter3D (alpha=0.2) → UDP binary float32 `[state, x, y, z]` to Godot.

**State machine** in `GameStream/stream_optimize_v2.py`:

| State | Code | Valid Commands |
|-------|------|----------------|
| IDLE | 0 | `CAPTURE_REF`, `LOAD_REF` |
| REFERENCE_CAPTURED | 1 | `START_TRACK`, `USER:<id>`, `SAVE_REF` |
| TRACKING | 2 | `STOP_TRACK`, `USER:<id>` |
| RECORDING | 3 | `STOP_TRACK`, `CHANGE:<id>` |

Reference frame auto-saved/loaded from `~/Documents/NOARK/reference_frames/reference_frame.json`. See [UDP_PROTOCOL.md](UDP_PROTOCOL.md) for full wire format.

**Marker offsets** (must match physical bracket geometry — [GameStream/stream_optimize_v2.py:22-28](GameStream/stream_optimize_v2.py#L22-L28)):
```python
MARKER_OFFSETS = {
    4:  [0.00,  0.1,   -0.069],
    8:  [0.00,  0.01,  -0.069],
    12: [0.00,  0.0,   -0.1075],
    14: [-0.09, 0.0,   -0.069],
    20: [0.1,   0.0,   -0.069],
}
```

### Force Control Subsystem (`vaideesh/`)

**Hardware topology:**
```
Raspberry Pi
├── /dev/ttyACM0 → Teensy LC  (dual encoder readback + motor PWM/DIR commands)
├── /dev/ttyACM1 → Seeeduino  (3-axis HX711 load cell, ~200 Hz)
└── Picamera2   → fisheye camera (160° FOV)

Teensy LC
├── Encoders: 6400 PPR, pins ENC1A=19/ENC1B=18, ENC2A=20/ENC2B=21
└── Motors:   Maxon EC-22 (max 1 Nm), torque constant 0.231 Nm/A
              PWM range 10–90% (410–3686 of 4095), max 5A
```

**Serial protocols:**
- Teensy → Pi: `{raw_e1},{raw_e2}\n`; reset confirmation: `ENC_RESET\n`
- Seeeduino → Pi: `avg X: {fx}\tavg Y: {fy}\t...`; tare command: `T\n` → `TARE DONE`

**Key files in `vaideesh/control_implementation_with_GUI/force_validation/`:**
- `force_validation.py` — PySide6 GUI (906 lines). Runs 4 threads (camera, Teensy, load cell, GUI 100 Hz). Solves cable tensions from desired force vector: T1/T3 via linear algebra, then τ = -T × R_spool (R_spool = 0.033 m). Contains `AutoSweep` class for automated force sweeps (7 angles × 5 magnitudes, 5 s/step).
- `camera_pose_fv.py` — Detects markers [12, 14, 20] on the NOARK handle using fisheye intrinsics + table frame extrinsics. Outputs `noark_in_table_frame` = [x, z] in meters.
- `teensy_seed_fv.py` — `TeensyPort` and `SeeduinoReceiver` serial classes with callback support. Stale detection: >100 ms without update flags data.
- `data_logger.py` — Non-blocking 4-stream CSV logger. Each stream writes to its own queue/daemon thread: `camera_*.csv`, `encoder_*.csv`, `loadcell_*.csv`, `gui_*.csv`. Sessions in `logs/{stream}_{YYYYMMDD_HHMMSS}.csv`.

**Calibration files used by force_validation:**
- Camera intrinsics: `notebooks/calibration/output/good.toml`
- Table frame extrinsics (R + T, camera → table): `estimator/charuco_pose/charuco_pose_picam.toml`
- Motor/pulley geometry: `vaideesh/table_frame_data.toml` (pulleys ML/MR at [±0.065, -0.707] m)

### Configuration System

TOML files drive all calibration and settings:

```toml
[calibration]
camera_matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
dist_coeffs = [[k1, k2, p1, p2, k3]]   # can be 4 coefficients for fisheye

[aruco]
marker_length = 0.05      # meters
marker_spacing = 0.01

[camera]
resolution = [1200, 800]

[stream_data]
udp = false
ip = "localhost"
port = 8000
```

Active configs in `calibration_toml/current/`; archived in `calibration_toml/old/`.

### Data Storage

**Runtime (gitignored):**
```
~/Documents/NOARK/
├── reference_frames/reference_frame.json   # auto-loaded on startup
└── data/[patient_id]/Session-YYYY-MM-DD/MovementData/timestamp_data.csv

./data/recordings_160fov/[name]/
├── webcam_color.msgpack
└── webcam_timestamp.msgpack
```

**Force validation logs** (gitignored): `vaideesh/control_implementation_with_GUI/force_validation/logs/`

## Camera Calibration Workflow

1. Capture chessboard frames: `python recorder/calibration_capture.py` → msgpack in `./data/calibration/`
2. Process in `notebooks/calibration/camera_calibration_*.ipynb` using `cv2.fisheye.calibrate()`
3. Save output to `calibration_toml/current/`

Alternatively, use `calibration_programs/simplified_fisheye_calibration.py` for automated fisheye calibration.

## Dependencies

From [requirements.txt](requirements.txt): `opencv-python`, `opencv-contrib-python`, `numpy==1.26.4`, `picamera2`, `toml`, `msgpack`, `msgpack-numpy`, `scipy`, `matplotlib`, `mediapipe`.

**Not in requirements.txt but required:**
- `pyserial` — serial communication with Teensy/Seeeduino
- `PySide6` — GUI framework for `vaideesh/` force control applications

## Testing Without RPi Hardware

The `example_dataset/` directory contains sample calibration and recording data. All `GameStream/` code runs on desktop via `cv2.VideoCapture` when `picamera2` is unavailable. Force control GUI (`vaideesh/`) requires physical hardware.
