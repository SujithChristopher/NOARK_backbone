# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi Python project for camera calibration and computer vision applications with UDP communication to Godot. The project focuses on ArUco marker detection, pose estimation, and real-time camera streaming for AR/VR applications.

## Directory Structure

```
NOARK_backbone/
├── GameStream/              # Real-time ArUco tracking applications (stream_*.py)
├── recorder/                # Data capture with GPIO sync (15+ variants)
├── notebooks/               # Jupyter notebooks + support libraries (ar_support.py, pd_support.py)
├── calibration_programs/    # Camera calibration tools
├── calibration_toml/        # TOML configuration files (current/ and old/)
├── helper_programs/         # Testing utilities (GPIO, distance, integrity checks)
├── example_dataset/         # Sample data for testing without RPi
├── data/                    # Local recordings (gitignored)
└── UDP_PROTOCOL.md          # Detailed UDP communication specification
```

## Key Commands

### Code Quality
```bash
ruff check --extend-include="*.ipynb"  # Lint Python files and notebooks
ruff format --extend-include="*.ipynb"  # Format code
```

### Running Applications

#### Production Tracking (Recommended)
```bash
# Main production tracking system with state machine and UDP control
python GameStream/stream_optimize_v2.py --calib calibration_toml/current/settings.toml --udp

# Desktop testing with webcam
python GameStream/webcamera3.py --calib calibration_toml/current/settings_webcam.toml --udp
```

#### Other Stream Variants
```bash
# Legacy base implementation
python GameStream/stream_april.py

# Optimized variants
python GameStream/stream_optimize.py
python GameStream/stream_optimize_v3.py

# Fisheye undistortion variants (experimental)
python GameStream/stream_april_undistort.py
python GameStream/stream_april_undistort_v2.py
```

#### Recording
```bash
# Main recorder with GPIO sync
python recorder/recorder.py -f recordings_160fov -n test_recording -c True

# Corner detection recorder
python recorder/recorder_corners.py -f recordings_160fov -n test_recording -c True

# Calibration capture (for chessboard patterns)
python recorder/calibration_capture.py
python recorder/record_chessboard.py

# Raw preview modes
python recorder/raw_preview.py
python recorder/undistort_preview.py
```

#### Helper Programs
```bash
# GPIO pin 17 testing (RPi only)
python helper_programs/gpio_test.py

# Distance measurement validation
python helper_programs/distance_test.py

# Data integrity check
python helper_programs/file_integrity.py
```

## Configuration System

The project uses TOML files for camera calibration and settings:

- **Current calibrations**: `calibration_toml/current/` - Active production configurations
- **Archived calibrations**: `calibration_toml/old/` - Previous calibration attempts
- **Settings files**: `settings.toml` (RPi camera), `settings_webcam.toml` (USB webcam)

### TOML Structure
```toml
[calibration]
camera_matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]  # 3x3 intrinsic matrix
dist_coeffs = [[k1, k2, p1, p2, k3]]  # Distortion coefficients (can be 5+ for fisheye)

[aruco]
marker_length = 0.05        # Marker size in meters (5cm)
marker_spacing = 0.01       # Spacing between markers (1cm)

[camera]
resolution = [1200, 800]    # Frame dimensions

[stream_data]
udp = false
ip = "localhost"
port = 8000                 # Default UDP port for Godot communication

[pose]                      # Optional: human pose estimation toggles
[model]                     # Optional: YOLO model paths
[display]                   # Optional: visualization settings
```

**Loading calibration in code:**
```python
import toml
calib_data = toml.load("calibration_toml/current/settings.toml")
camera_matrix = np.array(calib_data["calibration"]["camera_matrix"])
dist_coeffs = np.array(calib_data["calibration"]["dist_coeffs"])
```

## Architecture

### Core Components

1. **Camera Interface**: Picamera2 (RPi) or cv2.VideoCapture (desktop) with configurable resolution (1200×800) and frame rates
2. **ArUco Detection**: OpenCV ArUco marker detection with pose estimation for marker IDs **[4, 8, 12, 14, 20]** using DICT_APRILTAG_36h11
3. **Coordinate Transformation**: 3D coordinate system transformations with marker-specific offsets (see [GameStream/stream_optimize_v2.py:22-28](GameStream/stream_optimize_v2.py#L22-L28))
4. **UDP Communication**: Real-time state-based streaming to Godot (localhost:8000)
5. **Data Recording**: Msgpack-based binary recording with GPIO pin 17 synchronization

### Key Modules

- **GameStream/** - Real-time tracking applications
  - `stream_optimize_v2.py` - **Production system** with state machine, reference frame persistence, CSV recording
  - `stream_april.py` - Legacy base implementation
  - `filters.py` - ExponentialMovingAverageFilter3D for position smoothing
  - `webcamera3.py` - Desktop webcam variant
- **recorder/** - 15+ data capture variants (msgpack format with timestamps)
- **notebooks/** - Jupyter notebooks for analysis + support libraries:
  - `ar_support.py` - Rotation matrix calculations (`calculate_rotmat`, `calculate_rotmat_from_xyo`)
  - `pd_support.py` - Pandas data processing utilities
  - `calibration/` - 20+ calibration workflow notebooks
- **calibration_programs/** - Calibration tools (`calibration_manager.py`, `simplified_fisheye_calibration.py`)
- **helper_programs/** - Testing utilities (GPIO, distance, file integrity)

### UDP State Machine (stream_optimize_v2.py)

The production tracking system uses explicit state management:

| State | Code | Description | Valid Commands |
|-------|------|-------------|----------------|
| **IDLE** | 0 | No reference frame | `CAPTURE_REF`, `LOAD_REF` |
| **REFERENCE_CAPTURED** | 1 | Reference saved, ready to track | `START_TRACK`, `USER:<id>`, `SAVE_REF` |
| **TRACKING** | 2 | Live tracking (no recording) | `STOP_TRACK`, `USER:<id>` |
| **RECORDING** | 3 | Tracking + CSV recording | `STOP_TRACK`, `CHANGE:<id>` |

**Commands**: ASCII text from Godot → Python
**Responses**: Binary float32 array `[state_code, x, y, z]` from Python → Godot

See [UDP_PROTOCOL.md](UDP_PROTOCOL.md) for complete specification.

### Data Storage Paths

**Runtime data** (created at runtime, not in git):
```
~/Documents/NOARK/
├── calibration/
│   └── calibration.json              # JSON-based calibration storage (CalibrationManager)
├── reference_frames/
│   └── reference_frame.json          # Persistent reference pose (auto-loaded on startup)
└── data/
    └── [patient_id]/
        └── Session-YYYY-MM-DD/
            └── MovementData/
                └── timestamp_data.csv  # CSV recording from RECORDING state
```

**Repository data** (gitignored):
```
./data/
├── recordings_160fov/               # Msgpack recordings
│   └── [recording_name]/
│       ├── webcam_color.msgpack    # Compressed frames
│       └── webcam_timestamp.msgpack # Timestamps with GPIO sync
└── calibration/                     # Msgpack calibration captures
```

**Calibration configurations** (checked into git):
```
./calibration_toml/
├── current/                         # Active production configs
│   ├── settings.toml               # RPi camera calibration
│   └── settings_webcam.toml        # USB webcam calibration
└── old/                            # Archived calibrations
```

## Camera Calibration Workflow

1. **Capture calibration data**:
   ```bash
   python recorder/calibration_capture.py
   # or
   python recorder/record_chessboard.py
   ```
   - Captures chessboard pattern frames → stored as msgpack binary data in `./data/calibration/`

2. **Process calibration** using notebooks:
   - Open notebooks in `notebooks/calibration/camera_calibration_*.ipynb`
   - Detects chessboard corners with `cv2.findChessboardCorners()`
   - Runs `cv2.fisheye.calibrate()` or standard calibration
   - Outputs: camera_matrix (3×3), dist_coeffs (k1-k5)

3. **Export to TOML**:
   - Save calibration to `calibration_toml/current/`
   - Different configurations for various FOV settings (120fov, 160fov)

4. **Use in stream applications**:
   ```bash
   python GameStream/stream_optimize_v2.py --calib calibration_toml/current/settings.toml
   ```

**Alternative**: Use `calibration_programs/simplified_fisheye_calibration.py` for automated fisheye calibration

## Marker Configuration and Offsets

Each ArUco marker has a unique 3D offset from its detected center (see [GameStream/stream_optimize_v2.py:22-28](GameStream/stream_optimize_v2.py#L22-L28)):

```python
MARKER_OFFSETS = {
    4: [0.00, 0.1, -0.069],      # Y offset: 10cm
    8: [0.00, 0.01, -0.069],     # Y offset: 1cm
    12: [0.00, 0.0, -0.1075],    # Z offset: deeper mounting
    14: [-0.09, 0.0, -0.069],    # X offset: -9cm
    20: [0.1, 0.0, -0.069],      # X offset: +10cm
}
```

These offsets are critical for multi-marker systems and must match the physical mounting bracket geometry.

## Important Notes

- **Platform**: Designed for Raspberry Pi with GPIO pin 17 for synchronization (also runs on desktop with webcams)
- **Dictionary**: ArUco markers use `DICT_APRILTAG_36h11` (AprilTag format)
- **Marker IDs**: Production system expects **[4, 8, 12, 14, 20]** (older code may use different IDs)
- **UDP Port**: localhost:8000 (configurable in Config class)
- **Coordinate System**: Right-handed, centroid-based, filtered with exponential moving average (alpha=0.2)
- **Reference Frame Persistence**: Reference pose auto-saved to `~/Documents/NOARK/reference_frames/reference_frame.json` and auto-loaded on startup
- **Recording Formats**:
  - Video/image data: msgpack binary format with separate timestamp files
  - Tracking data: CSV format with Time,X,Y,Z columns

## Testing Without Hardware

Use the example dataset for development without Raspberry Pi:

```bash
# Example dataset contains sample calibration and recording data
ls example_dataset/calibration/
ls example_dataset/recordings/

# Load and process in Jupyter notebooks
# See notebooks in notebooks/calibration/ and notebooks/analysis_*.ipynb
```

## Dependencies

Key Python packages (from [requirements.txt](requirements.txt)):
- `opencv-python` + `opencv-contrib-python` - ArUco detection and pose estimation
- `numpy` (1.26.4) - Matrix operations
- `picamera2` - Raspberry Pi camera interface
- `toml` - Configuration file parsing
- `msgpack` + `msgpack-numpy` - Binary data serialization
- `scipy` - Rotation matrices and interpolation
- `matplotlib` - Visualization
- `mediapipe` - Optional human pose estimation