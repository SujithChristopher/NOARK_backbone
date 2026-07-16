# UDP Communication Protocol for stream_optimize_v2.py

## Overview

This document describes the UDP communication protocol between the Python ArUco tracking system and Godot. The v2 version provides explicit control over reference frame capture and state management.

## Connection Details

- **IP**: localhost (127.0.0.1)
- **Port**: 8000
- **Protocol**: UDP
- **Message Format**: ASCII text (commands) / Binary float32 array (responses)

## State Machine

The system operates in four distinct states:

1. **IDLE** (0)
   - Initial state
   - No reference frame captured
   - Waiting for `CAPTURE_REF` command

2. **REFERENCE_CAPTURED** (1)
   - Reference frame has been saved
   - Ready to start tracking
   - Can accept `START_TRACK` or `USER:` commands

3. **TRACKING** (2)
   - Actively tracking and sending coordinates
   - Coordinates are relative to reference frame
   - No data recording

4. **RECORDING** (3)
   - Tracking AND recording to CSV file
   - Associated with a hospital/user ID
   - Data saved to disk

## Commands (Godot → Python)

All commands are sent as ASCII bytes.

### Core Commands

| Command | Description | Valid States | Next State |
|---------|-------------|--------------|------------|
| `CAPTURE_REF` | Capture current pose as reference frame (auto-saves to disk) | IDLE, REFERENCE_CAPTURED | REFERENCE_CAPTURED |
| `SAVE_REF` | Manually save current reference frame to disk | REFERENCE_CAPTURED, TRACKING, RECORDING | (no change) |
| `LOAD_REF` | Load previously saved reference frame from disk | IDLE | REFERENCE_CAPTURED |
| `RESET_REF` | Clear reference frame and return to IDLE | Any | IDLE |
| `START_TRACK` | Start tracking (requires reference) | REFERENCE_CAPTURED | TRACKING |
| `STOP_TRACK` | Stop tracking | TRACKING, RECORDING | REFERENCE_CAPTURED |
| `STATUS` | Request current state | Any | (no change) |
| `STOP` | Stop everything and exit program | Any | Exit |

### User/Recording Commands

| Command | Format | Description | Valid States | Next State |
|---------|--------|-------------|--------------|------------|
| `USER:<id>` | `USER:patient123` | Set hospital ID and start recording | REFERENCE_CAPTURED, TRACKING | RECORDING |
| `CHANGE:<id>` | `CHANGE:patient456` | Change to new user/close previous file | RECORDING | RECORDING |

### Example Command Sequences

#### First Time Setup (One-Time Calibration)
```python
# Godot sends commands as bytes
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
python_address = ("localhost", 8000)

# Position patient at starting point, then:
sock.sendto(b"CAPTURE_REF", python_address)  # Saves to disk automatically

# Now you can start tracking
sock.sendto(b"START_TRACK", python_address)
sock.sendto(b"USER:patient_001", python_address)
```

#### Daily Usage (Reference Auto-Loaded)
```python
# The reference frame is automatically loaded when the program starts!
# You can start tracking immediately:

sock.sendto(b"START_TRACK", python_address)  # Uses saved reference
sock.sendto(b"USER:patient_001", python_address)

# ... patient exercises ...

sock.sendto(b"STOP_TRACK", python_address)

# Next patient
sock.sendto(b"START_TRACK", python_address)
sock.sendto(b"USER:patient_002", python_address)
```

#### Recalibrating (Changing Starting Position)
```python
# If you need to change the reference position:
sock.sendto(b"CAPTURE_REF", python_address)  # Overwrites old reference

# Continue as normal
sock.sendto(b"START_TRACK", python_address)
```

## Responses (Python → Godot)

Python sends binary data as float32 array:

```
[state_code, x, y, z]
```

### State Codes

| Code | State | Meaning |
|------|-------|---------|
| 0.0 | IDLE | No reference frame, waiting |
| 1.0 | REFERENCE_CAPTURED | Reference set, ready to track |
| 2.0 | TRACKING | Actively tracking |
| 3.0 | RECORDING | Tracking and recording |
| -1.0 | ERROR | Error occurred (check console) |
| -99.0 | STOP | Shutdown signal |

### Receiving Data in Godot (GDScript)

```gdscript
extends Node

var udp := PacketPeerUDP.new()
var python_port := 8000

func _ready():
    # Listen for responses from Python
    udp.bind(python_port + 1)  # Use different port for receiving

func send_command(command: String):
    var packet = command.to_utf8_buffer()
    udp.set_dest_address("127.0.0.1", python_port)
    udp.put_packet(packet)

func _process(delta):
    if udp.get_available_packet_count() > 0:
        var bytes = udp.get_packet()
        var data = parse_float_array(bytes)

        var state_code = data[0]
        var position = Vector3(data[1], data[2], data[3])

        match state_code:
            0.0:
                print("Python is IDLE")
            1.0:
                print("Reference captured")
            2.0:
                print("Tracking: ", position)
            3.0:
                print("Recording: ", position)
                update_player_position(position)
            -1.0:
                print("ERROR from Python")
            -99.0:
                print("Python stopped")

func parse_float_array(bytes: PackedByteArray) -> Array:
    var result = []
    for i in range(0, bytes.size(), 4):
        result.append(bytes.decode_float(i))
    return result
```

## Workflow Examples

### Basic Tracking Workflow

1. **System starts** → State: IDLE
2. **Position patient at starting point**
3. **Send `CAPTURE_REF`** → State: REFERENCE_CAPTURED
4. **Send `START_TRACK`** → State: TRACKING
   - Python now sends coordinates continuously
5. **Send `STOP_TRACK`** when done → State: REFERENCE_CAPTURED
6. **Send `RESET_REF`** to prepare for next session → State: IDLE

### Recording Workflow

1. **System starts** → State: IDLE
2. **Send `CAPTURE_REF`** → State: REFERENCE_CAPTURED
3. **Send `USER:patient_id`** → State: RECORDING
   - Python sends coordinates AND saves to CSV
   - File: `~/Documents/NOARK/data/patient_id/Session-YYYY-MM-DD/MovementData/timestamp_data.csv`
4. **Send `STOP_TRACK`** when done → State: REFERENCE_CAPTURED
   - CSV file closed automatically

### Multi-Patient Session

```python
# Patient 1
send("CAPTURE_REF")
send("USER:patient_001")
# ... exercise ...
send("STOP_TRACK")

# Change to Patient 2 (uses same reference frame)
send("CHANGE:patient_002")
# ... exercise ...
send("STOP_TRACK")

# Reset for new session with different starting position
send("RESET_REF")
```

## Reference Frame Storage

### Automatic Behavior

**Reference frames are persistent across program restarts:**

1. **When you send `CAPTURE_REF`**: The reference frame is automatically saved to disk
2. **When the program starts**: It automatically loads the saved reference frame if it exists
3. **Result**: You only need to capture the reference frame once, and it will be used for all future sessions

### Storage Location

Reference frames are automatically saved to:
- **Windows**: `C:\Users\<username>\Documents\NOARK\reference_frames\reference_frame.json`
- **Linux (RPi)**: `/home/<username>/Documents/NOARK/reference_frames/reference_frame.json`

### File Format

```json
{
  "ids": [[12], [88], [89]],
  "rvecs": [[0.123, -0.456, 0.789], ...],
  "tvecs": [[0.012, 0.034, 0.567], ...],
  "timestamp": "2025-10-21 14:30:45"
}
```

### When to Use Each Command

- **CAPTURE_REF**: Use when you want to set a NEW reference position (overwrites saved reference)
- **LOAD_REF**: Manually reload from disk (rarely needed since it auto-loads on startup)
- **SAVE_REF**: Manually save current reference (rarely needed since CAPTURE_REF auto-saves)

**Typical workflow:**
1. First time setup: Send `CAPTURE_REF` to set the starting position
2. From then on: Just start the program and it will use the saved reference automatically
3. If you need to change the starting position: Send `CAPTURE_REF` again

## Error Handling

### Common Errors

1. **"Cannot start tracking without reference frame"**
   - Cause: Tried `START_TRACK` or `USER:` before `CAPTURE_REF` or `LOAD_REF`
   - Solution: Send `CAPTURE_REF` or `LOAD_REF` first

2. **"No markers detected, cannot capture reference frame"**
   - Cause: No ArUco markers visible when `CAPTURE_REF` was sent
   - Solution: Ensure markers are visible in camera view

3. **"No saved reference frame found"**
   - Cause: Tried `LOAD_REF` but no saved reference frame exists
   - Solution: Use `CAPTURE_REF` first to create a reference frame


## Visual Feedback

The Python application displays status on the video window:

- **Red text "IDLE - Waiting for CAPTURE_REF"** → Need to capture reference
- **Green text "REF CAPTURED - Ready to track"** → Reference set, ready
- **Green text "TRACKING"** → Actively tracking
- **Green text "RECORDING"** → Recording data to file

## Data Format

### CSV Recording Format

When in RECORDING state, data is saved as:

```csv
Time,X,Y,Z
21/10/2025 14:23:45,0.123,-0.045,0.678
21/10/2025 14:23:45,0.125,-0.044,0.679
...
```

### Coordinate System

- **Origin**: Centroid of marker offsets at reference frame
- **Axes**: Based on first detected marker's orientation at reference
- **Units**: Meters
- **Filtering**: Exponential moving average (alpha=0.4)

## Configuration

Key parameters in `Config` class ([stream_optimize_v2.py:14-28](stream_optimize_v2.py#L14-L28)):

```python
FRAME_SIZE = (1200, 800)       # Camera resolution
MARKER_LENGTH = 0.05           # Marker size in meters
UDP_IP = "localhost"
UDP_PORT = 8000
DEFAULT_IDS = [4, 8, 12, 14, 20]  # Expected marker IDs
ALPHA = 0.4                    # Filter smoothing (0=no filter, 1=no smoothing)
```

## Differences from Original Version

### stream_optimize.py (Original)
- Reference frame captured automatically on first detection
- No explicit state management
- Limited command set
- Recording starts immediately with USER command
- No visual status feedback

### stream_optimize_v2.py (New)
- ✅ Explicit reference frame capture via `CAPTURE_REF`
- ✅ Clear state machine (IDLE → REF_CAPTURED → TRACKING → RECORDING)
- ✅ More commands for fine-grained control
- ✅ Separate tracking and recording states
- ✅ Visual status overlay on video
- ✅ Better error handling and feedback
- ✅ Proper CSV file closing
- ✅ Status query command
- ✅ Can reset reference frame without restarting

## Troubleshooting

### Python not receiving commands
- Check firewall settings
- Verify port 8000 is not in use: `netstat -an | findstr 8000`
- Ensure both Python and Godot use same IP/port

### Coordinates seem wrong
- Verify reference frame was captured at correct position
- Check marker IDs match `Config.DEFAULT_IDS`
- Verify marker offsets in `Config.MARKER_OFFSETS` are correct for your setup

### Recording file not created
- Check directory exists: `~/Documents/NOARK/data/`
- Verify write permissions
- Look for error messages in console

## Testing Without Godot

You can test the Python application using netcat or a simple Python script:

```python
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("localhost", 8000)

# Send commands
time.sleep(1)
sock.sendto(b"STATUS", addr)
time.sleep(1)
sock.sendto(b"CAPTURE_REF", addr)
time.sleep(1)
sock.sendto(b"START_TRACK", addr)
time.sleep(5)
sock.sendto(b"STOP", addr)
```
