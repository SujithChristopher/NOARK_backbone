#!/usr/bin/env python3
"""
Launcher script for Chessboard Calibration Recorder GUI.

Run: python run.py
Or:  ./run.py
"""

import sys
from pathlib import Path

# Add current directory to path for demo_mode import
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from chessboard_recorder_gui import main

if __name__ == "__main__":
    main()
