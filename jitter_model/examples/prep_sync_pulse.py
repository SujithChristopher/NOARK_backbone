# %% Imports
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import msgpack as mp
import msgpack_numpy as mpn
import numpy as np
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '../..')
from support.pd_support import *
# %% Config
PROJECT_ROOT = NB_DIR.resolve().parents[1]
SCRIPT_DIR = NB_DIR.resolve().parent

RECORDING_NAME = "dual_160_tframe_july1"
RECORDING_DIR = PROJECT_ROOT / "data" / "trunk_july1_2026" / RECORDING_NAME
STEREO_TOML = (
    PROJECT_ROOT
    / "data" / "calibration" / "dual_160"
    / "calib_cz30_dual_v2" / "stereo_calibration.toml"
)
CHARUCO_TOML = (
    PROJECT_ROOT
    / "data" / "trunk_july1_2026"
    / "dual_160_tframe_july1" / "charuco_basis.toml"
)


# %%
