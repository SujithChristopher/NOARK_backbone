import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Cam_Data_log import MainClass
from Enc_Data_log import TeensyPort
import numpy as np
from math import sqrt, degrees, pi, radians, acos
import sys
import termios
import tty
import keyboard
import os

BASE_DIR = r"E:\Ragav\MS Bio Engineering\NOARK_backbone\cam_enc_recordings"

class XYPlot:
    def __init__(self, save_dir):
        calib_path = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
        table_calibration_path = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"
        self.cam = MainClass(calib_path, table_calibration_path, save_dir=save_dir)
        self.enc = TeensyPort(save_dir=save_dir)
        self.enc.start()

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)

            while True:
                self.cam.process_frame()
                if keyboard.is_pressed('s'):
                    self.cam.trigger_cam = True
                    self.enc.trigger_sens = True
                    print("Recording started — camera + encoder CSV both active.")
                elif keyboard.is_pressed('z'):
                    self.enc.enc_reset = True
                    print("Encoder reset triggered")
                elif keyboard.is_pressed('q'):
                    self.cam.trigger_cam = False   # Stop camera CSV
                    self.enc.trigger_sens = False  # Stop encoder CSV
                    self.enc.running = False       # Stop serial thread
                    print("Recording stopped. Exiting...")
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def create_session_folder():
    """Prompt user for a folder name and create it inside BASE_DIR."""
    print("\n========================================")
    print("  NOARK Backbone — Data Recording Setup")
    print("========================================")
    print(f"  Base directory: {BASE_DIR}\n")

    while True:
        folder_name = input("  Enter session folder name (e.g. trial_01_walking): ").strip()
        if not folder_name:
            print("  [!] Folder name cannot be empty. Try again.")
            continue
        # Sanitize: replace spaces with underscores
        folder_name = folder_name.replace(" ", "_")
        save_dir = os.path.join(BASE_DIR, folder_name)
        if os.path.exists(save_dir):
            overwrite = input(f"  [!] Folder '{folder_name}' already exists. Use it anyway? (y/n): ").strip().lower()
            if overwrite == 'y':
                break
            else:
                continue
        else:
            os.makedirs(save_dir)
            print(f"\n  ✅ Created folder: {save_dir}")
            break

    print(f"  📁 Saving to: {save_dir}")
    print("========================================\n")
    return save_dir


if __name__ == "__main__":
    save_dir = create_session_folder()
    graph = XYPlot(save_dir=save_dir)
    try:
        graph.run()
    except KeyboardInterrupt:
        print("\nExiting...")
