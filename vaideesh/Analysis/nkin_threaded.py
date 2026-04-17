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

class XYPlot:
    def __init__(self):
        calib_path = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
        table_calibration_path = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"
        self.cam = MainClass(calib_path, table_calibration_path)
        self.enc = TeensyPort()
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


if __name__ == "__main__":
    graph = XYPlot()
    try:
        graph.run()
    except KeyboardInterrupt:
        print("\nExiting...")