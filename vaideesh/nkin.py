import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from camera_pose import MainClass
from pyteensy import TeensyPort
import time
import numpy as np
from math import sqrt,degrees,pi,radians ,acos
import sys 
import select
import termios 
import tty 
import csv
from datetime import datetime
import gpiod
import msgpack as mp
import msgpack_numpy as mpn

# from uncertainties import ufloat

# Parameters for kinematics 
ML = [-0.065,0,-0.707]  # left Motor position in table frame  
MR = [0.065,0,-0.707]   # right Motor position in table frame  
P1 = [-0.495,0,-0.773]  # pulley 1 position 
P2 = [-0.495,0,-0.027]  # pulley 2 position 
P3 = [0.495,0,-0.773]   # pulley 3 position
P4 = [0.495,0,-0.027]   # pulley 4 position
# ML_TO_P1 = 0.38  # Center to Center distance from motor L to left pulley 1 
# P1_TO_P2 = 0.62  # Distance from pulley 1 to pulley 2
Offset_t1 = 0.05  # offset from tabletop corner to motor pulley center 70mm
r_ML = r_MR = 0.033 # radius of motors in meters
r_p1 = r_p2 = r_p3 = r_p4 = 0.01  # radius of pulleys in meters




class XYPlot:
    def __init__(self):
        calib_path = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
        table_calibration_path = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"
        self.cam = MainClass(calib_path, table_calibration_path)
        self.enc = TeensyPort()
        self.enc.start()
        self.last_key = None

        self.enc1_offset = 0.0
        self.enc2_offset = 0.0 
        self.P2_to_NOARK_init = None
        self.P4_to_NOARK_init = None
        self.start_time = time.time()

        # Sync Pin Setup
        sync_pin = 17
        chip = gpiod.Chip("gpiochip4")
        self.sync_line = chip.get_line(sync_pin)
        self.sync_line.request(consumer="SyncPin", type=gpiod.LINE_REQ_DIR_IN)
        #Recording state
        self.start_recording = False

        #CSV setup 
        self.csv_path = "camera_data.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            "timestamp", "sync_pin",
            "marker_id",
            "tvec_x", "tvec_y", "tvec_z",
            "rvec_x", "rvec_y", "rvec_z"
        ])
        # Sensor CSV setup
        self.sensor_csv_path = "sensor_data.csv"
        with open(self.sensor_csv_path, "w", newline="") as f:
            writer_sen = csv.writer(f)
            writer_sen.writerow(["timestamp", "enc1", "enc2"])

    def parse_encoder_value(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    def get_key(self):
        # Check if there is data waiting in the buffer
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    def _write_frame(self, timestamp):
        sync = self.sync_line.get_value()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for marker in [self.cam.marker_12, self.cam.marker_14, self.cam.marker_20]:
                if marker["tvec"] is not None and marker["rvec"] is not None:
                    tx, ty, tz = float(marker["tvec"][0]), float(marker["tvec"][1]), float(marker["tvec"][2])
                    rx, ry, rz = float(marker["rvec"][0]), float(marker["rvec"][1]), float(marker["rvec"][2])
                else:
                    tx = ty = tz = float("nan")
                    rx = ry = rz = float("nan")

                writer.writerow([
                    timestamp, sync,
                    marker["id"],
                    tx, ty, tz,
                    rx, ry, rz
                ])
    def _write_sensor_frame(self, timestamp, enc1, enc2):
        with open(self.sensor_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, enc1, enc2])
    def _close_files(self):
        print("CSV saved and closed.")
    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)

            while True:
                self.cam.process_frame()
                N_Pos = self.cam.noark_in_table_frame  # Noark position in table frame 
                  # --- Write to CSV ---
                if self.start_recording:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    self._write_frame(timestamp)
                
                if N_Pos is None:
                    print("Camera: Searching for NOARK....")
                    time.sleep(0.1)
                    continue
                print(N_Pos * 100)
                # Reset encoder to zero at the start of the program
                raw_e1 = self.parse_encoder_value(self.enc.enc1)  # Encoder value from left motor pulley
                raw_e2 = self.parse_encoder_value(self.enc.enc2)  # Encoder value from right motor pulley
                enc_value_1 = round(raw_e1 - self.enc1_offset, 2)
                enc_value_2 = round(raw_e2 - self.enc2_offset, 2)
                print("E1 , E2 :",enc_value_1, enc_value_2)
                # --- Write sensor data ---
                if self.start_recording:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    self._write_sensor_frame(timestamp, enc_value_1, enc_value_2) 
                # Use a loop to catch the key even if it was pressed 
                key = self.get_key()
                if key == 's':  
                    self.start_recording = True
                    print("Recording started...")
                elif key == 'q':
                    print("Recording stopped. Exiting...")
                    break
                if key == 'r':
                    self.enc1_offset += enc_value_1 
                    print("Reset R")
                elif key == 't':
                    self.enc2_offset += enc_value_2
                    print("Reset T")
                elif key == 'z':
                    print(f"BEFORE zero: enc_value_1={enc_value_1}, enc_value_2={enc_value_2}")
                    self.enc1_offset = raw_e1
                    self.enc2_offset = raw_e2
                    # Recompute enc_values immediately after zeroing
                    enc_value_1 = round(raw_e1 - self.enc1_offset, 2)  # now = 0.0
                    enc_value_2 = round(raw_e2 - self.enc2_offset, 2)  # now = 0.0
                    # print(f"AFTER zero: enc_value_1={enc_value_1}, enc_value_2={enc_value_2}")  # must be 0.0

                    # Capture initial free lengths from camera at same moment
                    self.P2_to_NOARK_init = sqrt((N_Pos[0] - P2[0])**2 + (N_Pos[2] - P2[2])**2)
                    # print(f"Captured P2_to_NOARK_init: {self.P2_to_NOARK_init*100:.2f}cm")
                    self.P4_to_NOARK_init = sqrt((N_Pos[0] - P4[0])**2 + (N_Pos[2] - P4[2])**2)
                    print("Zeroed All")
                #math before zeroing
                # if self.P2_to_NOARK_init is not None and self.P4_to_NOARK_init is not None:
                #     print(f"Initial L_free_L: {self.P2_to_NOARK_init*100:.2f}cm | Initial L_free_R: {self.P4_to_NOARK_init*100:.2f}cm")
                #     continue


                # Distance calculation
                L_motor_to_P1 = sqrt((P1[0] - ML[0])**2 + (P1[2] - ML[2])**2)
                # print("L_motor_to_P1(cm):",(L_motor_to_P1 * 100))
                P1_to_P2 = sqrt((P2[0] - P1[0])**2 + (P2[2] - P1[2])**2)
                # print("P1-P2",P1_to_P2)
                P2_to_NOARK = sqrt((N_Pos[0] - P2[0])**2 + (N_Pos[2] - P2[2])**2)
                # print("L_motor_to_P1(cm):",(L_motor_to_P1 * 100))
               
                R_motor_to_P3 = sqrt((P3[0] - MR[0])**2 + (P3[2] - MR[2])**2)
                P3_to_P4 = sqrt((P4[0] - P3[0])**2 + (P4[2] - P3[2])**2)
                P4_to_NOARK = sqrt((N_Pos[0] - P4[0])**2 + (N_Pos[2] - P4[2])**2)
               
                # From camera (ground truth)
                L_cam_L = sqrt((N_Pos[0] - P2[0])**2 + (N_Pos[2] - P2[2])**2)
                L_cam_R = sqrt((N_Pos[0] - P4[0])**2 + (N_Pos[2] - P4[2])**2)
                # Cable wound around the the spool 
                if self.P2_to_NOARK_init is not None and self.P4_to_NOARK_init is not None:
                    delta_L = r_ML * np.deg2rad(enc_value_1)
                    delta_R = r_MR * np.deg2rad(enc_value_2)
                    #enc1 positive --> ML unwinds --> L_free_L increases --> ADD delta_L
                    L_free_L = self.P2_to_NOARK_init + delta_L #Left cable wound
                    #enc2 positive --> MR winds --> L_free_R decreases --> SUBTRACT delta_R
                    L_free_R = self.P4_to_NOARK_init - delta_R #right cable wound
                    
                    print(f"delta_L: {delta_L*100:.2f}cm  delta_R: {delta_R*100:.2f}cm")
                    # print(f"L_free_L (enc): {L_free_L*100:.2f}cm | L_cam_L: {L_cam_L*100:.2f}cm | error: {(L_free_L-L_cam_L)*100:.2f}cm")
                    # print(f"L_free_R (enc): {L_free_R*100:.2f}cm | L_cam_R: {L_cam_R*100:.2f}cm | error: {(L_free_R-L_cam_R)*100:.2f}cm")
                #Circle - Circle Intersection 
                #NOARK lies on circle centered at P2 with radius L_free_L and circle centered at P4 with radius L_free_R

                    x_P2, z_P2 = P2[0], P2[2]
                    x_P4, z_P4 = P4[0], P4[2]

                    #Distance between P2 and P4 
                    d = sqrt((x_P4 - x_P2)**2 + (z_P4 - z_P2)**2)

                    #Circle-Circle intersection formula 
                    a_val = (L_free_L**2 - L_free_R**2 + d**2) / (2 * d)
                    h_sq = (L_free_L**2 - a_val**2)
                    print(f"d={d*100:.2f}cm  L_free_L={L_free_L*100:.2f}cm  L_free_R={L_free_R*100:.2f}cm")
                    print(f"a_val={a_val*100:.2f}cm  h_sq={h_sq:.6f}")
                    print(f"L_free_L + L_free_R={(L_free_L+L_free_R)*100:.2f}cm  vs  d={d*100:.2f}cm")
                    print("h_sq check (must be >=0 for intersection):", h_sq)
                    if h_sq < 0:
                        print("NO INTERSECTION — skipping this frame")
                        time.sleep(0.1)
                        continue

                    h = sqrt(h_sq)

                    #Midpoint along P2-P4 
                    x_mid = x_P2 + a_val * (x_P4 - x_P2) / d
                    z_mid = z_P2 + a_val * (z_P4 - z_P2) / d 
                    print(f"Footpoint: x={x_mid*100:.2f}cm  z={z_mid*100:.2f}cm")

                    #Two possible intersection points
                    x_enc_1 = x_mid + h * (z_P4 - z_P2) / d
                    z_enc_1 = z_mid - h * (x_P4 - x_P2) / d #Above Baseline

                    x_enc_2 = x_mid - h * (z_P4 - z_P2) / d
                    z_enc_2 = z_mid + h * (x_P4 - x_P2) / d #below baseline
                    
                    # Pick correct point — NOARK is always between P2 and P4 in Z
                    # Choose point closer to current camera position as sanity check
                    # dist1 = sqrt((x_enc_1 - N_Pos[0])**2 + (z_enc_1 - N_Pos[2])**2)
                    # dist2 = sqrt((x_enc_2 - N_Pos[0])**2 + (z_enc_2 - N_Pos[2])**2)
                    # print("dist1:", dist1*100, "dist2:", dist2*100)

                    # if dist1 < dist2:
                        # x_noark_enc, z_noark_enc = x_enc_1, z_enc_1
                    # else:
                        # x_noark_enc, z_noark_enc = x_enc_2, z_enc_2
                    # Only use the solution that puts NOARK below the baseline
                    x_noark_enc = x_enc_1
                    z_noark_enc = z_enc_1
                    print(f"NOARK (encoder): x={x_noark_enc*100:.2f}cm  z={z_noark_enc*100:.2f}cm")
                    print(f"NOARK (camera):  x={N_Pos[0]*100:.2f}cm  z={N_Pos[2]*100:.2f}cm")
                    print(f"Position error:  x={(x_noark_enc - N_Pos[0])*100:.2f}cm  z={(z_noark_enc - N_Pos[2])*100:.2f}cm")
                else:
                    print("Press 'z' to initialize!")
                # Kinematics Calculation:
                ##LEFT ARM
                #To find theta 1 from first right angle triangle for left arm 
                c = (N_Pos[0],P2[2]) # opposite side of the triangle 
                x2 = sqrt((c[0]- P2[0])**2 + (c[1] - P2[2])**2) #distance between pulley 2 and adjacent side of the triangle
                # print("x2(cm):",x2 * 100)
                a = sqrt((c[0] - N_Pos[0])**2 + (c[1] - N_Pos[2])**2) #distance between NOARK and opp side of the triangle
                # a = sqrt((P2_to_NOARK)**2 - (x2)**2) #distance between NOARK and opp side of the triangle
                # print("a(cm):",a)
                cl = sqrt(a**2 + x2**2) #Distance between Pulley center to NOARK Center
                # print("cl(cm):",cl * 100)
                theta1 = np.arccos((x2)/(cl))
                l1 = sqrt((P2_to_NOARK)**2 - (r_p2)**2) # length from tangent to NOARK 
                # print("l1(cm):",l1 * 100)
                theta2 = np.arccos(l1 / P2_to_NOARK)
                theta2_a = np.pi/2 - theta2 # theta 2 dash is found using theta 2 refer notes for clarification
                # print("Theta2(deg):",degrees(theta2_a))
                sigma = (theta2_a - theta1)
                # print("Sigma :",degrees(sigma)) 
                phi_2 = (theta2 + theta1)
                # print('phi2', degrees(phi_2))
                omega2 = (1.5708 + phi_2)
                L_wrap_length = r_p2 * np.deg2rad(degrees(omega2)) # length of cable wrapped around the pulley
                # print("omega2",degrees(omega2))

                ##RIGHT ARM 
                #To find theta 3 from first right angle triangle for right arm 
                c2 = (N_Pos[0],P4[2])
                x21 = sqrt(((c2[0]) - (P4[0]))**2 + ((c2[1]) - (P4[2]))**2)
                # print("x21",x21)
                # a = sqrt((c[0] - N_Pos[0])**2 + (c[1] - N_Pos[2])**2)
                
                a1 = sqrt(((c2[0]) - (N_Pos[0]))**2 + ((c2[1]) - (N_Pos[2]) )**2)
                # print("a1",a1)
                cr = sqrt(a1**2 + x21 ** 2) # Distance between Pulley center to NOARK Center
                theta3 = np.arccos((x21)/(cr))
                # print("theta3", theta3)

                #To find theta2 from second right angle triangle
                l2 = sqrt((P4_to_NOARK)**2 - (r_p4)**2) # length from tangent to NOARK 
                print("l2(cm):",l2 * 100)
               
                # print("cr",cr * 100)
                theta4 = np.arccos(l2/P4_to_NOARK) 
                theta4a = np.pi/2 -theta4 # theta 4 dash is found using theta 2 refer notes for clarification
                sigma2 = (theta4a - theta3) 
                phi_4 = -(theta4 + theta3)
                omega4 = -3.14 + sigma2
             
                # time.sleep(0.1)         

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._close_files()




if __name__ == "__main__":
    graph = XYPlot()
    try:
        graph.run()

        
    except KeyboardInterrupt:
        print("\nExiting...")
        graph.stop() 