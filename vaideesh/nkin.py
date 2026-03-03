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
# from uncertainties import ufloat

# Parameters for kinematics 
ML = [-0.065,0,0.707]  # left Motor position in table frame  
MR = [0.065,0,0.707]   # right Motor position in table frame  
P1 = [-0.495,0,0.773]  # pulley 1 position 
P2 = [-0.495,0,0.027]  # pulley 2 position 
P3 = [0.495,0,0.773]   # pulley 3 position
P4 = [0.495,0,0.027]   # pulley 4 position
# ML_TO_P1 = 0.38  # Center to Center distance from motor L to left pulley 1 
# P1_TO_P2 = 0.62  # Distance from pulley 1 to pulley 2
Offset_t1 = 0.05  # offset from tabletop corner to motor pulley center 70mm
r_ML = r_MR = 0.033 # radius of motors in meters
r_p1 = r_p2 = r_p3 = r_p4 = 0.01  # radius of pulleys in meters




class XYPlot:
    def __init__(self):
        calib_path = "/home/sujith/Documents/NOARK_backbone/old_calibration/fisheye.toml"
        table_calibration_path = "/home/sujith/Documents/NOARK_backbone/vaideesh/table_frame_data.toml"
        self.cam = MainClass(calib_path, table_calibration_path)
        self.enc = TeensyPort()
        self.enc.start()
        self.last_key = None

        self.enc1_offset = 0.0
        self.enc2_offset = 0.0 
    
        self.start_time = time.time()
   
    def parse_encoder_value(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)

            while True:
                # Get current time
                # current_time = time.time() - self.start_time
                # Data after processing
                self.cam.process_frame()
                N_Pos = self.cam.noark_in_table_frame # Noark position in table frame 
                print(N_Pos)
                # Reset encoder to zero
            
                raw_e1 = self.parse_encoder_value(self.enc.enc1)  # Encoder value from left motor pulley
                raw_e2 = self.parse_encoder_value(self.enc.enc2)  # Encoder value from right motor pulley
                # print("rawe1 , rawe2 :", raw_e1,raw_e2)

                key = self.get_key()
                if key == 'r':
                    self.enc1_offset += enc_value_1 
                
                if key == 't':
                    self.enc2_offset += enc_value_2

                if key == 'z':
                    self.enc1_offset = raw_e1
                    self.enc2_offset = raw_e2
                
                enc_value_1 = round(raw_e1 - self.enc1_offset, 2)
                enc_value_2 = round(raw_e2 - self.enc2_offset, 2)
                print("E1 , E2 :",enc_value_1, enc_value_2)

                # Distance calculation
                L_motor_to_P1 = sqrt((P1[0] - ML[0])**2 + (P1[2] - ML[2])**2)
                P1_to_P2 = sqrt((P2[0] - P1[0])**2 + (P2[2] - P1[2])**2)
                P2_to_NOARK = sqrt((N_Pos[0] - P2[0])**2 + (N_Pos[2] - P2[2])**2)
                print("L_motor_to_P1(cm):",(L_motor_to_P1 * 100))
                # print("P1_to_P2 (cm):",(P1_to_P2 * 100))
                # print("Pulley 2 to NOARK(cm)",P2_to_NOARK * 100)
                # print("Enc1 :",enc_value_1)
                R_motor_to_P3 = sqrt((P3[0] - MR[0])**2 + (P3[2] - MR[2])**2)
                P3_to_P4 = sqrt((P4[0] - P3[0])**2 + (P4[2] - P3[2])**2)
                P4_to_NOARK = sqrt((N_Pos[0] - P4[0])**2 + (N_Pos[2] - P4[2])**2)
                # print("R_motor_to_P3(cm):",(R_motor_to_P3 * 100))
                # print("P3_to_P4 (cm):",(P3_to_P4 *  100))
                # print("Pulley 4 to NOARK(cm)",P4_to_NOARK * 100)
                # print("Enc2 :",enc_value_2)

                # Cable wound around the the spool 
                delta_L = r_ML * np.deg2rad(enc_value_1)
                # print("Left_Cable wrapped:",delta_L * 100)
                delta_R = -(r_MR * np.deg2rad(enc_value_2))
                # print("Right_Cable wrapped:",round(delta_R * 100,2))

                # Total cable length calculation
                # For left
                L1 = P2_to_NOARK + delta_L 
                
                # Tot_len_cable = L_motor_to_P1 + P1_to_P2 + P2_to_NOARK 
                # # Tot_len_cable_enc1 = L_motor_to_P1 + P1_to_P2 + delta_L
                # print("left_Length_pos(cm):", Tot_len_cable * 100)
                # print("left_Length_enc(cm)", Tot_len_cable_enc1 * 100)

                # For Right 
                # R_tot_len_cable = R_motor_to_P3 + P3_to_P4 + P4_to_NOARK 
                # R_tot_len_cable_enc2 = R_motor_to_P3 + P3_to_P4 + Right_cable_wrapped
                # print("right_Total length(cm):", R_tot_len_cable * 100)
                # print("right_Length_cable_enc2(cm)",round(R_tot_len_cable_enc2 * 100,2))

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
                # print("cl(cm):",cl * 100)
                # print("Theta1(deg):",degrees(theta1))
                
                #To find theta2 from second right angle triangle
                l1 = sqrt((P2_to_NOARK)**2 - (r_p2)**2) # length from tangent to NOARK 
                # print("l1(cm):",l1 * 100)
                theta2 = np.arccos(l1 / P2_to_NOARK)
                theta2_a = (1.570 - theta2) # theta 2 dash is found using theta 2 refer notes for clarification
                # print("Theta2(deg):",degrees(theta2_a))
                sigma = (theta2_a - theta1)
                # print("Sigma :",degrees(sigma)) 
                phi_2 = (theta2 + theta1)
                # print('phi2', degrees(phi_2))
                omega2 = (1.570 + phi_2)
                print("omega2",degrees(omega2))



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
                # print("l2(cm):",l2 * 100)
               
                # print("cr",cr * 100)
                theta4 = np.arccos(l2/P4_to_NOARK) 
                # print("theta4", theta4)
                theta4a = (1.570 -theta4) # theta 4 dash is found using theta 2 refer notes for clarification
                sigma2 = (theta4a - theta3) 
                # print("Sigma 2 :",degrees(sigma2)) 
                phi_4 = -(theta4 + theta3)
                # print("phi4",degrees(phi_4))
                # omega4 = -((1.570) + (phi_4))
                omega4 = -3.14 + sigma2
                print("omega4",degrees(omega4))


                # print("NOARK Position in table frame(cm):",N_Pos)
                # print(L_wrapped * 100)
                # L1 = (enc_value_1 * (2 * np.pi * r_p1))  # Length of cable from motor L to pulley 1
                # L2 = L1 + (enc_value_2 * (2 * np.pi * r_p2))  # Length of cable from pulley 1 to pulley 2       
                # try:
                    # X_calc = (ML_TO_P1**2 - P1_TO_P2**2 + L2**2) / (2 * ML_TO_P1)
                    # Y_calc = sqrt(abs(L2**2 - X_calc**2))
                    # print(f"Kinematics Position - X: {X_calc*100:.2f} cm, Y: {Y_calc*100:.2f} cm")
                # except ValueError as e:
                    # print(f"Error in kinematics calculation: {e}")
                # time.sleep(0.1)         

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)




if __name__ == "__main__":
    graph = XYPlot()
    try:
        graph.run()

        
    except KeyboardInterrupt:
        print("\nExiting...")
        graph.stop() 