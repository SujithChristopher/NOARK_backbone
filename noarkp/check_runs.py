import math
from camera_init import RpiCamera
from noarkp.motor_encoder import MotorEncoder
from noarkp.kinematics import NoarkKinematics
import threading
import time

MOTOR_L_TO_LP1 = 0.05  # Center to Center distance from motor L to left pulley 1
MOTOR_L_TO_LP2 = 0.05  # Center to Center distance from motor L to left pulley 2

MOTOR_R_TO_RP1 = 0.05  # Center to Center distance from motor R to right pulley 1
MOTOR_R_TO_RP2 = 0.05  # Center to Center distance from motor R to right pulley 2

LP1_TO_LP2 = 0.10      # Distance between left pulleys
RP1_TO_RP2 = 0.10      # Distance between right pulleys

MOTOR_RADIUS = 0.01      # Radius of the motor pulley
PULLEY_RADIUS = 0.01    # Radius of the cable pulley




class AssessCableLength:
    def __init__(self):
        self.camera = RpiCamera(debug=True)
        self.camera.start_thread()
        time.sleep(0.5)  # Allow camera to initialize
        self.kinematics = NoarkKinematics(serial_port='/dev/ttyACM0', baudrate=115200, debug=True)
        self.kinematics.start_thread()
        time.sleep(0.5)  # Allow kinematics to initialize
        
    
    def get_camera_coordinates(self):
        rvec, tvec = self.camera.get_pose()
        return rvec, tvec
        
    def cal_initial_length_left(self):
        rvec, tvec = self.get_camera_coordinates()

        # For left motor
        t1, t2 = self.tangent_points(MOTOR_L_TO_LP1, MOTOR_L_TO_LP1)
        
 
        
    def cal_right_cable_length(self, rvec, tvec):
        # Calculate lengths from right motor to pulleys
        pass

    def cal_left_cable_length(self, rvec, tvec):
        # Calculate lengths from left motor to pulleys
        pass

    def run(self):
        
        pass
    
if __name__ == "__main__":
    assessor = AssessCableLength()
    assessor.calculate_initial_lengths()
    
    assessor.run()