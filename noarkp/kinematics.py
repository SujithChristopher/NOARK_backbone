import threading
import numpy as np
import math
import serial
from scipy.spatial.transform import Rotation as R
import time

MOTOR_L_TO_LP1 = 0.05  # Distance from motor L to left pulley 1
MOTOR_L_TO_LP2 = 0.05  # Distance from motor L to left pulley 2

MOTOR_R_TO_RP1 = 0.05  # Distance from motor R to right pulley 1
MOTOR_R_TO_RP2 = 0.05  # Distance from motor R to

class NoarkKinematics:
    def __init__(self, serial_port=None, baudrate=115200, debug=False):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.debug = debug
        if serial_port:
            self.ser = serial.Serial(serial_port, baudrate)

    def start_thread(self): threading.Thread(target=self.start_serial).start()

    def get_angles(self): return [self.enc1, self.enc2] # Returns the latest read angles
    
    def start_serial(self):
        
        while True:
            if not self.serial_port:
                raise ValueError("Serial port not initialized.")
            line = self.ser.readline().decode('utf-8').strip().split(",")
            
            self.enc1 = float(line[0].split(":")[1])
            self.enc2 = float(line[1].split(":")[1])
            
            if self.debug:
                print(f"Encoder 1: {self.enc1}, Encoder 2: {self.enc2}")
            
if __name__ == "__main__":
    
    kinematics = NoarkKinematics(serial_port='/dev/ttyACM0', baudrate=115200, debug=True)
    kinematics.start_thread()
    
            