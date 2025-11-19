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
    def __init__(self, serial_port=None, baudrate=9600, debug=False):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.debug = debug
        if serial_port:
            self.ser = serial.Serial(serial_port, baudrate)
            time.sleep(2)  # Wait for the serial connection to initialize

            
    def read_angles(self):
        
        while True:
            if not self.serial_port:
                raise ValueError("Serial port not initialized.")
            line = self.ser.readline().decode('utf-8').strip().split(",")
            
            self.enc1 = line[0].split(":")[1]
            self.enc2 = line[1].split(":")[1]
            
            if self.debug:
                print(f"Encoder 1: {self.enc1}, Encoder 2: {self.enc2}")
            
if __name__ == "__main__":
    
    kinematics = NoarkKinematics(serial_port='COM3', baudrate=9600)
    kinematics.read_angles()
            