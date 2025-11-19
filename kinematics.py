import numpy as np
import math
import serial
from scipy.spatial.transform import Rotation as R
import time


from picamera2 import Picamera2

class NoarkKinematics:
    def __init__(self, link_lengths, serial_port=None, baudrate=9600):
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        self.serial_port = serial_port
        self.baudrate = baudrate
        if serial_port:
            self.ser = serial.Serial(serial_port, baudrate)
            time.sleep(2)  # Wait for the serial connection to initialize
            
    def init_camera(self):
        self.camera = Picamera2()
        self.camera.start()
            
    def read_angles(self):
        if not self.serial_port:
            raise ValueError("Serial port not initialized.")
        line = self.ser.readline().decode('utf-8').strip().split(",")
        
        self.enc1 = line[0].split(":")[1]
        self.enc2 = line[1].split(":")[1]

        return [self.enc1, self.enc2]
            
    def motor_left(self, angle):
        pass
    
    def motor_right(self, angle):
        pass
            