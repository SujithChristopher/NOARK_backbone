import threading
import numpy as np
import math
import serial
from scipy.spatial.transform import Rotation as R
import time

class MotorEncoder:
    def __init__(self, serial_port=None, baudrate=115200, debug=False):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.debug = debug
        if serial_port:
            self.ser = serial.Serial(serial_port, baudrate)

    def start_thread(self): 
        _thread = threading.Thread(target=self.start_serial)
        _thread.start()
        _thread.join()

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
    
    encoder = MotorEncoder(serial_port='/dev/ttyACM0', baudrate=115200, debug=True)
    encoder.start_thread()
    
            