import serial
import serial.tools.list_ports
import threading

from datetime import datetime

class TeensyPort:
    def __init__(self):
        self.ports = serial.tools.list_ports.comports()
        self.serialInst = serial.Serial()
        self.portsList = []
        self.use = None
        self.running = True
        self.e1 = 0
        self.e2 = 0
        # Find and list ports
        for port in self.ports:
            self.portsList.append(str(port))
            # print(self.portsList)
        
        # Find the Teensy port
        for i in range(len(self.portsList)):
            if self.portsList[i].startswith("/dev/ttyACM0"):
                self.use = "/dev/ttyACM0" or "/dev/ttyACM1"
                # print(f"Using port: {self.use}")
                break
        
        if self.use is None:
            raise Exception("Teensy port /dev/ttyACM0 not found!")
        
        # Configure serial port
        self.serialInst.baudrate = 115200
        self.serialInst.port = self.use  
        # self.serialInst.timeout = 0.05
    
    def read_serial(self):
        """Thread function to continuously read serial data"""
        while self.running:
            try:
                if self.serialInst.in_waiting > 0:
                    response = self.serialInst.readline()
                    self.encoder = response.decode('utf-8').strip()
                    # split by comma and filter out empty strings
                    values = [v for v in self.encoder.split(",") if v]
                    # self.enc1 = self.encoder.split(",")[0]
                    # self.enc2 = self.encoder.split(",")[1]
                    if len(values) >= 2:
                        self.enc1 = values[0]
                        self.enc2 = values[1]
                    # self.e1 = self.enc1.split(":")[1] 
                    # self.e2 = self.enc2.split(":")[1]
                    # print(f"e1: {self.enc1}, e2: {self.enc2}")
                else:
                    # print(f"Invalid data: {self.encoder}")
                    pass
                        
            except Exception as e:
                if self.running:
                    print(f"\r[ERROR] {e}")
    def start(self):
        """Start the serial communication"""
        try:
            self.serialInst.open()
            print(f"\nConnected to {self.use}")
                    
            # Start reading thread
            read_thread = threading.Thread(target=self.read_serial)
            read_thread.start()
         
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
            self.running = False
        
        except serial.SerialException as e:
            print(f"\n Serial Port Error: {e}")
            self.running = False
    
if __name__ == "__main__":
    teensy = TeensyPort()
    teensy.start()

