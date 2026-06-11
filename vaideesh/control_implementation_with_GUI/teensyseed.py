import csv
import serial
import serial.tools.list_ports
import threading
from datetime import datetime
import time

class SeeduinoPort:
    """Reads load cell X/Y force data from the Seeeduino over serial."""
    def __init__(self, port="/dev/ttyACM1", baud=115200):
        self.serialInst = serial.Serial()
        self.serialInst.port = port
        self.serialInst.baudrate = baud
        self.running = True
        self.fx = 0.0   # measured force X (N)
        self.fy = 0.0   # measured force Y (N)
        self.magnitude = 0.0
        self.direction = 0.0
        self._count = 0
        self._t0 = time.time()
       

    def _parse_line(self, line: str):
        """Parse lines like: 'avg X: 0.123\tavg Y: -0.456\tmagnitude: 0.789'"""
        try:
            parts = {}
            for segment in line.split("\t"):
                segment = segment.strip()
                if segment.startswith("X:"):
                    parts["fx"] = float(segment.replace("avg X:", "").strip())
                elif segment.startswith("Y:"):
                    parts["fy"] = float(segment.replace("avg Y:", "").strip())
                # elif segment.startswith("magnitude:"):
                #     parts["mag"] = float(segment.replace("magnitude:", "").strip())
                # elif segment.startswith("direction:"):
                    # parts["dir"] = float(segment.replace("direction:", "").strip())
            if "fx" in parts and "fy" in parts:
                self.fx = parts["fx"]
                self.fy = parts["fy"]
                # self.magnitude = parts.get("mag", 0.0)
                # self.direction = parts.get("dir", 0.0)
        except Exception as e:
            print(f"[seeeduino parse] {e} | raw: {line}")

    
    def read_serial(self):
        while self.running:
            try:
                waiting = self.serialInst.in_waiting
                # if waiting > 500:
                #     self.serialInst.reset_input_buffer()
                if waiting > 0:
                    line = self.serialInst.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        self._parse_line(line)
                    
                else:
                    time.sleep(0.0005)
            except Exception as e:
                if self.running:
                    print(f"[seeeduino] {e}")
    def start(self):
        try:
            self.serialInst.open()
            print(f"[seeeduino] Connected to {self.serialInst.port}")
            t = threading.Thread(target=self.read_serial, daemon=True)
            t.start()
        except serial.SerialException as e:
            print(f"[seeeduino] Could not open port: {e}")

    def stop(self):
        self.running = False
        if self.serialInst.is_open:
            self.serialInst.close()


class TeensyPort:
    def __init__(self):
        self.ports = serial.tools.list_ports.comports()
        self.serialInst = serial.Serial()
        self.portsList = []
        self.use = None
        self.running = True
        self.e1 = 0
        self.e2 = 0
        self.raw_e1 = 0.0
        self.raw_e2 = 0.0
        self.offset_e1 = 0.0
        self.offset_e2 = 0.0
        self.enc1 = 0
        self.enc2 = 0
        self.tau1_actual = 0.0
        self.tau2_actual = 0.0
        self.enc_reset = False
        self.encoder = ""

        for port in self.ports:
            self.portsList.append(str(port))
        print(self.portsList)

        for i in range(len(self.portsList)):
            if self.portsList[i].startswith("/dev/ttyACM0"):
                self.use = "/dev/ttyACM0"
                break

        if self.use is None:
            raise Exception("Teensy port /dev/ttyACM1 not found!")

        self.serialInst.baudrate = 115200
        self.serialInst.port = self.use

    def parse_encoder_value(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    def encoder_reset(self):
        # Reset on Arduino
        try:
            if self.serialInst.is_open:
                self.serialInst.write(b'R\n')
                print("[RESET] sent to Teensy")
        except Exception as e:
            print(f"[RESET] serial error: {e}")
        # Reset on Pi side too
        self.offset_e1 = self.raw_e1
        self.offset_e2 = self.raw_e2
        self.enc_reset = False
        print(f"[RESET] Pi-side zeroed at Raw: {self.offset_e1}, {self.offset_e2}")

    def read_serial(self):
        while self.running:
            try:
                waiting = self.serialInst.in_waiting
                # if waiting > 500:
                #     self.serialInst.reset_input_buffer()
                if waiting > 0:
                    response = self.serialInst.readline()
                    self.encoder = response.decode('utf-8').strip()
                    if self.encoder.startswith("ENC_RESET"):
                        print("[RESET] confirmed by Teensy")
                        continue
                    values = [v for v in self.encoder.split(",") if v]
                    if len(values) >= 2:
                        self.raw_e1 = self.parse_encoder_value(values[0])
                        self.raw_e2 = self.parse_encoder_value(values[1])
                        self.enc1 = round(self.raw_e1 - self.offset_e1, 2)
                        self.enc2 = round(self.raw_e2 - self.offset_e2, 2)
                    # # --- rate counter ---
                    # self._count += 1
                    # if self._count % 100 == 0:
                    #     elapsed = time.time() - self._t0
                    #     print(f"[teensy rate] {self._count/elapsed:.1f} Hz")
                    #     self._count = 0
                    #     self._t0 = time.time()
                    # # --- end rate counter --- 
                else:
                    time.sleep(0.0005)
            except Exception as e:
                if self.running:
                    print(f"\r[ERROR] {e}")

    def start(self):
        try:
            self.serialInst.open()
            print(f"\nConnected to {self.use}")
            read_thread = threading.Thread(target=self.read_serial, daemon=True)
            read_thread.start()
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
            self.running = False
        except serial.SerialException as e:
            print(f"\n Serial Port Error: {e}")
            self.running = False