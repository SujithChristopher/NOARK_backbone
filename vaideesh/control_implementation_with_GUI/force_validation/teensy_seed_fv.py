import serial
import serial.tools.list_ports
import threading
import time

# ── USB identity constants ────────────────────────────────────────────────────
_TEENSY_VID, _TEENSY_PID = 0x16C0, 0x0483   # Teensyduino USB Serial
_XIAO_VID                = 0x2886            # Seeed XIAO family (any PID)

def _find_port(vid, pid=None, mfr_hint=None):
    for p in serial.tools.list_ports.comports():
        if vid and p.vid == vid and (pid is None or p.pid == pid):
            return p.device
        if mfr_hint and mfr_hint.lower() in (p.manufacturer or "").lower():
            return p.device
    return None

def find_teensy_port():
    port = _find_port(_TEENSY_VID, _TEENSY_PID) or _find_port(None, mfr_hint="Teensyduino")
    if port is None:
        raise RuntimeError("Teensy not found — is it plugged in?")
    print(f"[auto-detect] Teensy → {port}")
    return port

def find_seeed_port():
    port = _find_port(_XIAO_VID) or _find_port(None, mfr_hint="Seeed")
    if port is None:
        raise RuntimeError("Seeed XIAO not found — is it plugged in?")
    print(f"[auto-detect] Seeed XIAO → {port}")
    return port

# ─────────────────────────────────────────────────────────────────────────────

class SeeduinoPort:
    """Reads load cell X/Y force data from the Seeduino over serial."""
    def __init__(self, port=None, baud=115200):
        if port is None:
            port = find_seeed_port()
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
        self._tare_confirmed =False
        self.on_update = None  # callback for new force values: on_update(fx, fy)
       

    def _parse_line(self, line: str):
        """Parse: '0.123,-0.456'"""
        self._tare_confirmed = True
        try:
            parts = line.split(",")
            if len(parts) == 2:
                self.fx = float(parts[0])
                self.fy = float(parts[1])
                if self.on_update:
                    self.on_update(self.fx, self.fy)
        except Exception as e:
            print(f"[seeeduino parse] {e} | raw: {line}")

    
    def read_serial(self):
        while self.running:
            try:
                waiting = self.serialInst.in_waiting
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

    @property
    def tare_confirmed(self):
        return self._tare_confirmed

    def send_tare(self):
        try:
            if self.serialInst.is_open:
                self._tare_confirmed = False
                self.serialInst.write(b'T\n')
                print("[seeeduino] tare command sent")
        except Exception as e:
            print(f"[seeeduino tare] {e}")
class TeensyPort:
    def __init__(self):
        self.serialInst = serial.Serial()
        self.serialInst.timeout = 0.1
        self.running = True
        self.e1 = 0
        self.e2 = 0
        self.raw_e1 = 0.0
        self.raw_e2 = 0.0
        self.offset_e1 = 0.0
        self.offset_e2 = 0.0
        self.enc1 = 0
        self.enc2 = 0
        self.enc_reset = False
        self.encoder = ""
        self.on_update = None
        self._tare_event = threading.Event()
        self._tare_event.clear()

        self.use = find_teensy_port()
        self.serialInst.baudrate = 115200
        self.serialInst.port = self.use

    def parse_encoder_value(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    def encoder_reset(self):
        self._tare_event.clear()
        # Reset on Pi side too
        self.offset_e1 = self.raw_e1
        self.offset_e2 = self.raw_e2
        self.enc_reset = False
        print(f"[RESET] Pi-side zeroed at Raw: {self.offset_e1}, {self.offset_e2}")
        try:
            if self.serialInst.is_open:
                self.serialInst.reset_input_buffer()   # discard stale encoder backlog
                self.serialInst.write(b'R\n')
                self.serialInst.flush()
                print("[RESET] sent to Teensy")
        except Exception as e:
            print(f"[RESET] serial error: {e}")

    def read_serial(self):
        while self.running:
            try:
                line = self.serialInst.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                if line.startswith("ENC_RESET"):
                    self._tare_event.set()
                    print("[RESET] confirmed by Teensy")
                    continue
                values = [v for v in line.split(",") if v]
                if len(values) >= 2:
                    self.raw_e1 = self.parse_encoder_value(values[0])
                    self.raw_e2 = self.parse_encoder_value(values[1])
                    self.enc1 = round(self.raw_e1 - self.offset_e1, 2)
                    self.enc2 = round(self.raw_e2 - self.offset_e2, 2)
                    if self.on_update:
                        self.on_update(self.enc1, self.enc2)
            except Exception as e:
                if self.running:
                    print(f"\r[ERROR] {e}")

    @property
    def tare_confirmed(self):
        return self._tare_event.is_set()

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