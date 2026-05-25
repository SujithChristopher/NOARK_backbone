"""
loadcell_sender.py
------------------
Run this on the LAPTOP connected to load cell Teensy.
Reads serial data and forwards to Pi via UDP WiFi.

Usage:
    pip install pyserial
    python loadcell_sender.py

Configure:
    SERIAL_PORT → your load cell Teensy COM port
    PI_IP       → your Raspberry Pi IP address
"""

import serial
import socket
import time

# ── CONFIG ────────────────────────────────────────────────────────────────
SERIAL_PORT = "COM6"          # change to your load cell Teensy port
BAUD_RATE   = 15200
PI_IP       = "10.172.175.221"  # change to your Pi's IP (run hostname -I on Pi)
UDP_PORT    = 5005
# ─────────────────────────────────────────────────────────────────────────

def main():
    print(f"[loadcell_sender] Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser  = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[loadcell_sender] Sending to {PI_IP}:{UDP_PORT}")
    print("[loadcell_sender] Press Ctrl+C to stop\n")

    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                # Append timestamp to help Pi detect stale data
                timestamp = time.time()
                packet = f"{line},{timestamp:.3f}"
                sock.sendto(packet.encode(), (PI_IP, UDP_PORT))
                print(f"[send] {packet}")
        except KeyboardInterrupt:
            print("\n[loadcell_sender] Stopped.")
            break
        except Exception as e:
            print(f"[error] {e}")
            time.sleep(0.01)

    ser.close()
    sock.close()

if __name__ == "__main__":
    main()
