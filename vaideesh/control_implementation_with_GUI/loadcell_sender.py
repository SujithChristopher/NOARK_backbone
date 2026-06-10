import serial
import socket
import time
import threading
import keyboard

# ── CONFIG ────────────────────────────────────────────────────────────────
SERIAL_PORT = "COM6"
BAUD_RATE   = 115200
PI_IP       = "10.172.175.221"  # change to your Pi IP
UDP_PORT    = 5005
# ─────────────────────────────────────────────────────────────────────────

def keyboard_listener(ser):
    """Press T to tare — no Enter needed."""
    print("[keyboard] Press 'T' to tare load cell")
    while True:
        if keyboard.is_pressed('t'):
            ser.write(b't')
            print("[tare] Tare command sent to Teensy")
            time.sleep(0.5)   # debounce
        time.sleep(0.01)

def main():
    print(f"[loadcell_sender] Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser  = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[loadcell_sender] Sending to {PI_IP}:{UDP_PORT}")
    print("[loadcell_sender] Press 'T' to tare, Ctrl+C to stop\n")

    # Start keyboard listener thread
    t = threading.Thread(target=keyboard_listener, args=(ser,), daemon=True)
    t.start()

    # while True:
    #     try:
    #         line = ser.readline().decode().strip()
            
    #         if line:
    #             parts = line.split(",")
    #             if len(parts) == 3:
    #                 x = float(parts[0])
    #                 y = float(parts[1])
    #                 z = float(parts[2])
    #                 timestamp = time.time()
    #                 packet = f"{x:.3f},{y:.3f},{z:.3f},{timestamp:.3f}"
    #                 sock.sendto(packet.encode(), (PI_IP, UDP_PORT))
    #                 print(f"[send] {packet}")
    
    #     except KeyboardInterrupt:
    #         print("\n[loadcell_sender] Stopped.")
    #         break
    #     except Exception as e:
    #         print(f"[error] {e}")
    #         time.sleep(0.01)
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                parts = line.split(",")
                if len(parts) == 2:        # X and Y only
                    x = float(parts[0])
                    y = float(parts[1])
                    timestamp = time.time()
                    packet = f"{x:.3f},{y:.3f},0.000,{timestamp:.3f}"
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