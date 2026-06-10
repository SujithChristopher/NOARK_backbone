# import serial
# import time

# ser = serial.Serial("COM13", 115200, timeout=1)
# time.sleep(2)
# ser.reset_input_buffer()

# print("Reading 20 samples from sensor Teensy:")
# for i in range(20):
#     line = ser.readline().decode("utf-8", errors="ignore").strip()
#     print(f"  {i+1}: '{line}'")

# ser.close()
# import serial, time
# ser = serial.Serial("COM5", 115200, timeout=1)
# time.sleep(2)
# ser.write(b"774\n")
# ser.flush()
# print("PWM 774 sent")
# time.sleep(5)
# ser.write(b"0\n")
# ser.flush()
# print("PWM 0 sent")
# ser.close()
# """
# find_kt.py  —  Torque Constant (Kt) Finder
# ===========================================
# Motor Teensy steps through PWM automatically — no Serial commands sent.
# Python only reads sensor Teensy (HX711, Nm) and motor Teensy (PWM announcements).

# Timing is fully automatic — motor Teensy announces each PWM step,
# Python settles then captures torque, then waits for next PWM announcement.

# Usage:
#     pip install pyserial
#     python find_kt.py --motor COM5 --sensor COM13
# """

# import serial
# import serial.tools.list_ports
# import argparse
# import time
# import csv
# import os
# import statistics
# from datetime import datetime

# # ── Match your Variables.h ─────────────────────────────────
# MINPWM = 410
# MAXPWM = 3686
# MIN_I  = 0.0    # A
# MAX_I  = 5.0    # A

# # ── Match your Arduino sketch ───────────────────────────────
# STEP_DURATION   = 7.0    # seconds — same as Arduino STEP_DURATION (7000ms)
# SETTLE_TIME     = 2.0    # seconds — wait after PWM changes before capturing
# AVERAGE_SAMPLES = 160    # HX711 samples to average (160 × 12.5ms = 2s at 80Hz)

# # ── Config ──────────────────────────────────────────────────
# DEFAULT_MOTOR_PORT  = "COM5"
# DEFAULT_SENSOR_PORT = "COM13"
# BAUD                = 115200
# LOG_DIR             = "kt_logs"
# # ───────────────────────────────────────────────────────────


# def pwm_to_current(pwm):
#     """Back-calculate current from PWM using inverted map()."""
#     return (pwm - MINPWM) / (MAXPWM - MINPWM) * (MAX_I - MIN_I) + MIN_I


# def list_ports():
#     return [p.device for p in serial.tools.list_ports.comports()]


# def wait_for_line(ser, keyword, timeout=30.0):
#     """Wait until a line containing keyword arrives, return the full line."""
#     deadline = time.time() + timeout
#     while time.time() < deadline:
#         line = ser.readline().decode("utf-8", errors="ignore").strip()
#         if keyword in line:
#             return line
#     return None


# def capture_torque(sensor_ser, n=AVERAGE_SAMPLES):
#     """Average n valid float readings from sensor Teensy."""
#     sensor_ser.reset_input_buffer()
#     readings = []
#     deadline = time.time() + 15.0
#     while len(readings) < n and time.time() < deadline:
#         line = sensor_ser.readline().decode("utf-8", errors="ignore").strip()
#         if not line:
#             continue
#         # Skip BOOT lines from sensor Teensy resets
#         if not line[0].lstrip('-').replace('.','').isdigit():
#             continue
#         try:
#             readings.append(float(line))
#         except ValueError:
#             continue
#     if not readings:
#         return None, None
#     return statistics.mean(readings), statistics.stdev(readings) if len(readings) > 1 else 0.0


# def main():
#     parser = argparse.ArgumentParser(description="Kt finder — automatic PWM sweep")
#     parser.add_argument("--motor",  default=DEFAULT_MOTOR_PORT,
#                         help=f"COM port for motor Teensy  (default: {DEFAULT_MOTOR_PORT})")
#     parser.add_argument("--sensor", default=DEFAULT_SENSOR_PORT,
#                         help=f"COM port for sensor Teensy (default: {DEFAULT_SENSOR_PORT})")
#     args = parser.parse_args()

#     print()
#     print("=" * 56)
#     print("  Torque Constant (Kt) Finder  —  Automatic PWM Sweep")
#     print("=" * 56)
#     print(f"  Motor  port    : {args.motor}")
#     print(f"  Sensor port    : {args.sensor}")
#     print(f"  Step duration  : {STEP_DURATION}s per PWM step")
#     print(f"  Settle time    : {SETTLE_TIME}s")
#     print(f"  Avg samples    : {AVERAGE_SAMPLES} per capture")
#     print()

#     ports = list_ports()
#     if ports:
#         print("  Detected COM ports:", ", ".join(ports))
#     print()

#     # ── open both ports ──
#     try:
#         # motor_ser  = serial.Serial(args.motor,  BAUD, timeout=0.5)
#         sensor_ser = serial.Serial(args.sensor, BAUD, timeout=0.5)
#     except serial.SerialException as e:
#         print(f"  ERROR: {e}")
#         return

#     # print("  Waiting for motor Teensy READY signal...", end="", flush=True)
#     # ready = wait_for_line(motor_ser, "READY", timeout=30.0)
#     # if not ready:
#     #     print("\n  ERROR: Motor Teensy did not send READY — check upload and COM port.")
#     #     motor_ser.close()
#     #     sensor_ser.close()
#     #     return
#     # print(" ready.")
#     print("  Motor Teensy running standalone — no Serial connection needed.")

#     # Tare window — sensor should be stable before motor starts
#     print("  Taring sensor (motor still off)...", end="", flush=True)
#     sensor_ser.reset_input_buffer()
#     time.sleep(1.0)
#     print(" done.")
#     print()
#     print("  Motor Teensy starting PWM sweep now...")
#     print()

#     samples = []   # list of (pwm, current_A, torque_Nm, kt)

#     while True:
#         # Wait for PWM announcement from motor Teensy
#         line = wait_for_line(motor_ser, "PWM:", timeout=STEP_DURATION + 5.0)

#         if line is None:
#             print("  Timeout waiting for next PWM step — test may be complete.")
#             break

#         if "DONE" in line:
#             print("  Motor Teensy reports DONE — all steps complete.")
#             break

#         # Parse PWM value from "PWM:774"
#         try:
#             pwm = int(line.split(":")[1].strip())
#         except (IndexError, ValueError):
#             continue

#         current = pwm_to_current(pwm)
#         print(f"  → Motor PWM={pwm}  I={current:.4f}A — settling {SETTLE_TIME}s...", end="", flush=True)

#         # Settle
#         time.sleep(SETTLE_TIME)
#         sensor_ser.reset_input_buffer()   # flush stale readings

#         # Capture
#         print(f" capturing {AVERAGE_SAMPLES} samples...", end="", flush=True)
#         avg_torque, std_torque = capture_torque(sensor_ser)

#         if avg_torque is None:
#             print(f"\n  WARNING: No torque data at PWM={pwm} — skipping.")
#             continue

#         if abs(current) < 0.01:
#             print(f"\n  Skipping PWM={pwm} — current too low for Kt calculation.")
#             continue

#         kt = avg_torque / current
#         samples.append((pwm, current, avg_torque, kt))

#         print(f"\n  ✓  PWM={pwm}  I={current:.4f}A  τ={avg_torque:.4f}±{std_torque:.4f}Nm  Kt={kt:.4f}Nm/A")
#         print()

#     motor_ser.close()
#     sensor_ser.close()

#     if not samples:
#         print("\n  No samples collected. Exiting.")
#         return

#     # ── results ──
#     print()
#     print("=" * 56)
#     print("  RESULTS")
#     print("=" * 56)
#     print(f"  {'#':<4}  {'PWM':<7}  {'I (A)':<10}  {'τ (Nm)':<12}  Kt (Nm/A)")
#     print("  " + "─" * 50)
#     for i, (p, c, t, k) in enumerate(samples, 1):
#         print(f"  {i:<4}  {p:<7}  {c:<10.4f}  {t:<12.4f}  {k:.4f}")

#     kt_vals      = [s[3] for s in samples]
#     kt_mean      = statistics.mean(kt_vals)
#     kt_stdev     = statistics.stdev(kt_vals) if len(kt_vals) > 1 else 0.0

#     # Linear regression forced through origin: τ = Kt × I
#     num = sum(c * t for _, c, t, _ in samples)
#     den = sum(c * c for _, c, t, _ in samples)
#     kt_regression = num / den if den > 0 else kt_mean

#     print("  " + "─" * 50)
#     print(f"  Mean Kt          : {kt_mean:.5f} Nm/A")
#     print(f"  Std deviation    : {kt_stdev:.5f} Nm/A")
#     print(f"  Kt (regression)  : {kt_regression:.5f} Nm/A  ← use this")
#     print()
#     print(f"  ➜  In Variables.h:  float Torque_Const = {kt_regression:.4f};")
#     print("=" * 56)

#     # ── save CSV ──
#     os.makedirs(LOG_DIR, exist_ok=True)
#     fname = os.path.join(LOG_DIR, f"kt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
#     with open(fname, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["sample", "pwm", "current_A", "torque_Nm", "Kt_NmPerA"])
#         for i, (p, c, t, k) in enumerate(samples, 1):
#             writer.writerow([i, p, f"{c:.5f}", f"{t:.5f}", f"{k:.5f}"])
#         writer.writerow([])
#         writer.writerow(["kt_mean",       f"{kt_mean:.5f}"])
#         writer.writerow(["kt_stdev",      f"{kt_stdev:.5f}"])
#         writer.writerow(["kt_regression", f"{kt_regression:.5f}"])

#     print(f"\n  Data saved → {fname}")
#     print()


# if __name__ == "__main__":
# #     main()
# """
# find_kt.py  —  Torque Constant (Kt) Finder
# ===========================================
# Motor Teensy runs standalone (powered from motor driver 5V, no USB).
# Python only reads sensor Teensy (COM13) — no motor Serial port needed.
# Timing syncs with Arduino sketch via STEP_DURATION.

# BEFORE RUNNING:
#   1. Upload TORQUE_CONSTANT_1Nm.ino to motor Teensy
#   2. Disconnect motor Teensy USB from laptop
#   3. Power motor Teensy from motor driver 5V pin
#   4. Run this script — it will prompt you to start motor Teensy

# Usage:
#     pip install pyserial
#     python find_kt.py --sensor COM13
# """

# import serial
# import serial.tools.list_ports
# import argparse
# import time
# import csv
# import os
# import statistics
# from datetime import datetime

# # ── Match your Variables.h ──────────────────────────────────
# MINPWM = 410
# MAXPWM = 3686
# MIN_I  = 0.0    # A
# MAX_I  = 5.0    # A

# # ── Must match TORQUE_CONSTANT_1Nm.ino exactly ──────────────
# PWM_STEPS     = [774, 1138, 1502, 1866, 2230, 2594, 2958, 3322, 3686]
# STEP_DURATION = 7.0    # seconds — same as Arduino STEP_DURATION (7000ms)
# SETTLE_TIME   = 2.0    # seconds — settle before capturing
# CAPTURE_TIME  = 2.0    # seconds — time to collect samples (160 × 12.5ms = 2s)

# # ── Config ──────────────────────────────────────────────────
# DEFAULT_SENSOR_PORT = "COM13"
# BAUD                = 115200
# AVERAGE_SAMPLES     = 160     # 160 × 12.5ms = 2s at 80Hz HX711
# LOG_DIR             = "kt_logs"
# # ───────────────────────────────────────────────────────────


# def pwm_to_current(pwm):
#     """Back-calculate current from PWM using inverted map()."""
#     return (pwm - MINPWM) / (MAXPWM - MINPWM) * (MAX_I - MIN_I) + MIN_I


# def list_ports():
#     return [p.device for p in serial.tools.list_ports.comports()]


# def capture_torque(sensor_ser, n=AVERAGE_SAMPLES):
#     """Average n valid float readings from sensor Teensy."""
#     sensor_ser.reset_input_buffer()
#     readings = []
#     deadline = time.time() + 15.0
#     while len(readings) < n and time.time() < deadline:
#         line = sensor_ser.readline().decode("utf-8", errors="ignore").strip()
#         if not line:
#             continue
#         # Skip non-numeric lines like BOOT
#         try:
#             val = float(line)
#             readings.append(val)
#         except ValueError:
#             continue
#     if not readings:
#         return None, None
#     return statistics.mean(readings), statistics.stdev(readings) if len(readings) > 1 else 0.0


# def main():
#     parser = argparse.ArgumentParser(description="Kt finder — sensor only, time-based sync")
#     parser.add_argument("--sensor", default=DEFAULT_SENSOR_PORT,
#                         help=f"COM port for sensor Teensy (default: {DEFAULT_SENSOR_PORT})")
#     args = parser.parse_args()

#     print()
#     print("=" * 56)
#     print("  Torque Constant (Kt) Finder  —  Sensor Only Mode")
#     print("=" * 56)
#     print(f"  Sensor port    : {args.sensor}")
#     print(f"  PWM steps      : {PWM_STEPS}")
#     print(f"  Currents       : {[round(pwm_to_current(p), 3) for p in PWM_STEPS]} A")
#     print(f"  Step duration  : {STEP_DURATION}s  (settle {SETTLE_TIME}s + capture {CAPTURE_TIME}s)")
#     print(f"  Total duration : ~{len(PWM_STEPS) * STEP_DURATION:.0f}s")
#     print()

#     ports = list_ports()
#     if ports:
#         print("  Detected COM ports:", ", ".join(ports))
#     print()

#     # ── open sensor port only ──
#     try:
#         sensor_ser = serial.Serial(args.sensor, BAUD, timeout=0.5)
#     except serial.SerialException as e:
#         print(f"  ERROR: {e}")
#         return

#     time.sleep(1.0)
#     sensor_ser.reset_input_buffer()

#     print("  HOW TO USE")
#     print("  ──────────────────────────────────────────────────────")
#     print("  1. Motor Teensy must be powered from motor driver 5V")
#     print("     (USB disconnected from laptop)")
#     print("  2. Make sure torque sensor shaft is locked")
#     print("  3. Press ENTER here — then immediately power on")
#     print("     motor Teensy (or press its reset button)")
#     print(f"  4. Script auto-captures {len(PWM_STEPS)} steps, ~{len(PWM_STEPS) * STEP_DURATION:.0f}s total")
#     print("  ──────────────────────────────────────────────────────")
#     print()

#     input("  Press ENTER when ready to start: ")

#     print()
#     print("  ► Power on motor Teensy NOW (or press reset button)")
#     print(f"  Waiting {STEP_DURATION}s for first PWM step to begin...")
#     print()

#     # Wait for motor Teensy to boot and start first step
#     time.sleep(STEP_DURATION * 0.8)   # wait 80% of first step duration

#     samples = []   # list of (pwm, current_A, torque_Nm, kt)

#     for idx, pwm in enumerate(PWM_STEPS):
#         current = pwm_to_current(pwm)

#         print(f"  [{idx+1}/{len(PWM_STEPS)}] PWM={pwm}  I={current:.4f}A", end="", flush=True)

#         # Settle
#         print(f" — settling {SETTLE_TIME}s...", end="", flush=True)
#         time.sleep(SETTLE_TIME)
#         sensor_ser.reset_input_buffer()   # flush stale data

#         # Capture
#         print(f" capturing...", end="", flush=True)
#         avg_torque, std_torque = capture_torque(sensor_ser)

#         if avg_torque is None:
#             print(f"\n  WARNING: No torque data at PWM={pwm} — skipping.")
#         elif abs(current) < 0.01:
#             print(f"\n  Skipping PWM={pwm} — current too low.")
#         else:
#             kt = avg_torque / current
#             samples.append((pwm, current, avg_torque, kt))
#             print(f"\n  ✓  τ={avg_torque:.4f}±{std_torque:.4f}Nm  Kt={kt:.4f}Nm/A")

#         print()

#         # Wait remaining time before next step (only if not last step)
#         if idx < len(PWM_STEPS) - 1:
#             remaining = STEP_DURATION - SETTLE_TIME - CAPTURE_TIME
#             if remaining > 0:
#                 time.sleep(remaining)

#     sensor_ser.close()

#     if not samples:
#         print("\n  No samples collected. Exiting.")
#         return

#     # ── results ──
#     print()
#     print("=" * 56)
#     print("  RESULTS")
#     print("=" * 56)
#     print(f"  {'#':<4}  {'PWM':<7}  {'I (A)':<10}  {'τ (Nm)':<12}  Kt (Nm/A)")
#     print("  " + "─" * 50)
#     for i, (p, c, t, k) in enumerate(samples, 1):
#         print(f"  {i:<4}  {p:<7}  {c:<10.4f}  {t:<12.4f}  {k:.4f}")

#     kt_vals      = [s[3] for s in samples]
#     kt_mean      = statistics.mean(kt_vals)
#     kt_stdev     = statistics.stdev(kt_vals) if len(kt_vals) > 1 else 0.0

#     # Linear regression forced through origin: τ = Kt × I
#     num = sum(c * t for _, c, t, _ in samples)
#     den = sum(c * c for _, c, t, _ in samples)
#     kt_regression = num / den if den > 0 else kt_mean

#     print("  " + "─" * 50)
#     print(f"  Mean Kt          : {kt_mean:.5f} Nm/A")
#     print(f"  Std deviation    : {kt_stdev:.5f} Nm/A")
#     print(f"  Kt (regression)  : {kt_regression:.5f} Nm/A  ← use this")
#     print()
#     print(f"  ➜  In Variables.h:  float Torque_Const = {kt_regression:.4f};")
#     print("=" * 56)

#     # ── save CSV ──
#     os.makedirs(LOG_DIR, exist_ok=True)
#     fname = os.path.join(LOG_DIR, f"kt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
#     with open(fname, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["sample", "pwm", "current_A", "torque_Nm", "Kt_NmPerA"])
#         for i, (p, c, t, k) in enumerate(samples, 1):
#             writer.writerow([i, p, f"{c:.5f}", f"{t:.5f}", f"{k:.5f}"])
#         writer.writerow([])
#         writer.writerow(["kt_mean",       f"{kt_mean:.5f}"])
#         writer.writerow(["kt_stdev",      f"{kt_stdev:.5f}"])
#         writer.writerow(["kt_regression", f"{kt_regression:.5f}"])

#     print(f"\n  Data saved → {fname}")
#     print()


# if __name__ == "__main__":
#     main()
import serial, time, statistics

ser = serial.Serial("COM13", 115200, timeout=1)
time.sleep(2)

while True:
    pwm = int(input("Enter PWM (0 to quit): "))
    if pwm == 0: break
    ser.reset_input_buffer()
    time.sleep(1.5)  # settle
    readings = []
    while len(readings) < 100:
        try: readings.append(float(ser.readline().decode().strip()))
        except: pass
    avg = statistics.mean(readings)
    current = (pwm - 410) / (3686 - 410) * 5.0
    kt = avg / current if current > 0.05 else 0
    print(f"Torque={avg:.4f}Nm | I={current:.4f}A | Kt={kt:.4f}Nm/A\n")

ser.close()