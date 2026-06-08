"""
find_kt.py  —  Torque Constant (Kt) Finder
===========================================
Each ENTER press:
  1. Sends current PWM value over Serial to motor Teensy
  2. Waits for motor to settle
  3. Reads torque from sensor Teensy (HX711, Nm)
  4. Back-calculates current from PWM
  5. Computes Kt = torque / current

PWM steps from MINPWM (410) to MAXPWM (3686) in equal steps.

Usage:
    pip install pyserial
    python find_kt.py --motor COM3 --sensor COM4 --steps 10
"""

import serial
import serial.tools.list_ports
import argparse
import time
import csv
import os
import statistics
from datetime import datetime

# ── Match your Variables.h ─────────────────────────────────
MINPWM = 410
MAXPWM = 3686
MIN_I  = 0.0    # A
MAX_I  = 5.0    # A

# ── Config ─────────────────────────────────────────────────
DEFAULT_MOTOR_PORT  = "COM3"
DEFAULT_SENSOR_PORT = "COM4"
DEFAULT_STEPS       = 10      # how many PWM steps between 410 and 3686
BAUD                = 115200
SETTLE_TIME         = 2.0     # seconds to settle after PWM changes
AVERAGE_SAMPLES     = 50      # HX711 readings to average per capture
LOG_DIR             = "kt_logs"
# ──────────────────────────────────────────────────────────


def pwm_to_current(pwm):
    """Back-calculate current from PWM using inverted map()."""
    return (pwm - MINPWM) / (MAXPWM - MINPWM) * (MAX_I - MIN_I) + MIN_I


def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


def capture_torque(sensor_ser, n=AVERAGE_SAMPLES):
    """Average n valid float readings from the sensor Teensy."""
    sensor_ser.reset_input_buffer()
    readings = []
    deadline = time.time() + 8.0
    while len(readings) < n and time.time() < deadline:
        line = sensor_ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            readings.append(float(line))
        except ValueError:
            continue
    if not readings:
        return None, None
    return statistics.mean(readings), statistics.stdev(readings) if len(readings) > 1 else 0.0


def send_pwm(motor_ser, pwm):
    """Send PWM value as integer string over Serial to motor Teensy."""
    motor_ser.reset_input_buffer()
    motor_ser.write(f"{pwm}\n".encode())
    motor_ser.flush()


def main():
    parser = argparse.ArgumentParser(description="Kt finder — PWM stepped via Serial")
    parser.add_argument("--motor",  default=DEFAULT_MOTOR_PORT,
                        help=f"COM port for motor Teensy  (default: {DEFAULT_MOTOR_PORT})")
    parser.add_argument("--sensor", default=DEFAULT_SENSOR_PORT,
                        help=f"COM port for sensor Teensy (default: {DEFAULT_SENSOR_PORT})")
    parser.add_argument("--steps",  type=int, default=DEFAULT_STEPS,
                        help=f"Number of PWM steps from {MINPWM} to {MAXPWM} (default: {DEFAULT_STEPS})")
    args = parser.parse_args()

    # Build PWM step list: evenly spaced from MINPWM to MAXPWM
    step_size = (MAXPWM - MINPWM) // (args.steps - 1)
    pwm_steps = list(range(MINPWM, MAXPWM + 1, step_size))
    if pwm_steps[-1] != MAXPWM:
        pwm_steps.append(MAXPWM)   # always include max

    print()
    print("=" * 56)
    print("  Torque Constant (Kt) Finder  —  Serial PWM Stepping")
    print("=" * 56)
    print(f"  Motor  port : {args.motor}")
    print(f"  Sensor port : {args.sensor}")
    print(f"  PWM steps   : {pwm_steps}")
    print(f"  Currents    : {[round(pwm_to_current(p), 3) for p in pwm_steps]} A")
    print()

    ports = list_ports()
    if ports:
        print("  Detected COM ports:", ", ".join(ports))
    print()

    # ── open both ports ──
    try:
        motor_ser  = serial.Serial(args.motor,  BAUD, timeout=0.1)
        sensor_ser = serial.Serial(args.sensor, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"  ERROR: {e}")
        return

    print("  Waiting for both Teensys to initialise...", end="", flush=True)
    time.sleep(2.0)
    motor_ser.reset_input_buffer()
    sensor_ser.reset_input_buffer()
    print(" ready.")
    print()

    # Send PWM=0 first so motor starts stopped
    send_pwm(motor_ser, 0)

    print("  HOW TO USE")
    print("  ──────────────────────────────────────────────────────")
    print("  - Make sure torque sensor locks the motor shaft.")
    print(f"  - Press ENTER to send next PWM and capture sample.")
    print(f"  - Script will step through {len(pwm_steps)} PWM values automatically.")
    print(f"  - Type 'skip' to skip current step.")
    print(f"  - Type 'stop' to end early and compute Kt.")
    print("  ──────────────────────────────────────────────────────")
    print()

    samples = []   # list of (pwm, current_A, torque_Nm, kt)

    for idx, pwm in enumerate(pwm_steps):
        current = pwm_to_current(pwm)

        try:
            user = input(f"  [{idx+1}/{len(pwm_steps)}] PWM={pwm}  I={current:.4f}A  — Press ENTER to capture (skip/stop): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Interrupted.")
            break

        if user == "stop":
            break
        if user == "skip":
            print(f"  Skipped PWM {pwm}.")
            continue

        # Send PWM to motor Teensy
        send_pwm(motor_ser, pwm)
        print(f"  → Sent PWM={pwm} to motor Teensy.")

        # Settle
        print(f"  Settling {SETTLE_TIME}s ...", end="", flush=True)
        time.sleep(SETTLE_TIME)

        # Capture torque
        print(f" capturing {AVERAGE_SAMPLES} samples ...", end="", flush=True)
        avg_torque, std_torque = capture_torque(sensor_ser)

        if avg_torque is None:
            print("\n  WARNING: No data from sensor Teensy — skipping.")
            continue

        kt = avg_torque / current if abs(current) > 0.01 else 0.0
        samples.append((pwm, current, avg_torque, kt))

        print(f"\n  ✓  PWM={pwm}  I={current:.4f}A  τ={avg_torque:.4f}±{std_torque:.4f}Nm  Kt={kt:.4f}Nm/A")
        print()

    # Stop motor
    send_pwm(motor_ser, 0)
    print("  Motor stopped (PWM=0 sent).")

    motor_ser.close()
    sensor_ser.close()

    if not samples:
        print("\n  No samples collected. Exiting.")
        return

    # ── results ──
    print()
    print("=" * 56)
    print("  RESULTS")
    print("=" * 56)
    print(f"  {'#':<4}  {'PWM':<7}  {'I (A)':<10}  {'τ (Nm)':<12}  Kt (Nm/A)")
    print("  " + "─" * 50)
    for i, (p, c, t, k) in enumerate(samples, 1):
        print(f"  {i:<4}  {p:<7}  {c:<10.4f}  {t:<12.4f}  {k:.4f}")

    kt_vals       = [s[3] for s in samples]
    kt_mean       = statistics.mean(kt_vals)
    kt_stdev      = statistics.stdev(kt_vals) if len(kt_vals) > 1 else 0.0

    # Linear regression forced through origin: τ = Kt × I
    num = sum(c * t for _, c, t, _ in samples)
    den = sum(c * c for _, c, t, _ in samples)
    kt_regression = num / den if den > 0 else kt_mean

    print("  " + "─" * 50)
    print(f"  Mean Kt          : {kt_mean:.5f} Nm/A")
    print(f"  Std deviation    : {kt_stdev:.5f} Nm/A")
    print(f"  Kt (regression)  : {kt_regression:.5f} Nm/A  ← use this")
    print()
    print(f"  ➜  In Variables.h:  float Torque_Const = {kt_regression:.4f};")
    print("=" * 56)

    # ── save CSV ──
    os.makedirs(LOG_DIR, exist_ok=True)
    fname = os.path.join(LOG_DIR, f"kt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "pwm", "current_A", "torque_Nm", "Kt_NmPerA"])
        for i, (p, c, t, k) in enumerate(samples, 1):
            writer.writerow([i, p, f"{c:.5f}", f"{t:.5f}", f"{k:.5f}"])
        writer.writerow([])
        writer.writerow(["kt_mean",       f"{kt_mean:.5f}"])
        writer.writerow(["kt_stdev",      f"{kt_stdev:.5f}"])
        writer.writerow(["kt_regression", f"{kt_regression:.5f}"])

    print(f"\n  Data saved → {fname}")
    print()


if __name__ == "__main__":
    main()
