import cv2
import os
import datetime
import msgpack as mp
import msgpack_numpy as mpn
import argparse
import time
import sys
import csv
import serial
import serial.tools.list_ports
import threading
import keyboard
from picamera2 import Picamera2
import gpiod
import libcamera
# /home/sujith/mocap_recordings
# ── Camera Config ──────────────────────────────────────────────────────────────
FRAME_SIZE = (1280, 800)
WIDTH, HEIGHT = FRAME_SIZE


class CombinedRecorder:
    def __init__(self, _pth=None, record_camera=True, record_sensors=True):
        self._pth = _pth
        self.record_camera = record_camera
        self.record_sensors = record_sensors
        self.start_recording = False   # toggled by 's' key
        self.kill_signal = False       # toggled by 'q' key
        self.start_time = time.time()

        # ── Camera setup ───────────────────────────────────────────────────────
        if self.record_camera:
            self.picam2 = Picamera2()
            main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
            _c = {"FrameRate": 100, "ExposureTime": 8000}
            config = self.picam2.create_video_configuration(
                main, controls=_c, transform=libcamera.Transform(vflip=1)
            )
            self.picam2.configure(config)
            self.picam2.start()

            # GPIO sync pin
            sync_pin = 17
            chip = gpiod.Chip("gpiochip4")
            self.sync_line = chip.get_line(sync_pin)
            self.sync_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)

        # ── Encoder / Serial setup ─────────────────────────────────────────────
        if self.record_sensors:
            self.raw_e1 = 0.0
            self.raw_e2 = 0.0
            self.offset_e1 = 0.0
            self.offset_e2 = 0.0
            self.enc1 = 0.0
            self.enc2 = 0.0
            self.enc_reset = False

            sensor_csv_path = os.path.join(self._pth, "encoder_data.csv")
            with open(sensor_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "millis", "enc1", "enc2"])
            self.sensor_csv_path = sensor_csv_path
            print(f"  Encoder CSV: {sensor_csv_path}")

            # Locate Teensy serial port
            ports = serial.tools.list_ports.comports()
            self.serialInst = serial.Serial()
            use = None
            for port in ports:
                if str(port).startswith("/dev/ttyACM0"):
                    use = "/dev/ttyACM0"
                    break
            if use is None:
                raise Exception("Teensy port /dev/ttyACM0 not found!")
            self.serialInst.baudrate = 115200
            self.serialInst.port = use
            print(f"  Teensy port: {use}")

    # ── Encoder thread ─────────────────────────────────────────────────────────
    def _parse_encoder_value(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _encoder_reset(self):
        self.offset_e1 = self.raw_e1
        self.offset_e2 = self.raw_e2
        self.enc_reset = False
        print(f"\n[RESET] Encoders zeroed — Raw: {self.offset_e1}, {self.offset_e2}")

    def _read_serial(self):
        """
        Runs in its own thread.
        Reads serial at the maximum rate the UART delivers data.
        Writes to CSV only after 's' key starts recording.
        """
        while not self.kill_signal:
            try:
                if self.serialInst.in_waiting > 0:
                    raw = self.serialInst.readline().decode("utf-8").strip()
                    values = [v for v in raw.split(",") if v]

                    if len(values) >= 2:
                        self.raw_e1 = self._parse_encoder_value(values[0])
                        self.raw_e2 = self._parse_encoder_value(values[1])

                        if self.enc_reset:
                            self._encoder_reset()

                        self.enc1 = round(self.raw_e1 - self.offset_e1, 2)
                        self.enc2 = round(self.raw_e2 - self.offset_e2, 2)

                        if self.start_recording:
                            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            millis = round((time.time() - self.start_time) * 1000, 2)
                            with open(self.sensor_csv_path, "a", newline="") as f:
                                csv.writer(f).writerow(
                                    [ts, millis, self.enc1, self.enc2]
                                )
                # No sleep — poll as fast as possible for maximum encoder speed
            except Exception as e:
                if not self.kill_signal:
                    print(f"\r[ENCODER ERROR] {e}")

    # ── Camera thread ──────────────────────────────────────────────────────────
    def _capture_webcam(self):
        """
        Runs in its own thread.
        Captures frames and writes msgpack data after 's' starts recording.
        """
        if self.record_camera:
            cam_path = os.path.join(self._pth, "webcam_color.msgpack")
            ts_path  = os.path.join(self._pth, "webcam_timestamp.msgpack")
            cam_file = open(cam_path, "wb")
            ts_file  = open(ts_path, "wb")

        while not self.kill_signal:
            frame = self.picam2.capture_array()
            gray  = frame[:HEIGHT, :WIDTH]
            gray  = cv2.flip(gray, 1)

            if self.record_camera and self.start_recording:
                cam_file.write(mp.packb(gray, default=mpn.encode))
                _ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                ts_file.write(
                    mp.packb([self.sync_line.get_value(), _ts])
                )

            # Lightweight preview (resize to keep display fast)
            preview = cv2.resize(gray, (250, 200))
            cv2.imshow("webcam", preview)
            sys.stdout.flush()
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        if self.record_camera:
            cam_file.close()
            ts_file.close()

    # ── Key listener (main thread) ─────────────────────────────────────────────
    def _key_listener(self):
        """Blocks until 'q' is pressed; enforced order: z → s → q"""
        print("\nControls:  [Z] Reset encoders   [S] Start recording   [Q] Quit")
        encoder_reset_done = False

        while True:
            if keyboard.is_pressed("z"):        # ← removed 'and not encoder_reset_done'
                self.enc_reset = True
                # encoder_reset_done = True       # still set True so 's' gets unlocked
                print("\n[Z] Encoder reset triggered.")
                time.sleep(0.5)  # debounce

            elif keyboard.is_pressed("s") and not self.start_recording:
                # if not encoder_reset_done:
                #     print("\n[!] Reset encoders first — press Z.")
                # else:
                #     self.start_recording = True
                #     print("\n[REC] Recording started.")
                self.enc_reset = True
                self.start_recording = True
                print("\n[REC] Recording started.")
                time.sleep(0.5)  # debounce

            if keyboard.is_pressed("q"):
                print("\n[STOP] Stopping...")
                self.kill_signal = True
                break

            time.sleep(0.01)
                    

    # ── Public entry point ─────────────────────────────────────────────────────
    def run(self):
        threads = []

        # Start encoder thread
        if self.record_sensors:
            self.serialInst.open()
            enc_thread = threading.Thread(target=self._read_serial, daemon=True)
            enc_thread.start()
            threads.append(enc_thread)
            print("  Encoder thread started.")

        # Start camera thread
        if self.record_camera:
            cam_thread = threading.Thread(target=self._capture_webcam, daemon=True)
            cam_thread.start()
            threads.append(cam_thread)
            print("  Camera thread started.")

        # Key listener runs on main thread (keyboard lib needs it)
        self._key_listener()

        # Wait for threads to finish (they check kill_signal)
        for t in threads:
            t.join(timeout=3)

        print("All threads stopped. Goodbye.")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Combined Camera + Encoder Recorder",
        description="Records PiCamera2 frames and Teensy encoder data simultaneously.",
    )
    parser.add_argument("-f", "--folder", help="Parent folder name", required=False)
    parser.add_argument("-n", "--name",   help="Recording name",     required=False)
    parser.add_argument("-c", "--camera",  help="Record camera (True/False)",  required=False)
    parser.add_argument("-s", "--sensors", help="Record sensors (True/False)", required=False)
    args = parser.parse_args()

    if not any(vars(args).values()):
        print("No arguments passed — entering interactive mode.\n")
        record_camera  = True
        record_sensors = True
        _folder_name   = "recordings"
        _name = input("Enter the name of the recording: ").strip()
    else:
        _folder_name   = args.folder
        _name          = args.name
        record_camera  = args.camera  != "False"
        record_sensors = args.sensors != "False"

    _pth = os.path.join(
        os.path.dirname(__file__), "..", "data", _folder_name, _name
    ).replace("\n", "")

    if not os.path.exists(_pth):
        os.makedirs(_pth)

    print(f"\nSave path : {_pth}")
    print(f"Camera    : {record_camera}")
    print(f"Sensors   : {record_sensors}\n")

    time.sleep(1)  # let hardware settle

    recorder = CombinedRecorder(
        _pth=_pth,
        record_camera=record_camera,
        record_sensors=record_sensors,
    )
    recorder.run()
