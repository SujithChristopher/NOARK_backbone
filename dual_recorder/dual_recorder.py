"""
This program records data from dual webcams and teensy controller
Camera 0: OV9281 (1280x800)
Camera 1: OV9281 (1280x800)
"""

import cv2
import os
import datetime
import msgpack as mp
import msgpack_numpy as mpn
import argparse
import time
from picamera2 import Picamera2
import gpiod
import libcamera
import keyboard
import sys

class RecordData:
    def __init__(self, _pth=None, record_camera=True, fps_value=15, display=True, flicker_hz=50):
        # 1. Auto-detect cameras using the camera manager
        camera_list = Picamera2.global_camera_info()
        num_detected = len(camera_list)

        print(f"Detected {num_detected} cameras.")

        if num_detected < 2:
            print("Error: This script requires 2 cameras.")
            sys.exit(1)

        # 2. Identify cameras by model name
        # global_camera_info() returns dicts with 'Model', 'Num', 'Id', etc.
        self.cam_ids = [camera_list[i]['Model'].lower() for i in range(num_detected)]
        print(f"Camera IDs found: {self.cam_ids}")

        # Exposure must be an integer multiple of the AC half-period to avoid
        # banding under fluorescent/tube lights: 10000µs for 50Hz, 8333µs for 60Hz.
        exposure_time = 1_000_000 // (flicker_hz * 2)
        print(f"Flicker compensation: {flicker_hz}Hz → ExposureTime={exposure_time}µs")

        # --- Initialize Camera 0 ---
        self.picam0 = Picamera2(camera_num=0)
        res0 = (1280, 800) if "ov9281" in self.cam_ids[0] else (1640, 1232)
        fmt0 = "YUV420" if "ov9281" in self.cam_ids[0] else "BGR888"

        config0 = self.picam0.create_video_configuration(
            main={"format": fmt0, "size": res0},
            controls={"FrameRate": fps_value, "AeEnable": False, "ExposureTime": exposure_time},
            transform=libcamera.Transform(vflip=1)
        )
        self.picam0.configure(config0)
        self.picam0.start()

        # --- Initialize Camera 1 ---
        self.picam1 = Picamera2(camera_num=1)
        res1 = (1280, 800) if "ov9281" in self.cam_ids[1] else (1640, 1232)
        fmt1 = "YUV420" if "ov9281" in self.cam_ids[1] else "BGR888"

        config1 = self.picam1.create_video_configuration(
            main={"format": fmt1, "size": res1},
            controls={"FrameRate": fps_value, "AeEnable": False, "ExposureTime": exposure_time},
            transform=libcamera.Transform(vflip=1)
        )
        self.picam1.configure(config1)
        self.picam1.start()

        # Common initializations...
        self.record_camera = record_camera
        self._pth = _pth
        self.start_recording = False
        self.display = display
        
        # GPIO Setup
        sync_pin = 17
        chip = gpiod.Chip("gpiochip4")
        self.sync_line = chip.get_line(sync_pin)
        self.sync_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)

    def capture_webcam(self):
        """capture webcams"""
        if self.record_camera:
            _save_pth0 = os.path.join(self._pth, "cam0_frame.msgpack")
            _save_file0 = open(_save_pth0, "wb")
            _timestamp_file0 = open(
                os.path.join(self._pth, "cam0_timestamp.msgpack"), "wb"
            )

            _save_pth1 = os.path.join(self._pth, "cam1_frame.msgpack")
            _save_file1 = open(_save_pth1, "wb")
            _timestamp_file1 = open(
                os.path.join(self._pth, "cam1_timestamp.msgpack"), "wb"
            )

        while True:
            # Inside capture_webcam loop
            frame0 = self.picam0.capture_array()
            frame1 = self.picam1.capture_array()

            # For OV9281, capture_array usually returns (H, W, 3) 
            # even in YUV, where Y is the first channel.
            # If using YUV420, just take the Y channel:
            img0 = cv2.flip(frame0[:800, :1280], 1)
            img1 = cv2.flip(frame1[:800, :1280], 1)

            if self.record_camera and self.start_recording:
                _time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                sync_val = self.sync_line.get_value()
                
                # Write Cam 0
                _packed_file0 = mp.packb(img0, default=mpn.encode)
                _save_file0.write(_packed_file0)
                _packed_timestamp0 = mp.packb([sync_val, _time_stamp])
                _timestamp_file0.write(_packed_timestamp0)

                # Write Cam 1
                _packed_file1 = mp.packb(img1, default=mpn.encode)
                _save_file1.write(_packed_file1)
                _packed_timestamp1 = mp.packb([sync_val, _time_stamp])
                _timestamp_file1.write(_packed_timestamp1)

            if self.display:
                image_scale0 = cv2.resize(img0, (250, 200))
                gray_image_scale1 = cv2.resize(img1, (250, 200))
                cv2.imshow("webcam 0 (IMX219)", image_scale0)
                cv2.imshow("webcam 1 (OV9281)", gray_image_scale1)
                # sys.stdout.flush()
                cv2.waitKey(1)

            if keyboard.is_pressed("s") and not self.start_recording:
                print("You Pressed a Key!, started recording from dual webcams")
                self.start_recording = True

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                if self.record_camera:
                    _save_file0.close()
                    _timestamp_file0.close()
                    _save_file1.close()
                    _timestamp_file1.close()
                self.picam0.stop()
                self.picam1.stop()
                break

    def run(self):
        """run the program"""
        self.capture_webcam()

if __name__ == "__main__":
    """get parameter from external program"""

    parser = argparse.ArgumentParser(
        prog="Dual camera recorder",
        description="Records data from two cameras simultaneously",
        epilog="Captures to cam0_color.msgpack and cam1_color.msgpack",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument("-s", "--sensors", help="record sensors", required=False)

    args = parser.parse_args()

    # if your not passing any arguments then the default values will be used
    if not any(vars(args).values()):
        print("No arguments passed, please enter manually")

        record_camera = True
        record_sensors = False

        if record_camera or record_sensors:
            _name = input("Enter the name of the recording: ")
        display = True
        _pth = None
        _folder_name = "recordings"

    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera
        record_sensors = args.sensors
        display = True

        if record_camera == "True":
            record_camera = True
        else:
            record_camera = False

    if record_camera or record_sensors:
        _pth = os.path.join(
            os.path.dirname(__file__), "..", "data", _folder_name, _name
        )

        if "\n" in _pth:
            _pth = _pth.replace("\n", "")

        if not os.path.exists(_pth):
            os.makedirs(_pth)
    time.sleep(1)

    record_data = RecordData(
        _pth=_pth, record_camera=record_camera, fps_value=30, display=display
    )
    record_data.run()
