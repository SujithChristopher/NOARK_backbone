"""
This program records data from dual webcams and teensy controller
Camera 0: IMX219 (1640x1232)
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
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        fps_value=30,
        isColor=True,
        default_res=False,
    ):
        # Initialize Camera 0 (IMX219)
        tuning_file = "/home/sujith/imx219_waveshare.json"
        
        if os.path.exists(tuning_file):
            tuning0 = Picamera2.load_tuning_file(tuning_file)
            self.picam0 = Picamera2(camera_num=0, tuning=tuning0)
        else:
            self.picam0 = Picamera2(camera_num=0)
            
        main0 = {"format": "BGR888", "size": (1640, 1232)}
        _c0 = {"FrameRate": fps_value, "ExposureTime": 5000}
        config0 = self.picam0.create_video_configuration(
            main0, controls=_c0, transform=libcamera.Transform(vflip=1)
        )
        self.picam0.configure(config0)
        self.picam0.start()

        # Apply custom colour gains for IMX219 if available
        self.picam0.set_controls({
            "AwbEnable": False,
            "ColourGains": (1.5, 1.8)
        })

        # Initialize Camera 1 (OV9281)
        self.picam1 = Picamera2(camera_num=1)
        main1 = {"format": "YUV420", "size": (1280, 800)}
        _c1 = {"FrameRate": fps_value, "ExposureTime": 5000}
        config1 = self.picam1.create_video_configuration(
            main1, controls=_c1, transform=libcamera.Transform(vflip=1)
        )
        self.picam1.configure(config1)
        self.picam1.start()

        self.record_camera = record_camera
        self.start_recording = False
        self._pth = _pth
        self.kill_signal = False
        self.fps_val = fps_value
        self.display = True

        self.isColor = isColor

        sync_pin = 17
        chip = gpiod.Chip("gpiochip4")
        self.sync_line = chip.get_line(sync_pin)
        self.sync_line.request(consumer="Button", type=gpiod.LINE_REQ_DIR_IN)

    def capture_webcam(self):
        """capture webcams"""
        if self.record_camera:
            _save_pth0 = os.path.join(self._pth, "cam0_color.msgpack")
            _save_file0 = open(_save_pth0, "wb")
            _timestamp_file0 = open(
                os.path.join(self._pth, "cam0_timestamp.msgpack"), "wb"
            )

            _save_pth1 = os.path.join(self._pth, "cam1_color.msgpack")
            _save_file1 = open(_save_pth1, "wb")
            _timestamp_file1 = open(
                os.path.join(self._pth, "cam1_timestamp.msgpack"), "wb"
            )

        while True:
            frame0 = self.picam0.capture_array()
            color_image0 = cv2.flip(frame0, 1)

            frame1 = self.picam1.capture_array()
            gray_image1 = frame1[:800, :1280]
            gray_image1 = cv2.flip(gray_image1, 1)

            if self.record_camera and self.start_recording:
                _time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                sync_val = self.sync_line.get_value()
                
                # Write Cam 0
                _packed_file0 = mp.packb(color_image0, default=mpn.encode)
                _save_file0.write(_packed_file0)
                _packed_timestamp0 = mp.packb([sync_val, _time_stamp])
                _timestamp_file0.write(_packed_timestamp0)

                # Write Cam 1
                _packed_file1 = mp.packb(gray_image1, default=mpn.encode)
                _save_file1.write(_packed_file1)
                _packed_timestamp1 = mp.packb([sync_val, _time_stamp])
                _timestamp_file1.write(_packed_timestamp1)

            if self.display:
                image_scale0 = cv2.resize(color_image0, (250, 200))
                gray_image_scale1 = cv2.resize(gray_image1, (250, 200))
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
        display = False
        _pth = None
        _folder_name = "recordings"

    else:
        print("Arguments passed")
        _folder_name = args.folder
        _name = args.name
        record_camera = args.camera
        record_sensors = args.sensors

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
        _pth=_pth, record_camera=record_camera, fps_value=30, isColor=True, default_res=True
    )
    record_data.run()
