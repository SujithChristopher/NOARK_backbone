"""
this program records data from webcam and teensy controller
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

frame_size = (1280, 800)

WIDTH = frame_size[0]
HEIGHT = frame_size[1]


class RecordData:
    def __init__(
        self,
        _pth=None,
        record_camera=True,
        fps_value=60,
        isColor=True,
        default_res=False,
    ):
        self.picam2 = Picamera2()
        main = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
        _c = {"FrameRate": 100, "ExposureTime": 5000}
        config = self.picam2.create_video_configuration(
            main, controls=_c, transform=libcamera.Transform(vflip=1)
        )
        self.picam2.configure(config)
        self.picam2.start()

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
        """capture webcam"""

        if self.record_camera:
            _save_pth = os.path.join(self._pth, "webcam_color.msgpack")
            _save_file = open(_save_pth, "wb")
            _timestamp_file = open(
                os.path.join(self._pth, "webcam_timestamp.msgpack"), "wb"
            )

        while True:
            frame = self.picam2.capture_array()
            gray_image = frame[:HEIGHT, :WIDTH]
            gray_image = cv2.flip(gray_image, 1)

            if self.record_camera and self.start_recording:
                _packed_file = mp.packb(gray_image, default=mpn.encode)
                _save_file.write(_packed_file)
                _time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                _packed_timestamp = mp.packb([self.sync_line.get_value(), _time_stamp])
                _timestamp_file.write(_packed_timestamp)

            if self.display:
                gray_image_ = cv2.resize(gray_image, (250, 200))
                cv2.imshow("webcam", gray_image_)
                sys.stdout.flush()
                cv2.waitKey(1)

            if keyboard.is_pressed("s"):
                print("You Pressed a Key!, started recording from webcam")
                self.start_recording = True

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                if self.record_camera:
                    _save_file.close()
                    _timestamp_file.close()
                break

    def run(self):
        """run the program"""
        self.capture_webcam()


if __name__ == "__main__":
    """get parameter from external program"""

    parser = argparse.ArgumentParser(
        prog="Single camera recorder",
        description="This basically records data from the camera and the sensors",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--folder", help="folder name", required=False)
    parser.add_argument("-n", "--name", help="name of the file", required=False)
    parser.add_argument("-c", "--camera", help="record camera", required=False)
    parser.add_argument("-s", "--sensors", help="record sensors", required=False)

    args = parser.parse_args()

    # if your not passing any arguments then the default values will be used
    # and you may have to enter the folder name and the name of the recording
    if not any(vars(args).values()):
        print("No arguments passed, please enter manually")

        """Enter the respective parameters"""
        record_camera = True
        record_sensors = False

        if record_camera or record_sensors:
            _name = input("Enter the name of the recording: ")
        display = False
        _pth = None  # this is default do not change, path gets updated by your input
        _folder_name = (
            "recordings"  # this is the parent folder name where the data will be saved
        )

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
        if record_sensors == "True":
            record_sensors = True
        else:
            record_sensors = False

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
        _pth=_pth, record_camera=record_camera, isColor=True, default_res=True
    )
    record_data.run()
