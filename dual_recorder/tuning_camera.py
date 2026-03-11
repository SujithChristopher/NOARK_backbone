from picamera2 import Picamera2
import time

tuning = Picamera2.load_tuning_file("/home/sujith/imx219_waveshare.json")
picam2 = Picamera2(camera_num=0, tuning=tuning)
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Manually tweak gains — increase blue to kill the pink
# Format: (red_gain, blue_gain)
picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.5, 1.8)   # reduce red, boost blue
})

time.sleep(2)
picam2.capture_file("/home/sujith/test_output5.jpg")

meta = picam2.capture_metadata()
print("ColourGains used:", meta["ColourGains"])

picam2.stop()