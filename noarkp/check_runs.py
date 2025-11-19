from camera_init import RpiCamera
from kinematics import NoarkKinematics
import threading
import time

def test_read_angles():
    camera = RpiCamera().start_thread()
    for i in range(10):
        camera.get_pose()  # Allow camera to initialize
        
    