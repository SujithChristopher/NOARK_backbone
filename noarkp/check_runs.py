from camera_init import RpiCamera
from kinematics import NoarkKinematics

def test_read_angles():
    kinematics = NoarkKinematics(serial_port='COM3', baudrate=9600)
    angles = kinematics.read_angles()
    assert len(angles) == 2  # Assuming there are two angles to read
    for angle in angles:
        assert isinstance(angle, float)  # Each angle should be a float
        
    