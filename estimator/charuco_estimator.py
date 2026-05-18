import cv2
import os
import toml
import numpy as np
import time
import argparse
import sys
from picamera2 import Picamera2
import libcamera
import keyboard

# Default frame size if not specified in config
WIDTH = 1280
HEIGHT = 800

class RealtimeCharucoTracker:
    def __init__(self, config_path, squares_x=6, squares_y=4, square_length=0.028, marker_length=0.02, dict_id=cv2.aruco.DICT_4X4_50):
        self.config_path = config_path
        
        # ChArUco Board Parameters
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.dict_id = dict_id
    
        self._load_config()
        self._init_camera()
        self._init_undistortion_maps()
        self._init_detector()

        self.rvec = None
        self.tvec = None

    def _load_config(self):
        """Loads calibration and application settings from the TOML config file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        print(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = toml.load(f)

        # Camera Intrinsics
        self.camera_matrix = np.array(self.config["calibration"]["camera_matrix"])
        self.dist_coeffs = np.array(self.config["calibration"]["dist_coeffs"])
        self.resolution = tuple(self.config["camera"]["resolution"])
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = self.resolution
             
        # Display settings
        self.display = self.config.get("display", {}).get("display", True) 

    def _init_camera(self):
        """Initializes the Picamera2 instance with OV9281 specific settings."""
        print(f"Initializing Picamera2 for resolution {WIDTH}x{HEIGHT}...")
        self.picam2 = Picamera2()
        main_config = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
        
        # Exposure / Framerate settings from reference
        controls = {"FrameRate": 100, "ExposureTime": 9000}
        
        config = self.picam2.create_video_configuration(
            main_config, 
            controls=controls, 
            transform=libcamera.Transform(vflip=1)
        )
        
        self.picam2.configure(config)
        self.picam2.start()
        print("Camera initialized and started.")

    def _init_undistortion_maps(self):
        """Pre-computes the undistortion maps for the fisheye lens."""
        print("Pre-computing fisheye undistortion maps...")
        
        # Compute a new camera matrix. Balance=0.5 for 160 FOV is a good compromise.
        # It retains mostly valid pixels but crops extremely stretched borders.
        # Alternatively, balance=1.0 retains all pixels for maximum corner detection.
        self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix, 
            self.dist_coeffs, 
            self.resolution, 
            np.eye(3), 
            balance=1.0
        )
        
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, 
            self.dist_coeffs, 
            np.eye(3), 
            self.new_camera_matrix, 
            self.resolution, 
            cv2.CV_16SC2
        )
        print("Undistortion maps ready.")

    def _init_detector(self):
        """Initializes the ChArUco board detector (OpenCV 4.7+ API)."""
        print(f"Initializing ChArUco ({self.squares_x}x{self.squares_y}) detector...")
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dict_id)
        
        # Create the ChArUco Board object
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), 
            self.square_length, 
            self.marker_length, 
            self.dictionary
        )
        # self.board.setLegacyPattern(True)
        # Detector Parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        
        # Parameters for Charuco Detector
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

    def process_frame(self):
        """Main processing loop."""
        print("Starting real-time processing loop. Press 'q' to quit.")

        # try:
        while True:
            # Capture YUV420 and extract the Y (grayscale) channel
            frame = self.picam2.capture_array()
            # The format is YUV420. The first (HEIGHT x WIDTH) bytes are the Y channel.
            gray_image = frame[:HEIGHT, :WIDTH]
            
            # Flip horizontally
            gray_image = cv2.flip(gray_image, 1)

            # Undistort the entire image
            undistorted_gray = cv2.remap(
                gray_image, 
                self.map1, 
                self.map2, 
                interpolation=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT
            )
            
            # Detect markers first
            corners, ids, rejected = self.detector.detectMarkers(undistorted_gray)

            display_img = cv2.cvtColor(undistorted_gray, cv2.COLOR_GRAY2BGR)
            
            if ids is not None and len(ids) > 0:
                # Draw Aruco markers
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                
                # Interpolate Charuco Corners
                # Notice we use the standard API for OpenCV 4.7+
                # charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(undistorted_gray)
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(undistorted_gray)
                
                # print(self.new_camera_matrix)
                # print(charuco_ids)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 4:
                    # Draw the detected charuco corners
                    cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (0, 0, 255))
                    
                    # Estimate the pose of the multi-marker board
                    # We use the new camera matrix and NO distortion (since the image is already undistorted)
                    success, self.rvec, self.tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, 
                        charuco_ids, 
                        self.board, 
                        self.new_camera_matrix,
                        # np.eye(3), 
                        np.zeros((4,1)), # No dist_coeffs
                        None, None
                    )

                    
                    if success:
                        # Draw axis  
                        cv2.drawFrameAxes(
                            display_img, 
                            self.new_camera_matrix, 
                            np.zeros((4,1)), # No dist_coeffs
                            # None, 
                            self.rvec, 
                            self.tvec, 
                            length=self.square_length, 
                            thickness=2
                        )
                    #     # Print pose
                    # print(success)
                    print(f"ChArUco Pose: tvec={self.tvec.ravel()}")
   
            
            # Resize for display to avoid filling the screen
            display_img_resized = cv2.resize(display_img, (480, 320))
            cv2.imshow("Realtime ChArUco Detection (Undistorted)", display_img_resized)
            cv2.waitKey(1) # Needed to update the display and check for key presses
            if keyboard.is_pressed('s'):
                print("Saving calibration data...")
                self.save_calibration_data(self.rvec, self.tvec)
            if keyboard.is_pressed('q'):
                print("Exiting...")
                break
                    
        # except KeyboardInterrupt:
        #     print("\nStopped by user.")
        # finally:
        #     self.cleanup()

    def save_calibration_data(self, rvec, tvec):
        """Saves the calibration data to a TOML file."""
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            'rotation_matrix': cv2.Rodrigues(rvec)[0].tolist(),
        }

        output_dir = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose"
        output_path = os.path.join(output_dir, "charuco_pose_picam.toml")
        
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(calibration_data, f)
        print("Calibration data saved.")

    def cleanup(self):
        """Releases camera resources and destroys windows."""
        print("Cleaning up resources...")
        self.picam2.stop()
        if self.display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Fisheye ChArUco Board detection using Picamera2")
    parser.add_argument(
        "-c", "--config", 
        help="Path to the TOML configuration file", 
        default=os.path.join(os.path.dirname(__file__), "..", "notebooks", "calibration", "output", "good.toml")
    )
    
    # Optional arguments for ChArUco parameters
    parser.add_argument("--squares_x", type=int, default=6, help="Number of squares in X direction")
    parser.add_argument("--squares_y", type=int, default=4, help="Number of squares in Y direction")
    parser.add_argument("--square_len", type=float, default=0.028, help="Square side length (in meters)")
    parser.add_argument("--marker_len", type=float, default=0.020, help="Marker side length (in meters)")
    
    args = parser.parse_args()

    # Create absolute path if relative
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    try:
        tracker = RealtimeCharucoTracker(
            config_path, 
            squares_x=args.squares_x, 
            squares_y=args.squares_y, 
            square_length=args.square_len, 
            marker_length=args.marker_len
        )
        tracker.process_frame()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
