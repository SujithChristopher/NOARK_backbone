import cv2
import os
import toml
import numpy as np
import time
import argparse
import sys
from picamera2 import Picamera2
import libcamera

# Default frame size if not specified in config
WIDTH = 1280
HEIGHT = 800

class RealtimeFisheyeAprilTagTracker:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()
        self._init_camera()
        self._init_undistortion_maps()
        self._init_detector()

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

        # Aruco/AprilTag Settings
        self.marker_length = self.config["aruco"]["marker_length"]
        # Use a default marker length if not properly set
        if not isinstance(self.marker_length, (int, float)) or self.marker_length <= 0:
             self.marker_length = 0.05 # 5cm default
             
        # Display settings
        self.display = self.config.get("display", {}).get("display", True)

    def _init_camera(self):
        """Initializes the Picamera2 instance with OV9281 specific settings."""
        print(f"Initializing Picamera2 for resolution {WIDTH}x{HEIGHT}...")
        self.picam2 = Picamera2()
        main_config = {"format": "YUV420", "size": (WIDTH, HEIGHT)}
        
        # Exposure / Framerate settings from reference
        controls = {"FrameRate": 100, "ExposureTime": 3000}
        
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
        
        # We compute a new camera matrix. 
        # Using balance=0.0 to retain only valid pixels (no black borders) if desired, 
        # or balance=1.0 to retain all pixels (some might be black).
        # We'll stick to a balance of 0.5 as a middle ground for 160 FOV.
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
        """Initializes the AprilTag detector."""
        # The user confirmed DICT_APRILTAG_36h11
        print("Initializing AprilTag (36h11) detector...")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        
        # Optimize detection parameters if necessary based on environment
        # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def process_frame(self):
        """Main processing loop."""
        print("Starting real-time processing loop. Press 'q' to quit.")
        
        prev_time = time.time()
        fps_filter = 0.0

        try:
            while True:
                # Capture YUV420 and extract the Y (grayscale) channel
                frame = self.picam2.capture_array()
                # The format is YUV420. The first (HEIGHT x WIDTH) bytes are the Y channel.
                gray_image = frame[:HEIGHT, :WIDTH]
                
                # Flip horizontally to match the reference recorder.py
                gray_image = cv2.flip(gray_image, 1)

                # Undistort the entire image
                undistorted_gray = cv2.remap(
                    gray_image, 
                    self.map1, 
                    self.map2, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT
                )

                # Detect markers
                corners, ids, rejected = self.detector.detectMarkers(undistorted_gray)

                if self.display:
                    # Create a color image for drawing overlays
                    display_img = cv2.cvtColor(undistorted_gray, cv2.COLOR_GRAY2BGR)
                    
                    if ids is not None:
                        # Draw bounding boxes
                        cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
                        
                        # Estimate pose for each marker
                        # Note: OpenCV >= 4.7 uses solvePnP instead of estimatePoseSingleMarkers
                        # We build an object points array for a single marker
                        marker_len = self.marker_length
                        obj_points = np.array([
                            [-marker_len / 2, marker_len / 2, 0],
                            [marker_len / 2, marker_len / 2, 0],
                            [marker_len / 2, -marker_len / 2, 0],
                            [-marker_len / 2, -marker_len / 2, 0]
                        ], dtype=np.float32)

                        for i in range(len(ids)):
                            # Solve PnP using the NEW camera matrix (since the image is undistorted)
                            success, rvec, tvec = cv2.solvePnP(
                                obj_points, 
                                corners[i][0], 
                                self.new_camera_matrix, 
                                None, # No distortion coeffs, image is already undistorted
                                flags=cv2.SOLVEPNP_IPPE_SQUARE
                            )
                            
                            if success:
                                # Draw axis
                                cv2.drawFrameAxes(
                                    display_img, 
                                    self.new_camera_matrix, 
                                    None, 
                                    rvec, 
                                    tvec, 
                                    length=marker_len * 0.5, 
                                    thickness=2
                                )
                                
                                # Print pose (optional, can be noisy in terminal)
                                # print(f"Marker {ids[i][0]}: tvec={tvec.ravel()}, rvec={rvec.ravel()}")

                    # Calculate FPS
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time)
                    prev_time = curr_time
                    fps_filter = fps_filter * 0.9 + fps * 0.1
                    
                    cv2.putText(
                        display_img, 
                        f"FPS: {fps_filter:.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )

                    # Resize for display to avoid filling the screen
                    display_img_resized = cv2.resize(display_img, (640, 400))
                    cv2.imshow("Realtime AprilTag Detection (Undistorted)", display_img_resized)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Releases camera resources and destroys windows."""
        print("Cleaning up resources...")
        self.picam2.stop()
        if self.display:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Fisheye AprilTag detection using Picamera2")
    parser.add_argument(
        "-c", "--config", 
        help="Path to the TOML configuration file", 
        default=os.path.join(os.path.dirname(__file__), "..", "notebooks", "calibration", "output", "good.toml")
    )
    args = parser.parse_args()

    # Create absolute path if relative
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    try:
        tracker = RealtimeFisheyeAprilTagTracker(config_path)
        tracker.process_frame()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
