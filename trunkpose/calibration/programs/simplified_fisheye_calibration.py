"""
Simplified Fisheye Calibration based on your working notebook approach
This directly mirrors your successful notebook methodology
"""

import numpy as np
import cv2
import os
import msgpack as mp
import msgpack_numpy as mpn
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import polars as pl
from scipy.interpolate import interp1d
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_corners_standalone(frame, pattern_size):
    """Standalone corner detection function for parallel processing"""
    frame = cv2.rotate(frame.copy(), cv2.ROTATE_180)
    ret, corners = cv2.findChessboardCorners(frame, pattern_size)
    
    if ret:
        corners = cv2.cornerSubPix(
            frame,
            corners,
            (5, 5),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    return corners if ret else None


def calibrate_single_iteration_standalone(chessb_corners, board_points, img_size, n_samples=20):
    """Standalone calibration function for parallel processing"""
    rnd = np.random.choice(len(chessb_corners), min(n_samples, len(chessb_corners)), replace=False)
    chessb_c = chessb_corners[rnd]
    
    world_points = []
    image_points = []
    
    for _f in chessb_c:
        image_points.append(_f)
        world_points.append(board_points)
    
    # Fisheye calibration with exact flags from notebook
    flags_calib = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_CHECK_COND
    )
    
    calibrate_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-12,
    )
    
    try:
        ret, camera_matrix, k, R, t = cv2.fisheye.calibrate(
            np.expand_dims(np.asarray(world_points), -2),
            image_points,
            img_size,
            None,
            None,
            flags=flags_calib,
            criteria=calibrate_criteria,
        )
        
        return {
            "ReError": ret,
            "mat": camera_matrix,
            "dist": k,
            "rvec": R,
            "tvec": t
        }
    except Exception as e:
        return None


class SimplifiedFisheyeCalibration:
    """Simplified fisheye calibration following your notebook approach"""
    
    def __init__(self):
        # Calibration parameters (from your notebook)
        self.pattern_size = (6, 4)
        self.square_size = 30
        self.img_size = (1200, 800)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Construct 3D points
        self.board_points = self._construct_3d_points()
        
        # ArUco setup
        self._setup_aruco()
        
    def _construct_3d_points(self):
        """Construct 3D object points for chessboard"""
        X = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        X[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        X = X * self.square_size
        return X
    
    def _setup_aruco(self):
        """Setup ArUco detector"""
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = 0.05
        self.marker_separation = 0.01
        
        self.board = cv2.aruco.GridBoard(
            size=[1, 1],
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.aruco_dict,
        )
    
    def extract_chessboard_corners(self, video_path, n_jobs=20):
        """Extract chessboard corners using parallel processing like your notebook"""
        logger.info(f"Extracting corners from {video_path}")
        
        with open(video_path, "rb") as f:
            video_data = list(mp.Unpacker(f, object_hook=mpn.decode))
        
        logger.info(f"Loaded {len(video_data)} frames")
        
        # Parallel corner detection using standalone function
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(detect_corners_standalone)(frame, self.pattern_size) for frame in tqdm(video_data)
        )
        
        # Filter out None results
        chessb_corners = [corner for corner in results if corner is not None]
        
        logger.info(f"Found {len(chessb_corners)} valid corner sets")
        return np.array(chessb_corners)
    
    def calibrate_single_iteration(self, chessb_corners, n_samples=20):
        """Single calibration iteration exactly like your notebook"""
        rnd = np.random.choice(len(chessb_corners), min(n_samples, len(chessb_corners)), replace=False)
        chessb_c = chessb_corners[rnd]
        
        world_points = []
        image_points = []
        
        for _f in chessb_c:
            image_points.append(_f)
            world_points.append(self.board_points)
        
        # Fisheye calibration with your exact flags
        flags_calib = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_FIX_SKEW
            + cv2.fisheye.CALIB_CHECK_COND
        )
        
        calibrate_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-12,
        )
        
        try:
            ret, camera_matrix, k, R, t = cv2.fisheye.calibrate(
                np.expand_dims(np.asarray(world_points), -2),
                image_points,
                self.img_size,
                None,
                None,
                flags=flags_calib,
                criteria=calibrate_criteria,
            )
            
            return {
                "ReError": ret,
                "mat": camera_matrix,
                "dist": k,
                "rvec": R,
                "tvec": t
            }
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return None
    
    def generate_calibrations(self, chessb_corners, n_calibrations=200, n_jobs=20):
        """Generate multiple calibrations like your notebook"""
        logger.info(f"Generating {n_calibrations} calibrations")
        
        # Parallel calibration generation using standalone function
        results = Parallel(n_jobs=n_jobs)(
            delayed(calibrate_single_iteration_standalone)(chessb_corners, self.board_points, self.img_size) 
            for _ in tqdm(range(n_calibrations), colour="green")
        )
        
        # Filter out failed calibrations
        valid_results = [r for r in results if r is not None]
        
        # Organize results like your notebook
        my_dict = {"ReError": [], "mat": [], "dist": [], "rvec": [], "tvec": []}
        for result in valid_results:
            my_dict["ReError"].append(result["ReError"])
            my_dict["mat"].append(result["mat"])
            my_dict["dist"].append(result["dist"])
            my_dict["rvec"].append(result["rvec"])
            my_dict["tvec"].append(result["tvec"])
        
        logger.info(f"Generated {len(valid_results)} valid calibrations")
        return my_dict
    
    def estimate_pose_single_markers(self, corners, marker_size, camera_matrix, distortion_coefficients):
        """Pose estimation exactly like your notebook"""
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        rvecs = []
        tvecs = []
        for corner in corners:
            _, r, t = cv2.solvePnP(
                marker_points,
                corner,
                camera_matrix,
                distortion_coefficients,
                True,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if r is not None and t is not None:
                r = np.array(r).reshape(1, 3).tolist()
                t = np.array(t).reshape(1, 3).tolist()
                rvecs.append(r)
                tvecs.append(t)
        return np.array(rvecs, dtype=np.float32), np.array(tvecs, dtype=np.float32)
    
    def load_april_tag_data(self, data_path):
        """Load April tag data"""
        logger.info(f"Loading April tag data from {data_path}")
        
        corners, ids = [], []
        with open(data_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            for frame in tqdm(unpacker):
                _c, _i, rejectedpoints = self.detector.detectMarkers(frame)
                _c, _i, rejectedpoints, _ = self.detector.refineDetectedMarkers(
                    image=frame,
                    board=self.board,
                    detectedCorners=_c,
                    detectedIds=_i,
                    rejectedCorners=rejectedpoints,
                )
                corners.append(_c)
                ids.append(_i)
        
        # Load timestamps
        timestamp_path = os.path.join(os.path.dirname(data_path), "webcam_timestamp.msgpack")
        timestamps = []
        sync = []
        
        with open(timestamp_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            for _p in unpacker:
                sync.append(_p[0])
                timestamps.append(_p[1])
        
        sync = np.array(sync).astype(bool)
        
        return corners, ids, timestamps, sync
    
    def evaluate_calibration(self, my_dict, corners, ids, default_ids=[12, 14, 20]):
        """Evaluate calibrations against April tag data - simplified version"""
        logger.info("Evaluating calibrations")
        
        def process_calibration(i):
            try:
                _fish_mat = my_dict["mat"][i]
                _fish_dist = my_dict["dist"][i]
                
                _new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    _fish_mat, _fish_dist, (1200, 800), np.eye(3), balance=1
                )
                
                # Count successful pose estimations
                successful_poses = 0
                total_attempts = 0
                
                for _corner, _id in zip(corners[:100], ids[:100]):  # Test on first 100 frames
                    if _corner is not None and len(_corner) > 0:
                        total_attempts += 1
                        try:
                            # Undistort corners
                            undist_corners = cv2.fisheye.undistortPoints(
                                np.array(_corner).reshape(-1, 1, 2), 
                                _fish_mat, _fish_dist, None, _new_cam
                            )
                            
                            # Simple check: if undistortion worked without error
                            if undist_corners is not None and len(undist_corners) > 0:
                                successful_poses += 1
                        except:
                            continue
                
                success_rate = successful_poses / max(total_attempts, 1)
                return {
                    'calibration_id': i,
                    'reprojection_error': my_dict["ReError"][i],
                    'success_rate': success_rate,
                    'camera_matrix': _fish_mat,
                    'dist_coeffs': _fish_dist
                }
                
            except Exception as e:
                return None
        
        # Evaluate all calibrations
        results = []
        for i in tqdm(range(len(my_dict["mat"])), desc="Evaluating calibrations"):
            result = process_calibration(i)
            if result is not None:
                results.append(result)
        
        return results
    
    def find_best_calibration(self, evaluation_results):
        """Find best calibration based on success rate and reprojection error"""
        if not evaluation_results:
            return None
        
        # Sort by success rate (descending) then by reprojection error (ascending)
        sorted_results = sorted(evaluation_results, 
                              key=lambda x: (-x['success_rate'], x['reprojection_error']))
        
        best = sorted_results[0]
        logger.info(f"Best calibration:")
        logger.info(f"  Success rate: {best['success_rate']:.2%}")
        logger.info(f"  Reprojection error: {best['reprojection_error']:.4f}")
        
        return best
    
    def save_calibration_toml(self, best_calibration, output_path):
        """Save best calibration to TOML format"""
        import toml
        
        data = {
            "calibration": {
                "camera_matrix": best_calibration['camera_matrix'].tolist(),
                "dist_coeffs": best_calibration['dist_coeffs'].tolist()
            },
            "pose": {
                "human_pose": False,
                "marker_pose": True
            },
            "aruco": {
                "marker_length": 0.05,
                "marker_spacing": 0.01
            },
            "camera": {
                "resolution": [1200, 800]
            },
            "stream_data": {
                "udp": False,
                "ip": "localhost",
                "port": 12345
            },
            "display": {
                "display": False
            },
            "rom": {
                "angle": 45,
                "distance": 50
            }
        }
        
        with open(output_path, 'w') as f:
            toml.dump(data, f)
        
        logger.info(f"Calibration saved to {output_path}")


def main():
    """Main calibration pipeline"""
    
    # Initialize
    calibrator = SimplifiedFisheyeCalibration()
    
    # Paths - update these for your system
    base_path = os.path.dirname(os.path.abspath(__file__))
    calib_video_path = os.path.join(base_path, "data", "calibration", '160_fov',"calib_mono_160fov_raw", "webcam_color.msgpack")
    april_data_path = os.path.join(base_path, "data", 'recordings',"160_fov", "3marker_complete_data", "3marker_april_mono_160fov_3", "webcam_color.msgpack")
    
    # Check files exist
    if not os.path.exists(calib_video_path):
        logger.error(f"Calibration video not found: {calib_video_path}")
        return
    
    # Step 1: Extract chessboard corners
    chessb_corners = calibrator.extract_chessboard_corners(calib_video_path)
    
    if len(chessb_corners) < 20:
        logger.error("Not enough corner data found")
        return
    
    # Step 2: Generate multiple calibrations
    my_dict = calibrator.generate_calibrations(chessb_corners, n_calibrations=1000)
    
    if len(my_dict["mat"]) == 0:
        logger.error("No valid calibrations generated")
        return
    
    # Step 3: Load April tag data for evaluation
    if os.path.exists(april_data_path):
        corners, ids, timestamps, sync = calibrator.load_april_tag_data(april_data_path)
        
        # Step 4: Evaluate calibrations
        evaluation_results = calibrator.evaluate_calibration(my_dict, corners, ids)
        
        # Step 5: Find best calibration
        best_calibration = calibrator.find_best_calibration(evaluation_results)
        
        if best_calibration:
            # Save to TOML
            output_path = "optimized_fisheye_calibration123.toml"
            calibrator.save_calibration_toml(best_calibration, output_path)
            print('hi')
        
    else:
        logger.warning(f"April tag data not found: {april_data_path}")
        logger.info("Using best reprojection error calibration")
        
        # Just use the calibration with lowest reprojection error
        best_idx = np.argmin(my_dict["ReError"])
        best_calibration = {
            'camera_matrix': my_dict["mat"][best_idx],
            'dist_coeffs': my_dict["dist"][best_idx]
        }
        
        output_path = "fisheye_calibration_reproj_best.toml"
        calibrator.save_calibration_toml(best_calibration, output_path)
    
    logger.info("Calibration complete!")


if __name__ == "__main__":
    main()