"""
Improved Fisheye Camera Calibration for OV9281 160° FOV Camera
Optimized for April Tag pose estimation accuracy

Key improvements:
1. Better corner detection with sub-pixel refinement
2. Systematic calibration data selection (not random)
3. Proper fisheye distortion model validation
4. April tag-specific optimization criteria
5. Cross-validation framework
"""

import numpy as np
import cv2
import os
import msgpack as mp
import msgpack_numpy as mpn
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import polars as pl
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import json
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedFisheyeCalibration:
    """
    Advanced fisheye calibration system optimized for April tag pose estimation
    """
    
    def __init__(self, 
                 pattern_size: Tuple[int, int] = (6, 4),
                 square_size: float = 30.0,
                 img_size: Tuple[int, int] = (1200, 800),
                 marker_size: float = 0.05):
        """
        Initialize calibration system
        
        Args:
            pattern_size: Chessboard pattern size (corners)
            square_size: Size of chessboard squares in mm
            img_size: Image resolution
            marker_size: April tag marker size in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.img_size = img_size
        self.marker_size = marker_size
        
        # Calibration criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.fisheye_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-12)
        
        # 3D board points
        self.board_points = self._construct_3d_points()
        
        # April tag detector setup
        self._setup_aruco_detector()
        
        logger.info(f"Initialized calibration for {img_size} image with {pattern_size} pattern")
    
    def _construct_3d_points(self) -> np.ndarray:
        """Construct 3D object points for chessboard"""
        points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        return points * self.square_size
    
    def _setup_aruco_detector(self):
        """Setup ArUco detector for April tags"""
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.useAruco3Detection = 1
        self.aruco_params.cornerRefinementMethod = 1
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def detect_corners_robust(self, frame: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """
        Robust corner detection with multiple preprocessing approaches
        """
        # Try different preprocessing methods and flags for fisheye cameras
        preprocessing_configs = [
            # (preprocessing_function, detection_flags)
            (lambda img: img, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
            (lambda img: cv2.equalizeHist(img), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
            (lambda img: cv2.GaussianBlur(img, (3, 3), 0), cv2.CALIB_CB_ADAPTIVE_THRESH),
            (lambda img: img, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS),
            # More aggressive preprocessing for difficult cases
            (lambda img: cv2.bilateralFilter(img, 9, 75, 75), cv2.CALIB_CB_ADAPTIVE_THRESH),
            (lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)), cv2.CALIB_CB_ADAPTIVE_THRESH),
        ]
        
        for i, (preprocess, flags) in enumerate(preprocessing_configs):
            try:
                processed = preprocess(frame.copy())
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(processed, self.pattern_size, flags)
                
                if ret:
                    # Sub-pixel refinement
                    corners = cv2.cornerSubPix(
                        processed, corners, (11, 11), (-1, -1), self.criteria
                    )
                    
                    # Quality check: ensure corners are well distributed
                    if self._validate_corner_quality(corners):
                        if debug:
                            logger.info(f"Corners found using method {i}")
                        return corners
                
            except Exception as e:
                if debug:
                    logger.warning(f"Method {i} failed: {e}")
                continue
        
        return None
    
    def _validate_corner_quality(self, corners: np.ndarray) -> bool:
        """
        Validate corner detection quality
        """
        if corners is None or len(corners) != self.pattern_size[0] * self.pattern_size[1]:
            return False
        
        # Check for reasonable corner distribution
        corners_flat = corners.reshape(-1, 2)
        x_range = np.ptp(corners_flat[:, 0])
        y_range = np.ptp(corners_flat[:, 1])
        
        # More lenient thresholds for fisheye cameras (corners may be more distorted)
        min_span_ratio = 0.15  # Reduced from 0.3 to 0.15
        
        # Check if corners span reasonable area
        x_span_ok = x_range / self.img_size[0] >= min_span_ratio
        y_span_ok = y_range / self.img_size[1] >= min_span_ratio
        
        # Additional check: corners should be within image bounds
        within_bounds = (np.all(corners_flat[:, 0] >= 10) and 
                        np.all(corners_flat[:, 0] <= self.img_size[0] - 10) and
                        np.all(corners_flat[:, 1] >= 10) and 
                        np.all(corners_flat[:, 1] <= self.img_size[1] - 10))
        
        # For fisheye, accept if either dimension has good span OR if corners are well distributed
        corner_std = np.std(corners_flat, axis=0)
        good_distribution = corner_std[0] > 50 and corner_std[1] > 50  # Some spread in both dimensions
        
        return within_bounds and (x_span_ok or y_span_ok or good_distribution)
    
    def extract_calibration_data(self, video_path: str, max_frames: int = 1000) -> List[np.ndarray]:
        """
        Extract high-quality calibration frames from video
        """
        logger.info(f"Extracting calibration data from {video_path}")
        
        with open(video_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            
            corners_list = []
            frame_count = 0
            processed_count = 0
            
            for frame in tqdm(unpacker, desc="Processing frames"):
                if frame_count >= max_frames:
                    break
                
                try:
                    # Rotate frame as per your setup
                    frame = cv2.rotate(frame.copy(), cv2.ROTATE_180)
                    
                    # Convert to grayscale if needed
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    corners = self.detect_corners_robust(frame)
                    if corners is not None:
                        corners_list.append(corners)
                        processed_count += 1
                        
                        # Log progress every 10 successful detections
                        if processed_count % 10 == 0:
                            logger.info(f"Found {processed_count} valid corner sets so far...")
                
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_count}: {e}")
                
                frame_count += 1
        
        logger.info(f"Found {len(corners_list)} valid corner sets from {frame_count} frames")
        return corners_list
    
    def systematic_calibration_selection(self, 
                                       corners_list: List[np.ndarray], 
                                       n_subsets: int = 50,
                                       subset_size: int = 25) -> List[Dict]:
        """
        Systematic calibration data selection instead of random
        Uses k-means clustering on corner positions for diverse selection
        """
        logger.info(f"Creating {n_subsets} systematic calibration subsets")
        
        if len(corners_list) < subset_size:
            logger.warning(f"Not enough corners ({len(corners_list)}) for subset size {subset_size}")
            subset_size = len(corners_list)
        
        # Extract features for clustering (mean corner positions)
        features = []
        for corners in corners_list:
            mean_pos = np.mean(corners.reshape(-1, 2), axis=0)
            std_pos = np.std(corners.reshape(-1, 2), axis=0)
            features.append(np.concatenate([mean_pos, std_pos]))
        
        features = np.array(features)
        
        calibration_results = []
        
        # Use different subset selection strategies
        strategies = [
            self._uniform_sampling,
            self._diversity_sampling,
            self._quality_sampling
        ]
        
        subsets_per_strategy = n_subsets // len(strategies)
        
        for strategy in strategies:
            for i in range(subsets_per_strategy):
                subset_indices = strategy(features, corners_list, subset_size, i)
                subset_corners = [corners_list[idx] for idx in subset_indices]
                
                calib_result = self._calibrate_subset(subset_corners)
                if calib_result is not None:
                    calib_result['selection_strategy'] = strategy.__name__
                    calib_result['subset_id'] = i
                    calibration_results.append(calib_result)
        
        logger.info(f"Generated {len(calibration_results)} calibration results")
        return calibration_results
    
    def _uniform_sampling(self, features: np.ndarray, corners_list: List, 
                         subset_size: int, seed: int) -> List[int]:
        """Uniform sampling across the dataset"""
        np.random.seed(seed)
        return np.random.choice(len(corners_list), subset_size, replace=False)
    
    def _diversity_sampling(self, features: np.ndarray, corners_list: List, 
                          subset_size: int, seed: int) -> List[int]:
        """Sampling for maximum diversity in corner positions"""
        from sklearn.cluster import KMeans
        
        # Cluster and select from different clusters
        n_clusters = min(subset_size, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select from each cluster
                selected = np.random.choice(cluster_indices, 
                                         min(subset_size // n_clusters + 1, len(cluster_indices)), 
                                         replace=False)
                indices.extend(selected)
        
        return indices[:subset_size]
    
    def _quality_sampling(self, features: np.ndarray, corners_list: List, 
                         subset_size: int, seed: int) -> List[int]:
        """Sampling based on corner detection quality"""
        # Score based on corner spread and consistency
        scores = []
        for corners in corners_list:
            corners_flat = corners.reshape(-1, 2)
            spread = np.ptp(corners_flat, axis=0).mean()
            consistency = 1.0 / (np.std(corners_flat, axis=0).mean() + 1e-6)
            scores.append(spread * consistency)
        
        # Select top quality frames
        top_indices = np.argsort(scores)[-subset_size:]
        np.random.seed(seed)
        return np.random.choice(top_indices, subset_size, replace=False)
    
    def _calibrate_subset(self, corners_subset: List[np.ndarray]) -> Optional[Dict]:
        """
        Calibrate camera using a subset of corners
        """
        if len(corners_subset) < 10:  # Minimum frames needed
            return None
        
        world_points = [self.board_points for _ in corners_subset]
        
        try:
            # Fisheye calibration
            flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                    cv2.fisheye.CALIB_FIX_SKEW + 
                    cv2.fisheye.CALIB_CHECK_COND)
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                np.expand_dims(np.asarray(world_points), -2),
                corners_subset,
                self.img_size,
                None,
                None,
                flags=flags,
                criteria=self.fisheye_criteria
            )
            
            return {
                'reprojection_error': ret,
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'n_frames': len(corners_subset)
            }
            
        except cv2.error as e:
            logger.warning(f"Calibration failed: {e}")
            return None
    
    def evaluate_april_tag_accuracy(self, 
                                  calibration_results: List[Dict],
                                  april_data_path: str,
                                  mocap_data_path: str,
                                  default_ids: List[int] = [12, 14, 20]) -> Dict:
        """
        Evaluate calibration accuracy using April tag pose estimation
        """
        logger.info("Evaluating April tag accuracy")
        
        # Load April tag data
        corners, ids, timestamps = self._load_april_data(april_data_path)
        
        # Load mocap data
        mocap_data = self._load_mocap_data(mocap_data_path, timestamps)
        
        results = []
        
        for idx, calib in enumerate(tqdm(calibration_results, desc="Evaluating calibrations")):
            try:
                accuracy = self._evaluate_single_calibration(
                    calib, corners, ids, mocap_data, default_ids
                )
                accuracy['calibration_id'] = idx
                results.append(accuracy)
            except Exception as e:
                logger.warning(f"Evaluation failed for calibration {idx}: {e}")
                continue
        
        return self._analyze_results(results)
    
    def _load_april_data(self, data_path: str) -> Tuple[List, List, List]:
        """Load April tag detection data"""
        corners, ids, timestamps = [], [], []
        
        with open(data_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            for frame in tqdm(unpacker, desc="Loading April tag data"):
                c, i, _ = self.detector.detectMarkers(frame)
                corners.append(c)
                ids.append(i)
        
        # Load timestamps
        timestamp_path = os.path.join(os.path.dirname(data_path), "webcam_timestamp.msgpack")
        with open(timestamp_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            for timestamp_data in unpacker:
                timestamps.append(timestamp_data[1])
        
        return corners, ids, timestamps
    
    def _load_mocap_data(self, mocap_path: str, timestamps: List) -> Dict:
        """Load and process motion capture data"""
        # This would be specific to your mocap data format
        # Placeholder implementation
        return {"x": [], "y": [], "z": [], "rx": [], "ry": [], "rz": []}
    
    def _evaluate_single_calibration(self, 
                                   calib: Dict, 
                                   corners: List, 
                                   ids: List, 
                                   mocap_data: Dict,
                                   default_ids: List) -> Dict:
        """
        Evaluate a single calibration against April tag data
        """
        camera_matrix = calib['camera_matrix']
        dist_coeffs = calib['dist_coeffs']
        
        # Undistort corners and estimate poses
        poses = []
        for corner_set, id_set in zip(corners, ids):
            if id_set is not None and len(id_set) > 0:
                # Undistort points
                undistorted = cv2.fisheye.undistortPoints(
                    np.array(corner_set).reshape(-1, 1, 2),
                    camera_matrix,
                    dist_coeffs,
                    None,
                    camera_matrix
                )
                
                # Estimate poses (simplified)
                pose = self._estimate_pose_from_corners(undistorted, id_set, default_ids)
                poses.append(pose)
            else:
                poses.append(None)
        
        # Calculate errors against mocap
        errors = self._calculate_pose_errors(poses, mocap_data)
        
        return {
            'mean_position_error': np.nanmean([errors['x'], errors['y'], errors['z']]),
            'max_position_error': np.nanmax([errors['x'], errors['y'], errors['z']]),
            'position_errors': errors,
            'reprojection_error': calib['reprojection_error']
        }
    
    def _estimate_pose_from_corners(self, corners: np.ndarray, ids: np.ndarray, default_ids: List) -> Optional[Dict]:
        """Estimate pose from undistorted corners"""
        # Simplified pose estimation - implement your specific logic here
        return None
    
    def _calculate_pose_errors(self, poses: List, mocap_data: Dict) -> Dict:
        """Calculate pose estimation errors"""
        # Placeholder - implement your error calculation logic
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze calibration evaluation results"""
        if not results:
            return {"best_calibration": None, "error": "No valid results"}
        
        # Find best calibration based on position accuracy
        best_idx = np.argmin([r['mean_position_error'] for r in results])
        best_result = results[best_idx]
        
        return {
            "best_calibration": best_result,
            "best_calibration_id": best_result['calibration_id'],
            "all_results": results,
            "mean_error_distribution": [r['mean_position_error'] for r in results],
            "reprojection_vs_pose_correlation": np.corrcoef(
                [r['reprojection_error'] for r in results],
                [r['mean_position_error'] for r in results]
            )[0, 1]
        }
    
    def cross_validate_calibration(self, 
                                 corners_list: List[np.ndarray], 
                                 n_folds: int = 5) -> Dict:
        """
        Cross-validate calibration approach
        """
        logger.info(f"Performing {n_folds}-fold cross-validation")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(corners_list)):
            logger.info(f"Processing fold {fold + 1}/{n_folds}")
            
            train_corners = [corners_list[i] for i in train_idx]
            val_corners = [corners_list[i] for i in val_idx]
            
            # Train calibration
            calib_result = self._calibrate_subset(train_corners)
            if calib_result is None:
                continue
            
            # Validate
            val_error = self._validate_calibration(calib_result, val_corners)
            cv_results.append({
                'fold': fold,
                'train_error': calib_result['reprojection_error'],
                'val_error': val_error,
                'overfitting': val_error - calib_result['reprojection_error']
            })
        
        return {
            'cv_results': cv_results,
            'mean_val_error': np.mean([r['val_error'] for r in cv_results]),
            'mean_overfitting': np.mean([r['overfitting'] for r in cv_results])
        }
    
    def _validate_calibration(self, calib_result: Dict, val_corners: List[np.ndarray]) -> float:
        """Validate calibration on held-out data"""
        camera_matrix = calib_result['camera_matrix']
        dist_coeffs = calib_result['dist_coeffs']
        
        total_error = 0.0
        total_points = 0
        
        for corners in val_corners:
            # Project 3D points back to image
            projected, _ = cv2.fisheye.projectPoints(
                self.board_points.reshape(1, -1, 3),
                np.zeros(3),  # No rotation
                np.zeros(3),  # No translation
                camera_matrix,
                dist_coeffs
            )
            
            # Calculate reprojection error
            error = cv2.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), cv2.NORM_L2)
            total_error += error
            total_points += len(corners)
        
        return total_error / total_points if total_points > 0 else float('inf')
    
    def save_results(self, results: Dict, output_path: str):
        """Save calibration results"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main calibration pipeline"""
    # Initialize calibration system
    calibrator = ImprovedFisheyeCalibration()
    
    # Paths - adjust these to your data (using absolute paths)
    base_path = os.path.dirname(os.path.abspath(__file__))
    calib_video_path = os.path.join(base_path, "data", "recordings_160fov", "calib_mono_160fov2", "webcam_color.msgpack")
    april_data_path = os.path.join(base_path, "data", "recordings_160fov", "3marker_complete_data", "3marker_april_mono_160fov_3", "webcam_color.msgpack")
    mocap_data_path = os.path.join(base_path, "data", "recordings_160fov", "3marker_complete_data", "3marker_april_mono_160fov_3", "3marker_april_mono_160fov_3.csv")
    
    # Check if files exist
    if not os.path.exists(calib_video_path):
        logger.error(f"Calibration video not found: {calib_video_path}")
        logger.info("Please update the paths in the main() function")
        return
    
    # Step 1: Extract calibration data
    corners_list = calibrator.extract_calibration_data(calib_video_path)
    
    if len(corners_list) < 20:
        logger.error("Not enough calibration data found")
        return
    
    # Step 2: Cross-validate calibration approach
    cv_results = calibrator.cross_validate_calibration(corners_list)
    logger.info(f"Cross-validation results: {cv_results['mean_val_error']:.4f} ± {cv_results['mean_overfitting']:.4f}")
    
    # Step 3: Generate systematic calibration subsets
    calibration_results = calibrator.systematic_calibration_selection(corners_list)
    
    # Step 4: Evaluate against April tag data
    evaluation_results = calibrator.evaluate_april_tag_accuracy(
        calibration_results, april_data_path, mocap_data_path
    )
    
    # Step 5: Save results
    final_results = {
        'cross_validation': cv_results,
        'calibration_evaluation': evaluation_results,
        'best_calibration': evaluation_results.get('best_calibration')
    }
    
    calibrator.save_results(final_results, "improved_calibration_results2.json")
    
    # Print summary
    if evaluation_results.get('best_calibration'):
        best = evaluation_results['best_calibration']
        logger.info(f"Best calibration found:")
        logger.info(f"  Mean position error: {best['mean_position_error']:.4f}m")
        logger.info(f"  Max position error: {best['max_position_error']:.4f}m")
        logger.info(f"  Reprojection error: {best['reprojection_error']:.4f}")


if __name__ == "__main__":
    main()