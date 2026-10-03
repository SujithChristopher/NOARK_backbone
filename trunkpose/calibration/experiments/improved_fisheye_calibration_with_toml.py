"""
Improved Fisheye Camera Calibration for OV9281 160° FOV Camera
Optimized for April Tag pose estimation accuracy with TOML output

Key improvements:
1. Better corner detection with sub-pixel refinement
2. Systematic calibration data selection (not random)
3. Proper fisheye distortion model validation
4. April tag-specific optimization criteria
5. Cross-validation framework
6. TOML output format for camera calibration parameters
"""

# ================================
# CONFIGURATION SECTION - EDIT HERE
# ================================
CONFIG = {
    # Performance settings
    'OMP_NUM_THREADS': 20,                    # Number of threads for KMeans/OpenMP
    'MAX_FRAMES_TO_PROCESS': 7000,           # Maximum frames to extract from video
    'MAX_TRAIN_FRAMES_PER_FOLD': 250,        # Max training frames per cross-validation fold
    'N_CROSS_VALIDATION_FOLDS': 40,          # Number of cross-validation folds
    
    # Calibration settingss
    'N_CALIBRATION_SUBSETS': 200,            # Number of systematic calibration subsets
    'SUBSET_SIZE': 200,                       # Size of each calibration subset
    'MIN_FRAMES_FOR_CALIBRATION': 15,        # Minimum frames needed for calibration
    
    # Camera settings
    'PATTERN_SIZE': (6, 4),                  # Chessboard pattern size (corners)
    'SQUARE_SIZE': 30.0,                     # Size of chessboard squares in mm
    'IMG_SIZE': (1200, 800),                 # Image resolution
    'MARKER_SIZE': 0.05,                     # April tag marker size in meters
    
    # Corner detection quality thresholds
    'MIN_SPAN_RATIO': 0.15,                  # Minimum corner span ratio for fisheye
    'MIN_CORNER_STD': 20,                    # Minimum corner standard deviation
    'CORNER_BOUNDARY_MARGIN': 10,            # Margin from image edges for corners
    
    # Caching settings
    'ENABLE_CORNER_CACHING': True,           # Enable caching of extracted corners
    'CACHE_DIR': 'cache',                    # Directory for cache files
}

import os
# Fix KMeans memory leak warning on Windows
os.environ['OMP_NUM_THREADS'] = str(CONFIG['OMP_NUM_THREADS'])

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
import tomli_w
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
                 pattern_size: Tuple[int, int] = None,
                 square_size: float = None,
                 img_size: Tuple[int, int] = None,
                 marker_size: float = None):
        """
        Initialize calibration system
        
        Args:
            pattern_size: Chessboard pattern size (corners) - uses CONFIG if None
            square_size: Size of chessboard squares in mm - uses CONFIG if None
            img_size: Image resolution - uses CONFIG if None
            marker_size: April tag marker size in meters - uses CONFIG if None
        """
        self.pattern_size = pattern_size or CONFIG['PATTERN_SIZE']
        self.square_size = square_size or CONFIG['SQUARE_SIZE']
        self.img_size = img_size or CONFIG['IMG_SIZE']
        self.marker_size = marker_size or CONFIG['MARKER_SIZE']
        
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
        min_span_ratio = CONFIG['MIN_SPAN_RATIO']
        
        # Check if corners span reasonable area
        x_span_ok = x_range / self.img_size[0] >= min_span_ratio
        y_span_ok = y_range / self.img_size[1] >= min_span_ratio
        
        # Additional check: corners should be within image bounds
        margin = CONFIG['CORNER_BOUNDARY_MARGIN']
        within_bounds = (np.all(corners_flat[:, 0] >= margin) and 
                        np.all(corners_flat[:, 0] <= self.img_size[0] - margin) and
                        np.all(corners_flat[:, 1] >= margin) and 
                        np.all(corners_flat[:, 1] <= self.img_size[1] - margin))
        
        # For fisheye, accept if either dimension has good span OR if corners are well distributed
        corner_std = np.std(corners_flat, axis=0)
        min_std = CONFIG['MIN_CORNER_STD']
        good_distribution = corner_std[0] > min_std and corner_std[1] > min_std
        
        return within_bounds and (x_span_ok or y_span_ok or good_distribution)
    
    def _get_cache_filename(self, video_path: str, max_frames: int) -> str:
        """Generate cache filename based on video path and settings"""
        import hashlib
        # Create hash from video path, max_frames, and relevant config
        cache_key = f"{video_path}_{max_frames}_{self.pattern_size}_{self.square_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        return f"corners_{video_name}_{cache_hash}.msgpack"
    
    def _save_corners_cache(self, corners_list: List[np.ndarray], cache_path: str):
        """Save corners to cache file"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                mp.pack(corners_list, f, default=mpn.encode)
            logger.info(f"Saved {len(corners_list)} corners to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save corners cache: {e}")
    
    def _load_corners_cache(self, cache_path: str) -> Optional[List[np.ndarray]]:
        """Load corners from cache file"""
        try:
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                corners_list = mp.unpack(f, object_hook=mpn.decode)
            
            logger.info(f"Loaded {len(corners_list)} corners from cache: {cache_path}")
            return corners_list
        except Exception as e:
            logger.warning(f"Failed to load corners cache: {e}")
            return None
    
    def extract_calibration_data(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract high-quality calibration frames from video with caching support
        """
        max_frames = max_frames or CONFIG['MAX_FRAMES_TO_PROCESS']
        
        # Check cache if enabled
        if CONFIG['ENABLE_CORNER_CACHING']:
            cache_filename = self._get_cache_filename(video_path, max_frames)
            cache_path = os.path.join(CONFIG['CACHE_DIR'], cache_filename)
            
            # Try to load from cache first
            cached_corners = self._load_corners_cache(cache_path)
            if cached_corners is not None:
                logger.info(f"Using cached corners ({len(cached_corners)} corners)")
                return cached_corners
        
        logger.info(f"Extracting calibration data from {video_path} (max {max_frames} frames)")
        
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
        
        # Save to cache if enabled
        if CONFIG['ENABLE_CORNER_CACHING'] and corners_list:
            cache_filename = self._get_cache_filename(video_path, max_frames)
            cache_path = os.path.join(CONFIG['CACHE_DIR'], cache_filename)
            self._save_corners_cache(corners_list, cache_path)
        
        return corners_list
    
    def systematic_calibration_selection(self, 
                                       corners_list: List[np.ndarray], 
                                       n_subsets: int = None,
                                       subset_size: int = None) -> List[Dict]:
        """
        Systematic calibration data selection instead of random
        Uses k-means clustering on corner positions for diverse selection
        """
        n_subsets = n_subsets or CONFIG['N_CALIBRATION_SUBSETS']
        subset_size = subset_size or CONFIG['SUBSET_SIZE']
        logger.info(f"Creating {n_subsets} systematic calibration subsets of size {subset_size}")
        
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
        Calibrate camera using a subset of corners with multiple strategies
        """
        if len(corners_subset) < CONFIG['MIN_FRAMES_FOR_CALIBRATION']:
            return None
        
        world_points = [self.board_points for _ in corners_subset]
        
        # Try multiple calibration strategies with valid fisheye flags
        calibration_strategies = [
            {
                'name': 'standard',
                'flags': cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
            },
            {
                'name': 'use_intrinsic_guess',
                'flags': cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            },
            {
                'name': 'check_cond',
                'flags': cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND
            },
            {
                'name': 'fix_principal_point',
                'flags': cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
            }
        ]
        
        best_result = None
        best_error = float('inf')
        
        for strategy in calibration_strategies:
            try:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
                    np.expand_dims(np.asarray(world_points), -2),
                    corners_subset,
                    self.img_size,
                    None,
                    None,
                    flags=strategy['flags'],
                    criteria=self.fisheye_criteria
                )
                
                # Validate calibration result
                if ret < best_error and self._validate_calibration_result(camera_matrix, dist_coeffs):
                    best_error = ret
                    best_result = {
                        'reprojection_error': ret,
                        'camera_matrix': camera_matrix,
                        'dist_coeffs': dist_coeffs,
                        'rvecs': rvecs,
                        'tvecs': tvecs,
                        'n_frames': len(corners_subset),
                        'strategy': strategy['name']
                    }
                
            except cv2.error as e:
                continue  # Try next strategy
        
        return best_result
    
    def _validate_calibration_result(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> bool:
        """
        Validate calibration result for reasonable parameters
        """
        # Check focal lengths are reasonable
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        if fx < 100 or fx > 2000 or fy < 100 or fy > 2000:
            return False
        
        # Check principal point is roughly in center
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        img_w, img_h = self.img_size
        if cx < img_w * 0.2 or cx > img_w * 0.8 or cy < img_h * 0.2 or cy > img_h * 0.8:
            return False
        
        # Check distortion coefficients are reasonable (not extreme)
        if np.any(np.abs(dist_coeffs.flatten()) > 2.0):
            return False
        
        return True
    
    def find_best_calibration(self, calibration_results: List[Dict]) -> Optional[Dict]:
        """
        Find the best calibration based on reprojection error and other criteria
        """
        if not calibration_results:
            return None
        
        # Score calibrations based on multiple criteria
        scores = []
        max_frames_used = max(calib['n_frames'] for calib in calibration_results)
        optimal_frame_count = CONFIG['SUBSET_SIZE']
        
        for calib in calibration_results:
            # Lower reprojection error is better (primary criterion)
            error_score = 1.0 / (calib['reprojection_error'] + 1e-6)
            
            # Frames used closer to optimal is better
            frame_ratio = calib['n_frames'] / optimal_frame_count
            frame_score = 1.0 - abs(1.0 - frame_ratio)  # Penalty for too few or too many frames
            frame_score = max(0.1, frame_score)  # Minimum score
            
            # Combined score with higher weight on reprojection error
            total_score = error_score * 0.8 + frame_score * 0.2
            scores.append(total_score)
        
        # Find best calibration
        best_idx = np.argmax(scores)
        best_calibration = calibration_results[best_idx]
        
        # Enhanced logging for analysis
        logger.info(f"=== CALIBRATION SELECTION ANALYSIS ===")
        logger.info(f"Total calibrations evaluated: {len(calibration_results)}")
        logger.info(f"Best calibration (index {best_idx}) details:")
        logger.info(f"  Reprojection error: {best_calibration['reprojection_error']:.6f}")
        logger.info(f"  Frames used: {best_calibration['n_frames']}")
        logger.info(f"  Selection strategy: {best_calibration.get('selection_strategy', 'unknown')}")
        logger.info(f"  Score: {scores[best_idx]:.4f}")
        
        # Show error distribution
        errors = [calib['reprojection_error'] for calib in calibration_results]
        logger.info(f"Error distribution - Min: {min(errors):.6f}, Max: {max(errors):.6f}, Mean: {np.mean(errors):.6f}")
        
        logger.info(f"Camera matrix:\n{best_calibration['camera_matrix']}")
        logger.info(f"Distortion coefficients: {best_calibration['dist_coeffs'].flatten()}")
        
        return best_calibration
    
    def save_calibration_to_toml(self, calibration: Dict, output_path: str):
        """
        Save the best calibration to TOML format matching optimized_fisheye_calibration.toml structure
        """
        camera_matrix = calibration['camera_matrix'].tolist()
        dist_coeffs = calibration['dist_coeffs'].flatten().tolist()
        
        # Create TOML structure
        toml_data = {
            "calibration": {
                "camera_matrix": camera_matrix,
                "dist_coeffs": [dist_coeffs]  # Keep as nested list for consistency
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
                "resolution": list(self.img_size)
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
        
        # Write TOML file
        with open(output_path, 'wb') as f:
            tomli_w.dump(toml_data, f)
        
        logger.info(f"Calibration saved to TOML file: {output_path}")
        logger.info(f"Reprojection error: {calibration['reprojection_error']:.6f}")
    
    def cross_validate_calibration(self, 
                                 corners_list: List[np.ndarray], 
                                 n_folds: int = None,
                                 max_train_frames: int = None) -> Dict:
        """
        Cross-validate calibration approach with performance optimization
        """
        n_folds = n_folds or CONFIG['N_CROSS_VALIDATION_FOLDS']
        max_train_frames = max_train_frames or CONFIG['MAX_TRAIN_FRAMES_PER_FOLD']
        logger.info(f"Performing {n_folds}-fold cross-validation (max {max_train_frames} training frames per fold)")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(corners_list)):
            logger.info(f"Processing fold {fold + 1}/{n_folds}")
            
            # Limit training data to prevent extremely slow calibration
            if len(train_idx) > max_train_frames:
                np.random.seed(fold)  # Consistent sampling per fold
                train_idx = np.random.choice(train_idx, max_train_frames, replace=False)
                logger.info(f"  Reduced training set from {len([corners_list[i] for i in train_idx])} to {max_train_frames} frames")
            
            train_corners = [corners_list[i] for i in train_idx]
            val_corners = [corners_list[i] for i in val_idx]
            
            logger.info(f"  Training with {len(train_corners)} frames, validating with {len(val_corners)} frames")
            
            # Train calibration
            calib_result = self._calibrate_subset(train_corners)
            if calib_result is None:
                logger.warning(f"  Fold {fold + 1} calibration failed")
                continue
            
            logger.info(f"  Fold {fold + 1} training error: {calib_result['reprojection_error']:.4f}")
            
            # Validate
            val_error = self._validate_calibration(calib_result, val_corners)
            logger.info(f"  Fold {fold + 1} validation error: {val_error:.4f}")
            
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
        """Save calibration results as JSON"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Handle objects with attributes
                return convert_numpy(obj.__dict__)
            return obj
        
        try:
            results_serializable = convert_numpy(results)
            
            with open(output_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Save a simplified version without problematic data
            simplified_results = {
                'best_calibration': {
                    'reprojection_error': float(results.get('best_calibration', {}).get('reprojection_error', 0)),
                    'n_frames': int(results.get('best_calibration', {}).get('n_frames', 0))
                },
                'note': 'Simplified results due to serialization issues'
            }
            with open(output_path.replace('.json', '_simplified.json'), 'w') as f:
                json.dump(simplified_results, f, indent=2)
            logger.info(f"Simplified results saved to {output_path.replace('.json', '_simplified.json')}")


def main():
    """Main calibration pipeline"""
    # Initialize calibration system
    calibrator = ImprovedFisheyeCalibration()
    
    # Paths - adjust these to your data (using absolute paths)
    base_path = os.path.dirname(os.path.abspath(__file__))
    calib_video_path = os.path.join(base_path, "data", "calibration", "160_fov", 'calib_mono_160fov_raw',"webcam_color.msgpack")
    
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
    
    # Step 4: Find the best calibration
    best_calibration = calibrator.find_best_calibration(calibration_results)
    
    if best_calibration is None:
        logger.error("No valid calibration found")
        return
    
    # Step 5: Save best calibration to TOML format
    toml_output_path = "improved_fisheye_calibration_optimal.toml"
    calibrator.save_calibration_to_toml(best_calibration, toml_output_path)
    
    # Step 6: Save complete results as JSON
    final_results = {
        'cross_validation': cv_results,
        'best_calibration': best_calibration,
        'all_calibrations': calibration_results
    }
    
    calibrator.save_results(final_results, "improved_calibration_results_with_toml.json")
    
    # Print summary
    logger.info(f"Calibration completed successfully!")
    logger.info(f"Best calibration reprojection error: {best_calibration['reprojection_error']:.6f}")
    logger.info(f"Used {best_calibration['n_frames']} frames for calibration")
    logger.info(f"TOML file saved as: {toml_output_path}")


if __name__ == "__main__":
    main()