"""
Enhanced Fisheye Calibration with Advanced Techniques
Includes multiple sophisticated approaches that can be enabled/disabled
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
from scipy.optimize import least_squares, dual_annealing
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import sys
import logging
import toml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for advanced calibration features"""
    # Basic settings
    pattern_size: Tuple[int, int] = (6, 4)
    square_size: float = 30
    img_size: Tuple[int, int] = (1200, 800)
    
    # Advanced features flags
    use_adaptive_selection: bool = False
    use_multi_scale: bool = False
    use_ransac: bool = False
    use_bundle_adjustment: bool = False
    use_cross_validation: bool = False
    use_temperature_selection: bool = False
    cache_undistortion_maps: bool = False
    use_adaptive_aruco: bool = False
    use_fisheye_refinement: bool = False
    
    # Advanced parameters
    ransac_iterations: int = 100
    ransac_threshold: float = 2.0
    cv_folds: int = 5
    multi_scale_weights: List[float] = None
    
    def __post_init__(self):
        if self.multi_scale_weights is None:
            self.multi_scale_weights = [1.0, 1.5, 2.0]  # center, mid, periphery


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
            "tvec": t,
            "metadata": {
                'bundle_adjusted': False,
                'ransac': False,
                'cv_fold': -1,
                'weights': []
            }
        }
    except Exception as e:
        return None


def calibrate_with_advanced_features_standalone(chessb_corners, board_points, img_size, config, indices=None):
    """Standalone calibration with advanced features for sequential processing"""
    if indices is None:
        if config.use_adaptive_selection:
            # Simple diversity selection without sklearn
            n_samples = min(20, len(chessb_corners))
            # Spread samples evenly across the dataset
            step = len(chessb_corners) // n_samples
            indices = np.arange(0, len(chessb_corners), step)[:n_samples]
        else:
            n_samples = min(20, len(chessb_corners))
            indices = np.random.choice(len(chessb_corners), n_samples, replace=False)
    
    chessb_c = chessb_corners[indices]
    
    # Multi-scale weighting
    weights = np.ones(len(chessb_c))
    if config.use_multi_scale:
        h, w = img_size[1], img_size[0]
        for i, corners in enumerate(chessb_c):
            centroid = corners.mean(axis=0)
            dist_from_center = np.linalg.norm(centroid - [w/2, h/2])
            
            if dist_from_center < min(h, w) * 0.3:
                weights[i] = config.multi_scale_weights[0]
            elif dist_from_center < min(h, w) * 0.6:
                weights[i] = config.multi_scale_weights[1]
            else:
                weights[i] = config.multi_scale_weights[2]
    
    world_points = []
    image_points = []
    
    for _f in chessb_c:
        image_points.append(_f)
        world_points.append(board_points)
    
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
        
        result = {
            "ReError": ret,
            "mat": camera_matrix,
            "dist": k,
            "rvec": R,
            "tvec": t,
            "weights": weights.tolist(),
            "metadata": {
                'bundle_adjusted': False,
                'ransac': False,
                'cv_fold': -1,
                'weights': weights.tolist()
            }
        }
        
        return result
        
    except Exception as e:
        return None


class EnhancedFisheyeCalibration:
    """Enhanced fisheye calibration with advanced techniques"""
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        
        # Basic setup
        self.pattern_size = self.config.pattern_size
        self.square_size = self.config.square_size
        self.img_size = self.config.img_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Construct 3D points
        self.board_points = self._construct_3d_points()
        
        # ArUco setup
        self._setup_aruco()
        
        # Cache for undistortion maps
        self.undistortion_maps = {}
        
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
    
    def select_diverse_samples(self, chessb_corners, n_samples=20):
        """Intelligently select calibration images based on spatial diversity"""
        if not self.config.use_adaptive_selection:
            return np.random.choice(len(chessb_corners), min(n_samples, len(chessb_corners)), replace=False)
        
        logger.info("Using adaptive sample selection")
        
        # Compute centroid of each corner set
        centroids = np.array([corners.mean(axis=0).flatten() for corners in chessb_corners])
        
        # Use k-means clustering to ensure spatial diversity
        n_clusters = min(n_samples, len(chessb_corners))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(centroids)
        
        # Select samples from each cluster
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > 0:
                # Could add additional criteria here (e.g., corner sharpness)
                selected_indices.append(cluster_indices[0])
        
        return np.array(selected_indices)
    
    def classify_corners_by_region(self, chessb_corners):
        """Classify corners into regions for multi-scale calibration"""
        h, w = self.img_size[1], self.img_size[0]
        center_radius = min(h, w) * 0.3
        mid_radius = min(h, w) * 0.6
        
        regions = {'center': [], 'mid': [], 'periphery': []}
        indices = {'center': [], 'mid': [], 'periphery': []}
        
        for idx, corners in enumerate(chessb_corners):
            centroid = corners.mean(axis=0)
            dist_from_center = np.linalg.norm(centroid - [w/2, h/2])
            
            if dist_from_center < center_radius:
                regions['center'].append(corners)
                indices['center'].append(idx)
            elif dist_from_center < mid_radius:
                regions['mid'].append(corners)
                indices['mid'].append(idx)
            else:
                regions['periphery'].append(corners)
                indices['periphery'].append(idx)
        
        return regions, indices
    
    def robust_calibration_ransac(self, chessb_corners):
        """RANSAC-based robust calibration"""
        if not self.config.use_ransac:
            return None
        
        logger.info(f"Running RANSAC calibration with {self.config.ransac_iterations} iterations")
        
        best_inliers = 0
        best_params = None
        threshold = self.config.ransac_threshold
        
        for iteration in tqdm(range(self.config.ransac_iterations), desc="RANSAC iterations"):
            # Random minimal sample
            sample_size = max(10, len(chessb_corners) // 10)
            indices = np.random.choice(len(chessb_corners), min(sample_size, len(chessb_corners)), replace=False)
            
            # Calibrate with subset
            result = self.calibrate_single_iteration(chessb_corners[indices])
            
            if result is None:
                continue
            
            # Test on all data
            inliers = 0
            total_error = 0
            
            for corners in chessb_corners:
                try:
                    # Project points and compute reprojection error
                    rvec = np.zeros((3, 1))
                    tvec = np.zeros((3, 1))
                    projected, _ = cv2.fisheye.projectPoints(
                        self.board_points.reshape(-1, 1, 3),
                        rvec, tvec,
                        result['mat'], result['dist']
                    )
                    error = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean()
                    
                    if error < threshold:
                        inliers += 1
                        total_error += error
                except:
                    continue
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_params = result
                best_params['inliers'] = inliers
                best_params['avg_inlier_error'] = total_error / max(inliers, 1)
                logger.info(f"RANSAC iteration {iteration}: {inliers} inliers, avg error: {best_params['avg_inlier_error']:.3f}")
        
        return best_params
    
    def refine_with_bundle_adjustment(self, initial_params, chessb_corners):
        """Refine calibration using bundle adjustment"""
        if not self.config.use_bundle_adjustment:
            return initial_params
        
        logger.info("Refining with bundle adjustment")
        
        def residuals(params, corners, points):
            # Unpack parameters
            k1, k2, k3, k4 = params[:4]
            fx, fy = params[4:6]
            cx, cy = params[6:8]
            
            # Build camera matrix and distortion coefficients
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            D = np.array([k1, k2, k3, k4])
            
            # Compute reprojection errors for all points
            errors = []
            for corner_set in corners:
                try:
                    rvec = np.zeros((3, 1))
                    tvec = np.zeros((3, 1))
                    projected, _ = cv2.fisheye.projectPoints(
                        points.reshape(-1, 1, 3),
                        rvec, tvec,
                        K, D
                    )
                    error = (corner_set.reshape(-1, 2) - projected.reshape(-1, 2)).flatten()
                    errors.extend(error)
                except:
                    # If projection fails, add large error
                    errors.extend(np.ones(len(points) * 2) * 1000)
            
            return np.array(errors)
        
        # Initial guess from calibration
        x0 = np.concatenate([
            initial_params['dist'].flatten(),
            [initial_params['mat'][0,0], initial_params['mat'][1,1]],
            [initial_params['mat'][0,2], initial_params['mat'][1,2]]
        ])
        
        # Bounds for parameters
        bounds = (
            [-10, -10, -10, -10, 100, 100, 0, 0],  # Lower bounds
            [10, 10, 10, 10, 2000, 2000, self.img_size[0], self.img_size[1]]  # Upper bounds
        )
        
        try:
            result = least_squares(
                residuals, x0,
                args=(chessb_corners, self.board_points),
                bounds=bounds,
                method='trf',
                verbose=0,
                max_nfev=1000
            )
            
            # Reconstruct parameters
            refined_params = initial_params.copy()
            refined_params['dist'] = result.x[:4]
            refined_params['mat'] = np.array([
                [result.x[4], 0, result.x[6]],
                [0, result.x[5], result.x[7]],
                [0, 0, 1]
            ])
            refined_params['bundle_adjusted'] = True
            
            logger.info(f"Bundle adjustment converged: {result.success}")
            return refined_params
        except Exception as e:
            logger.warning(f"Bundle adjustment failed: {e}")
            return initial_params
    
    def cross_validate_calibration(self, chessb_corners):
        """Perform cross-validation to evaluate calibration stability"""
        if not self.config.use_cross_validation:
            return None
        
        logger.info(f"Running {self.config.cv_folds}-fold cross-validation")
        
        kf = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(chessb_corners)):
            logger.info(f"Cross-validation fold {fold+1}/{self.config.cv_folds}")
            
            # Train on subset
            train_corners = chessb_corners[train_idx]
            val_corners = chessb_corners[val_idx]
            
            # Calibrate
            result = self.calibrate_single_iteration(train_corners)
            
            if result is None:
                continue
            
            # Validate
            val_errors = []
            for corners in val_corners:
                try:
                    rvec = np.zeros((3, 1))
                    tvec = np.zeros((3, 1))
                    projected, _ = cv2.fisheye.projectPoints(
                        self.board_points.reshape(-1, 1, 3),
                        rvec, tvec,
                        result['mat'], result['dist']
                    )
                    error = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean()
                    val_errors.append(error)
                except:
                    continue
            
            if val_errors:
                result['val_error'] = np.mean(val_errors)
                result['fold'] = fold
                cv_results.append(result)
        
        return cv_results
    
    def create_undistortion_maps(self, camera_matrix, dist_coeffs):
        """Create and cache undistortion maps"""
        if not self.config.cache_undistortion_maps:
            return None, None
        
        key = (camera_matrix.tobytes(), dist_coeffs.tobytes())
        if key in self.undistortion_maps:
            return self.undistortion_maps[key]
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3),
            camera_matrix, self.img_size, cv2.CV_16SC2
        )
        
        self.undistortion_maps[key] = (map1, map2)
        return map1, map2
    
    def adaptive_aruco_detection(self, image, camera_matrix, dist_coeffs):
        """Adaptive ArUco detection with region-specific parameters"""
        if not self.config.use_adaptive_aruco:
            return self.detector.detectMarkers(image)
        
        h, w = image.shape[:2]
        
        # Create masks for different regions
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w//2, h//2), int(min(h, w) * 0.3), 255, -1)
        
        periph_mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(periph_mask, (w//2, h//2), int(min(h, w) * 0.6), 0, -1)
        
        # Different parameters for different regions
        center_params = cv2.aruco.DetectorParameters()
        center_params.minMarkerPerimeterRate = 0.03
        center_params.adaptiveThreshWinSizeMax = 23
        
        periph_params = cv2.aruco.DetectorParameters()
        periph_params.minMarkerPerimeterRate = 0.02
        periph_params.adaptiveThreshWinSizeMax = 33
        periph_params.minCornerDistanceRate = 0.05
        periph_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Detect in different regions
        center_detector = cv2.aruco.ArucoDetector(self.aruco_dict, center_params)
        periph_detector = cv2.aruco.ArucoDetector(self.aruco_dict, periph_params)
        
        # Mask images
        center_img = cv2.bitwise_and(image, image, mask=center_mask)
        periph_img = cv2.bitwise_and(image, image, mask=periph_mask)
        
        # Detect
        corners_c, ids_c, _ = center_detector.detectMarkers(center_img)
        corners_p, ids_p, _ = periph_detector.detectMarkers(periph_img)
        
        # Combine results
        all_corners = []
        all_ids = []
        
        if len(corners_c) > 0:
            all_corners.extend(corners_c)
            all_ids.extend(ids_c)
        
        if len(corners_p) > 0:
            all_corners.extend(corners_p)
            all_ids.extend(ids_p)
        
        return all_corners, np.array(all_ids) if all_ids else None, None
    
    def extract_chessboard_corners(self, video_path, n_jobs=20):
        """Extract chessboard corners using parallel processing"""
        logger.info(f"Extracting corners from {video_path}")
        
        with open(video_path, "rb") as f:
            video_data = list(mp.Unpacker(f, object_hook=mpn.decode))
        
        logger.info(f"Loaded {len(video_data)} frames")
        
        # Parallel corner detection
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(detect_corners_standalone)(frame, self.pattern_size) for frame in tqdm(video_data)
        )
        
        # Filter out None results
        chessb_corners = [corner for corner in results if corner is not None]
        
        logger.info(f"Found {len(chessb_corners)} valid corner sets")
        return np.array(chessb_corners)
    
    def calibrate_single_iteration(self, chessb_corners, n_samples=20):
        """Single calibration iteration with optional advanced features"""
        # Use standalone function for advanced features
        result = calibrate_with_advanced_features_standalone(
            chessb_corners, self.board_points, self.img_size, self.config
        )
        
        if result and self.config.use_bundle_adjustment:
            result = self.refine_with_bundle_adjustment(result, chessb_corners)
        
        return result
    
    def generate_calibrations(self, chessb_corners, n_calibrations=200, n_jobs=20):
        """Generate multiple calibrations with advanced options"""
        logger.info(f"Generating {n_calibrations} calibrations")
        logger.info(f"Advanced features enabled: {[k for k, v in vars(self.config).items() if v and k.startswith('use_')]}")
        
        # RANSAC calibration
        ransac_result = None
        if self.config.use_ransac:
            ransac_result = self.robust_calibration_ransac(chessb_corners)
            if ransac_result:
                logger.info(f"RANSAC calibration: {ransac_result['inliers']} inliers, error: {ransac_result['avg_inlier_error']:.3f}")
        
        # Cross-validation
        cv_results = None
        if self.config.use_cross_validation:
            cv_results = self.cross_validate_calibration(chessb_corners)
            if cv_results:
                cv_errors = [r['val_error'] for r in cv_results]
                logger.info(f"Cross-validation errors: mean={np.mean(cv_errors):.3f}, std={np.std(cv_errors):.3f}")
        
        # Regular calibrations - use standalone function for parallel processing
        if self.config.use_adaptive_selection or self.config.use_multi_scale or self.config.use_bundle_adjustment:
            # If using advanced features, run sequentially
            logger.info("Running sequential calibration (advanced features enabled)")
            results = []
            for _ in tqdm(range(n_calibrations), colour="green"):
                result = self.calibrate_single_iteration(chessb_corners, n_samples=20)
                results.append(result)
        else:
            # Use parallel processing with standalone function
            results = Parallel(n_jobs=n_jobs)(
                delayed(calibrate_single_iteration_standalone)(chessb_corners, self.board_points, self.img_size) 
                for _ in tqdm(range(n_calibrations), colour="green")
            )
        
        # Add special calibrations
        if ransac_result:
            results.append(ransac_result)
        if cv_results:
            results.extend(cv_results)
        
        # Filter out failed calibrations
        valid_results = [r for r in results if r is not None]
        
        # Organize results
        my_dict = {"ReError": [], "mat": [], "dist": [], "rvec": [], "tvec": [], "metadata": []}
        for result in valid_results:
            my_dict["ReError"].append(result["ReError"])
            my_dict["mat"].append(result["mat"])
            my_dict["dist"].append(result["dist"])
            my_dict["rvec"].append(result["rvec"])
            my_dict["tvec"].append(result["tvec"])
            
            # Store metadata
            metadata = {
                'bundle_adjusted': result.get('bundle_adjusted', False),
                'ransac': 'inliers' in result,
                'cv_fold': result.get('fold', -1),
                'weights': result.get('weights', [])
            }
            my_dict["metadata"].append(metadata)
        
        logger.info(f"Generated {len(valid_results)} valid calibrations")
        return my_dict
    
    def temperature_aware_selection(self, calibrations, evaluation_metrics):
        """Use simulated annealing for calibration selection"""
        if not self.config.use_temperature_selection:
            return None
        
        logger.info("Using temperature-aware calibration selection")
        
        def objective(weights):
            # Weighted combination of metrics
            score = 0
            for i, w in enumerate(weights):
                if i < len(evaluation_metrics):
                    # Combine reprojection error and success rate
                    metric = evaluation_metrics[i]
                    score += w * (metric['success_rate'] - 0.1 * metric['reprojection_error'])
            return -score  # Minimize negative score
        
        bounds = [(0, 1) for _ in range(len(calibrations))]
        
        try:
            result = dual_annealing(objective, bounds, maxiter=100)
            best_idx = np.argmax(result.x)
            logger.info(f"Temperature selection chose calibration {best_idx}")
            return best_idx
        except:
            return None
    
    def evaluate_calibration(self, my_dict, corners, ids, default_ids=[12, 14, 20]):
        """Evaluate calibrations with enhanced metrics"""
        logger.info("Evaluating calibrations")
        
        def process_calibration(i):
            try:
                _fish_mat = my_dict["mat"][i]
                _fish_dist = my_dict["dist"][i]
                
                _new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    _fish_mat, _fish_dist, self.img_size, np.eye(3), balance=1
                )
                
                # Create undistortion maps if caching enabled
                if self.config.cache_undistortion_maps:
                    self.create_undistortion_maps(_fish_mat, _fish_dist)
                
                # Enhanced evaluation metrics
                successful_poses = 0
                total_attempts = 0
                reprojection_errors = []
                region_success = {'center': 0, 'mid': 0, 'periphery': 0}
                region_attempts = {'center': 0, 'mid': 0, 'periphery': 0}
                
                for _corner, _id in zip(corners[:100], ids[:100]):
                    if _corner is not None and len(_corner) > 0:
                        total_attempts += 1
                        
                        try:
                            # Determine region
                            centroid = np.mean(_corner[0], axis=0)
                            h, w = self.img_size[1], self.img_size[0]
                            dist_from_center = np.linalg.norm(centroid - [w/2, h/2])
                            
                            if dist_from_center < min(h, w) * 0.3:
                                region = 'center'
                            elif dist_from_center < min(h, w) * 0.6:
                                region = 'mid'
                            else:
                                region = 'periphery'
                            
                            region_attempts[region] += 1
                            
                            # Undistort corners
                            undist_corners = cv2.fisheye.undistortPoints(
                                np.array(_corner).reshape(-1, 1, 2), 
                                _fish_mat, _fish_dist, None, _new_cam
                            )
                            
                            if undist_corners is not None and len(undist_corners) > 0:
                                successful_poses += 1
                                region_success[region] += 1
                                
                                # Estimate pose and compute reprojection error
                                rvec, tvec = self.estimate_pose_single_markers(
                                    _corner, self.marker_length, _new_cam, np.zeros(4)
                                )
                                
                                if len(rvec) > 0:
                                    # Project back and compute error
                                    projected, _ = cv2.projectPoints(
                                        self._get_marker_3d_points(),
                                        rvec[0], tvec[0],
                                        _new_cam, np.zeros(4)
                                    )
                                    error = np.linalg.norm(
                                        _corner[0].reshape(-1, 2) - projected.reshape(-1, 2),
                                        axis=1
                                    ).mean()
                                    reprojection_errors.append(error)
                        except:
                            continue
                
                success_rate = successful_poses / max(total_attempts, 1)
                
                # Compute region-specific success rates
                region_rates = {}
                for region in ['center', 'mid', 'periphery']:
                    if region_attempts[region] > 0:
                        region_rates[region] = region_success[region] / region_attempts[region]
                    else:
                        region_rates[region] = 0
                
                return {
                    'calibration_id': i,
                    'reprojection_error': my_dict["ReError"][i],
                    'success_rate': success_rate,
                    'region_success_rates': region_rates,
                    'mean_pose_error': np.mean(reprojection_errors) if reprojection_errors else float('inf'),
                    'camera_matrix': _fish_mat,
                    'dist_coeffs': _fish_dist,
                    'metadata': my_dict["metadata"][i]
                }
                
            except Exception as e:
                logger.warning(f"Evaluation failed for calibration {i}: {e}")
                return None
        
        # Evaluate all calibrations
        results = []
        for i in tqdm(range(len(my_dict["mat"])), desc="Evaluating calibrations"):
            result = process_calibration(i)
            if result is not None:
                results.append(result)
        
        return results
    
    def _get_marker_3d_points(self):
        """Get 3D points for ArUco marker"""
        marker_size = self.marker_length
        return np.array([
            [-marker_size/2, marker_size/2, 0],
            [marker_size/2, marker_size/2, 0],
            [marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
    
    def estimate_pose_single_markers(self, corners, marker_size, camera_matrix, distortion_coefficients):
        """Pose estimation with optional fisheye-specific refinement"""
        marker_points = self._get_marker_3d_points()
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
    
    def find_best_calibration(self, evaluation_results):
        """Find best calibration with sophisticated selection criteria"""
        if not evaluation_results:
            return None
        
        # Temperature-aware selection if enabled
        if self.config.use_temperature_selection:
            best_idx = self.temperature_aware_selection(
                [r['camera_matrix'] for r in evaluation_results],
                evaluation_results
            )
            if best_idx is not None:
                return evaluation_results[best_idx]
        
        # Multi-criteria sorting
        def score_calibration(result):
            # Base score from success rate and reprojection error
            score = result['success_rate'] - 0.1 * result['reprojection_error']
            
            # Bonus for good peripheral performance
            if 'periphery' in result['region_success_rates']:
                periph_rate = result['region_success_rates']['periphery']
                score += 0.2 * periph_rate  # Extra weight for periphery
            
            # Bonus for low pose estimation error
            if result['mean_pose_error'] < float('inf'):
                score -= 0.05 * result['mean_pose_error']
            
            # Bonus for advanced methods
            metadata = result.get('metadata', {})
            if metadata.get('bundle_adjusted', False):
                score += 0.1
            if metadata.get('ransac', False):
                score += 0.1
            
            return score
        
        # Sort by score
        sorted_results = sorted(evaluation_results, key=score_calibration, reverse=True)
        
        best = sorted_results[0]
        logger.info(f"\nBest calibration found:")
        logger.info(f"  Overall success rate: {best['success_rate']:.2%}")
        logger.info(f"  Reprojection error: {best['reprojection_error']:.4f}")
        logger.info(f"  Mean pose error: {best['mean_pose_error']:.4f}")
        logger.info(f"  Region success rates:")
        for region, rate in best['region_success_rates'].items():
            logger.info(f"    {region}: {rate:.2%}")
        
        metadata = best.get('metadata', {})
        if metadata:
            logger.info(f"  Advanced methods used:")
            if metadata.get('bundle_adjusted'):
                logger.info(f"    - Bundle adjustment")
            if metadata.get('ransac'):
                logger.info(f"    - RANSAC")
            if metadata.get('cv_fold', -1) >= 0:
                logger.info(f"    - Cross-validation fold {metadata['cv_fold']}")
        
        return best
    
    def load_april_tag_data(self, data_path):
        """Load April tag data with optional adaptive detection"""
        logger.info(f"Loading April tag data from {data_path}")
        
        corners, ids = [], []
        with open(data_path, "rb") as f:
            unpacker = mp.Unpacker(f, object_hook=mpn.decode)
            for frame in tqdm(unpacker):
                if self.config.use_adaptive_aruco:
                    # Use adaptive detection (requires camera params, using defaults for now)
                    _c, _i, _ = self.adaptive_aruco_detection(
                        frame, 
                        np.eye(3) * 600,  # Dummy camera matrix
                        np.zeros(4)       # Dummy distortion
                    )
                else:
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
    
    def save_calibration_toml(self, best_calibration, output_path):
        """Save calibration with enhanced metadata"""
        data = {
            "calibration": {
                "camera_matrix": best_calibration['camera_matrix'].tolist(),
                "dist_coeffs": best_calibration['dist_coeffs'].tolist(),
                "reprojection_error": float(best_calibration['reprojection_error']),
                "success_rate": float(best_calibration['success_rate']),
                "mean_pose_error": float(best_calibration.get('mean_pose_error', 0))
            },
            "pose": {
                "human_pose": False,
                "marker_pose": True
            },
            "aruco": {
                "marker_length": self.marker_length,
                "marker_spacing": self.marker_separation
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
            },
            "advanced_features": {
                "adaptive_selection": self.config.use_adaptive_selection,
                "multi_scale": self.config.use_multi_scale,
                "ransac": self.config.use_ransac,
                "bundle_adjustment": self.config.use_bundle_adjustment,
                "cross_validation": self.config.use_cross_validation,
                "temperature_selection": self.config.use_temperature_selection,
                "cache_undistortion": self.config.cache_undistortion_maps,
                "adaptive_aruco": self.config.use_adaptive_aruco,
                "fisheye_refinement": self.config.use_fisheye_refinement
            }
        }
        
        # Add region-specific performance
        if 'region_success_rates' in best_calibration:
            data["calibration"]["region_performance"] = {
                k: float(v) for k, v in best_calibration['region_success_rates'].items()
            }
        
        with open(output_path, 'w') as f:
            toml.dump(data, f)
        
        logger.info(f"Calibration saved to {output_path}")


def main():
    """Main calibration pipeline with configurable advanced features"""
    
    # Configuration - EDIT THESE TO ENABLE/DISABLE FEATURES
    config = CalibrationConfig(
        # Basic settings
        pattern_size=(6, 4),
        square_size=30,
        img_size=(1200, 800),
        
        # Advanced features - set to True to enable
        use_adaptive_selection=True,      # Intelligent sample selection
        use_multi_scale=True,            # Region-based weighting
        use_ransac=True,                 # RANSAC robust estimation
        use_bundle_adjustment=False,      # Bundle adjustment refinement (slower)
        use_cross_validation=True,        # Cross-validation evaluation
        use_temperature_selection=True,  # Simulated annealing selection
        cache_undistortion_maps=True,     # Cache for performance
        use_adaptive_aruco=False,         # Adaptive ArUco detection
        use_fisheye_refinement=True,     # Fisheye-specific corner refinement
        
        # Advanced parameters
        ransac_iterations=50,
        ransac_threshold=2.0,
        cv_folds=50,
        multi_scale_weights=[1.0, 1.5, 2.0]  # center, mid, periphery
    )
    
    # Initialize calibrator
    calibrator = EnhancedFisheyeCalibration(config)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("ENHANCED FISHEYE CALIBRATION")
    logger.info("=" * 60)
    logger.info("Configuration:")
    for key, value in vars(config).items():
        if key.startswith('use_') and value:
            logger.info(f"  ✓ {key.replace('use_', '').replace('_', ' ').title()}")
    logger.info("=" * 60)
    
    # Paths - update these for your system
    base_path = os.path.dirname(os.path.abspath(__file__))
    calib_video_path = os.path.join(base_path, "data", "calibration", '160_fov', "calib_mono_160fov_raw", "webcam_color.msgpack")
    april_data_path = os.path.join(base_path, "data", 'recordings', "160_fov", "3marker_complete_data", "3marker_april_mono_160fov_3", "webcam_color.msgpack")
    
    # Check files exist
    if not os.path.exists(calib_video_path):
        logger.error(f"Calibration video not found: {calib_video_path}")
        return
    
    # Step 1: Extract chessboard corners
    logger.info("\nStep 1: Extracting chessboard corners")
    chessb_corners = calibrator.extract_chessboard_corners(calib_video_path)
    
    if len(chessb_corners) < 20:
        logger.error("Not enough corner data found")
        return
    
    # Step 2: Generate multiple calibrations
    logger.info("\nStep 2: Generating calibrations")
    n_calibrations = 100 if config.use_bundle_adjustment else 500  # Fewer if using slow methods
    my_dict = calibrator.generate_calibrations(chessb_corners, n_calibrations=n_calibrations)
    
    if len(my_dict["mat"]) == 0:
        logger.error("No valid calibrations generated")
        return
    
    # Step 3: Load April tag data for evaluation
    logger.info("\nStep 3: Evaluation")
    if os.path.exists(april_data_path):
        corners, ids, timestamps, sync = calibrator.load_april_tag_data(april_data_path)
        
        # Step 4: Evaluate calibrations
        evaluation_results = calibrator.evaluate_calibration(my_dict, corners, ids)
        
        # Step 5: Find best calibration
        best_calibration = calibrator.find_best_calibration(evaluation_results)
        
        if best_calibration:
            # Save to TOML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"enhanced_fisheye_calibration_{timestamp}.toml"
            calibrator.save_calibration_toml(best_calibration, output_path)
            
            # Also save a comparison without advanced features
            logger.info("\nGenerating baseline calibration for comparison...")
            baseline_config = CalibrationConfig()  # All features disabled
            baseline_calibrator = EnhancedFisheyeCalibration(baseline_config)
            baseline_dict = baseline_calibrator.generate_calibrations(chessb_corners, n_calibrations=50)
            
            if len(baseline_dict["mat"]) > 0:
                baseline_results = baseline_calibrator.evaluate_calibration(baseline_dict, corners, ids)
                baseline_best = baseline_calibrator.find_best_calibration(baseline_results)
                
                if baseline_best:
                    logger.info("\nComparison:")
                    logger.info(f"Enhanced - Success: {best_calibration['success_rate']:.2%}, "
                              f"RepErr: {best_calibration['reprojection_error']:.4f}")
                    logger.info(f"Baseline - Success: {baseline_best['success_rate']:.2%}, "
                              f"RepErr: {baseline_best['reprojection_error']:.4f}")
                    
                    improvement = (best_calibration['success_rate'] - baseline_best['success_rate']) * 100
                    logger.info(f"Improvement: {improvement:+.1f} percentage points")
        
    else:
        logger.warning(f"April tag data not found: {april_data_path}")
        logger.info("Using best reprojection error calibration")
        
        # Just use the calibration with lowest reprojection error
        best_idx = np.argmin(my_dict["ReError"])
        best_calibration = {
            'camera_matrix': my_dict["mat"][best_idx],
            'dist_coeffs': my_dict["dist"][best_idx],
            'reprojection_error': my_dict["ReError"][best_idx],
            'success_rate': 0.0  # Unknown without April tags
        }
        
        output_path = "enhanced_fisheye_calibration_reproj_best.toml"
        calibrator.save_calibration_toml(best_calibration, output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Calibration complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    from datetime import datetime
    main()