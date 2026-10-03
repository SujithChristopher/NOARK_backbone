"""
Iterative Fisheye Calibration with AprilTag-based Refinement
Uses known geometry of 3 AprilTags for calibration validation and refinement
"""

import numpy as np
import cv2
import os
import msgpack as mp
import msgpack_numpy as mpn
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, differential_evolution
from scipy.spatial.distance import cdist
import logging
import toml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, NamedTuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AprilTagBoard:
    """Represents the 3D printed board with 3 AprilTags in isosceles triangle configuration"""
    
    def __init__(self, tag_ids=[12, 14, 20], side_length=92.0, base_length=82.0, marker_size=50.0):
        """
        Initialize AprilTag board configuration for isosceles triangle
        
        Args:
            tag_ids: List of AprilTag IDs [top, bottom_left, bottom_right]
            side_length: Length of the two equal sides in mm (default: 92mm)
            base_length: Length of the base in mm (default: 82mm)
            marker_size: Size of each AprilTag marker in mm
        """
        self.tag_ids = tag_ids
        self.side_length = side_length  # The two equal sides
        self.base_length = base_length  # The base
        self.marker_size = marker_size
        
        # Calculate height of isosceles triangle
        # Using the formula: h = sqrt(side_length^2 - (base_length/2)^2)
        height = np.sqrt(side_length**2 - (base_length/2)**2)
        
        # Define tag positions in board coordinate system (mm)
        # Place origin at the centroid of the triangle
        centroid_y = height / 3  # Centroid is at 1/3 height from base
        
        self.tag_positions_3d = {
            tag_ids[0]: np.array([0, height - centroid_y, 0]),  # Top (apex)
            tag_ids[1]: np.array([-base_length/2, -centroid_y, 0]),  # Bottom left
            tag_ids[2]: np.array([base_length/2, -centroid_y, 0])   # Bottom right
        }
        
        # Store relative constraints with actual measured distances
        self.expected_distances = {
            (tag_ids[0], tag_ids[1]): side_length,  # Top to bottom-left
            (tag_ids[0], tag_ids[2]): side_length,  # Top to bottom-right  
            (tag_ids[1], tag_ids[2]): base_length   # Bottom-left to bottom-right
        }
        
        # Also store in sorted order for easier lookup
        self.expected_distances.update({
            tuple(sorted(k)): v for k, v in self.expected_distances.items()
        })
        
        logger.info(f"AprilTag board initialized:")
        logger.info(f"  Isosceles triangle: sides={side_length}mm, base={base_length}mm")
        logger.info(f"  Height: {height:.1f}mm")
        logger.info(f"  Tag positions:")
        for tag_id, pos in self.tag_positions_3d.items():
            logger.info(f"    Tag {tag_id}: {pos}")
        logger.info(f"  Expected distances:")
        for (id1, id2), dist in self.expected_distances.items():
            if id1 < id2:  # Only print each pair once
                logger.info(f"    Tag {id1} to Tag {id2}: {dist}mm")
    
    def get_tag_corners_3d(self, tag_id):
        """Get 3D corners of a specific tag in board coordinates"""
        if tag_id not in self.tag_ids:
            return None
        
        center = self.tag_positions_3d[tag_id]
        half_size = self.marker_size / 2
        
        # Corners in standard order (same as OpenCV)
        corners = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ]) + center
        
        return corners
    
    def validate_detections(self, detected_positions, tolerance_mm=10.0):
        """
        Validate detected tag positions against expected geometry
        
        Returns:
            dict: Validation results including errors and validity
        """
        validation_result = {
            'valid': True,
            'distance_errors': {},
            'max_error': 0,
            'mean_error': 0
        }
        
        errors = []
        for (id1, id2), expected_dist in self.expected_distances.items():
            if id1 in detected_positions and id2 in detected_positions:
                actual_dist = np.linalg.norm(
                    detected_positions[id1] - detected_positions[id2]
                )
                error = abs(actual_dist - expected_dist)
                errors.append(error)
                validation_result['distance_errors'][(id1, id2)] = error
                
                if error > tolerance_mm:
                    validation_result['valid'] = False
        
        if errors:
            validation_result['max_error'] = max(errors)
            validation_result['mean_error'] = np.mean(errors)
        
        return validation_result


def objective_function_standalone(params, all_detections, april_board, img_size):
    """
    Standalone objective function for calibration optimization
    """
    # Unpack parameters
    fx, fy, cx, cy = params[:4]
    k1, k2, k3, k4 = params[4:8]
    
    # Build camera matrix and distortion
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([k1, k2, k3, k4])
    
    total_error = 0
    constraint_penalty = 0
    valid_frames = 0
    
    for detections in all_detections:
        if len(detections) < 2:
            continue
        
        # Estimate board pose with current parameters
        object_points = []
        image_points = []
        
        for tag_id, corners_2d in detections.items():
            corners_3d = april_board.get_tag_corners_3d(tag_id)
            if corners_3d is not None:
                object_points.extend(corners_3d)
                image_points.extend(corners_2d)
        
        if len(object_points) < 4:
            continue
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        try:
            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K, D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                continue
            
            # Compute reprojection error
            projected, _ = cv2.fisheye.projectPoints(
                object_points.reshape(-1, 1, 3),
                rvec, tvec,
                K, D
            )
            
            error = np.linalg.norm(
                image_points.reshape(-1, 2) - projected.reshape(-1, 2),
                axis=1
            ).mean()
            
            total_error += error
            valid_frames += 1
            
            # Add geometric constraint penalty
            # Project tag centers and check distances
            tag_centers_3d = []
            tag_centers_2d = []
            for tag_id in detections.keys():
                center_3d = april_board.tag_positions_3d[tag_id]
                tag_centers_3d.append(center_3d)
                
                # Project center
                center_proj, _ = cv2.fisheye.projectPoints(
                    center_3d.reshape(1, 1, 3),
                    rvec, tvec,
                    K, D
                )
                tag_centers_2d.append(center_proj.reshape(2))
            
            # Check relative distances in image space
            if len(tag_centers_2d) >= 2:
                for i in range(len(tag_centers_2d)):
                    for j in range(i+1, len(tag_centers_2d)):
                        # Expected ratio based on 3D distances
                        dist_3d = np.linalg.norm(tag_centers_3d[i] - tag_centers_3d[j])
                        dist_2d = np.linalg.norm(tag_centers_2d[i] - tag_centers_2d[j])
                        
                        # This is a soft constraint - we expect consistent scaling
                        # but allow for perspective effects
                        # Use the maximum expected distance as reference
                        max_expected = max(april_board.expected_distances.values())
                        expected_ratio = dist_3d / max_expected
                        if valid_frames > 0:
                            constraint_penalty += 0.1 * abs(dist_2d - expected_ratio * dist_2d)
            
        except Exception as e:
            continue
    
    if valid_frames == 0:
        return 1e6
    
    return total_error / valid_frames + constraint_penalty


class IterativeFisheyeCalibrator:
    """Calibrator that uses AprilTag constraints for iterative refinement"""
    
    def __init__(self, initial_calibration, april_board: AprilTagBoard, img_size=(1200, 800)):
        # Extract and validate camera matrix
        if 'camera_matrix' in initial_calibration:
            self.camera_matrix = np.array(initial_calibration['camera_matrix'], dtype=np.float64)
        else:
            raise ValueError("camera_matrix not found in calibration data")
        
        # Extract and validate distortion coefficients
        if 'dist_coeffs' in initial_calibration:
            dist = np.array(initial_calibration['dist_coeffs'], dtype=np.float64).flatten()
            # Ensure we have exactly 4 coefficients for fisheye model
            if len(dist) < 4:
                self.dist_coeffs = np.pad(dist, (0, 4 - len(dist)), 'constant')
            else:
                self.dist_coeffs = dist[:4]
        else:
            raise ValueError("dist_coeffs not found in calibration data")
        
        self.img_size = img_size
        self.april_board = april_board
        
        # Setup ArUco detector
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Optimization history
        self.optimization_history = []
    
    def detect_and_validate_frame(self, frame):
        """Detect AprilTags and validate geometry"""
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is None or len(ids) < 2:
            return None, None, None
        
        # Filter for our specific tags
        valid_detections = {}
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id in self.april_board.tag_ids:
                valid_detections[tag_id] = corners[i][0]
        
        if len(valid_detections) < 2:
            return None, None, None
        
        return valid_detections, ids, corners
    
    def estimate_board_pose(self, detections):
        """
        Estimate full board pose from detected tags
        Uses PnP with all available tag corners
        """
        # Collect all 3D-2D correspondences
        object_points = []
        image_points = []
        
        for tag_id, corners_2d in detections.items():
            corners_3d = self.april_board.get_tag_corners_3d(tag_id)
            if corners_3d is not None:
                object_points.extend(corners_3d)
                image_points.extend(corners_2d)
        
        if len(object_points) < 4:
            return None, None
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None
        
        return rvec, tvec
    
    def compute_reprojection_error(self, detections, rvec, tvec):
        """Compute reprojection error for all detected tags"""
        errors = []
        
        for tag_id, corners_2d in detections.items():
            corners_3d = self.april_board.get_tag_corners_3d(tag_id)
            if corners_3d is not None:
                # Project 3D points
                projected, _ = cv2.fisheye.projectPoints(
                    corners_3d.reshape(-1, 1, 3),
                    rvec, tvec,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                # Compute error
                error = np.linalg.norm(
                    corners_2d.reshape(-1, 2) - projected.reshape(-1, 2),
                    axis=1
                ).mean()
                errors.append(error)
        
        return np.mean(errors) if errors else float('inf')
    
    def refine_calibration_simple(self, april_detections, max_iterations=100, tolerance=0.01):
        """
        Alternative simpler refinement using scipy.optimize.minimize
        """
        logger.info("Starting simple iterative calibration refinement (no multiprocessing)")
        
        # Filter valid detections (multiple tags visible)
        valid_detections = []
        for det in april_detections:
            if isinstance(det, dict) and len(det) >= 2:
                valid_detections.append(det)
        
        logger.info(f"Using {len(valid_detections)} frames with multiple tags")
        
        if len(valid_detections) < 10:
            logger.warning("Too few valid detections for refinement")
            return False
        
        # Initial parameters
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        k1 = float(self.dist_coeffs[0])
        k2 = float(self.dist_coeffs[1])
        k3 = float(self.dist_coeffs[2])
        k4 = float(self.dist_coeffs[3])
        
        x0 = np.array([fx, fy, cx, cy, k1, k2, k3, k4])
        
        # Bounds for parameters
        bounds = [
            (x0[0]*0.8, x0[0]*1.2),  # fx
            (x0[1]*0.8, x0[1]*1.2),  # fy
            (x0[2]*0.8, x0[2]*1.2),  # cx
            (x0[3]*0.8, x0[3]*1.2),  # cy
            (-2, 2),                  # k1
            (-2, 2),                  # k2
            (-2, 2),                  # k3
            (-2, 2)                   # k4
        ]
        
        # Initial error
        initial_error = objective_function_standalone(x0, valid_detections, self.april_board, self.img_size)
        logger.info(f"Initial error: {initial_error:.4f}")
        
        # Use L-BFGS-B which doesn't require multiprocessing
        from scipy.optimize import minimize
        
        result = minimize(
            objective_function_standalone,
            x0,
            args=(valid_detections, self.april_board, self.img_size),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': tolerance, 'disp': True}
        )
        
        if result.success:
            # Update calibration
            self.camera_matrix[0, 0] = result.x[0]  # fx
            self.camera_matrix[1, 1] = result.x[1]  # fy
            self.camera_matrix[0, 2] = result.x[2]  # cx
            self.camera_matrix[1, 2] = result.x[3]  # cy
            self.dist_coeffs[0] = result.x[4]       # k1
            self.dist_coeffs[1] = result.x[5]       # k2
            self.dist_coeffs[2] = result.x[6]       # k3
            self.dist_coeffs[3] = result.x[7]       # k4
            
            final_error = result.fun
            logger.info(f"Final error: {final_error:.4f}")
            logger.info(f"Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")
            
            # Store optimization history
            self.optimization_history.append({
                'initial_error': initial_error,
                'final_error': final_error,
                'iterations': result.nit,
                'parameters': result.x.tolist()
            })
            
            return True
        
        return False
    
    def objective_function(self, params, all_detections):
        """
        Objective function wrapper that calls standalone version
        """
        return objective_function_standalone(
            params, all_detections, self.april_board, self.img_size
        )
    
    def refine_calibration(self, april_detections, max_iterations=10, tolerance=0.01):
        """
        Iteratively refine calibration using AprilTag detections
        """
        logger.info("Starting iterative calibration refinement")
        
        # Filter valid detections (multiple tags visible)
        valid_detections = []
        for det in april_detections:
            if isinstance(det, dict) and len(det) >= 2:
                valid_detections.append(det)
        
        logger.info(f"Found {len(valid_detections)} frames with multiple tags")
        
        # Sample if too many frames (for computational efficiency)
        max_frames = 200
        if len(valid_detections) > max_frames:
            logger.info(f"Sampling {max_frames} frames from {len(valid_detections)} for efficiency")
            # Sample evenly across the dataset
            indices = np.linspace(0, len(valid_detections)-1, max_frames, dtype=int)
            valid_detections = [valid_detections[i] for i in indices]
        
        logger.info(f"Using {len(valid_detections)} frames for optimization")
        
        if len(valid_detections) < 10:
            logger.warning("Too few valid detections for refinement")
            return False
        
        # Initial parameters
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        k1 = float(self.dist_coeffs[0])
        k2 = float(self.dist_coeffs[1])
        k3 = float(self.dist_coeffs[2])
        k4 = float(self.dist_coeffs[3])
        
        x0 = np.array([fx, fy, cx, cy, k1, k2, k3, k4])
        
        # Bounds for parameters
        bounds = [
            (x0[0]*0.8, x0[0]*1.2),  # fx
            (x0[1]*0.8, x0[1]*1.2),  # fy
            (x0[2]*0.8, x0[2]*1.2),  # cx
            (x0[3]*0.8, x0[3]*1.2),  # cy
            (-2, 2),                  # k1
            (-2, 2),                  # k2
            (-2, 2),                  # k3
            (-2, 2)                   # k4
        ]
        
        # Initial error
        initial_error = self.objective_function(x0, valid_detections)
        logger.info(f"Initial error: {initial_error:.4f}")
        
        # Create callback for progress tracking
        iteration_count = 0
        def callback(xk, convergence=None):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count % 5 == 0:
                current_error = self.objective_function(xk, valid_detections)
                logger.info(f"Iteration {iteration_count}: error = {current_error:.4f}")
        
        # Optimize using differential evolution for global optimization
        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(valid_detections,),
                maxiter=max_iterations,
                tol=tolerance,
                seed=42,
                workers=1,  # Use single worker to avoid pickling issues
                popsize=10,  # Smaller population for faster convergence
                callback=callback,
                updating='deferred',
                disp=True
            )
            
            success = result.success
            final_params = result.x
            final_error = result.fun
            
        except Exception as e:
            logger.warning(f"Differential evolution failed: {e}")
            logger.info("Falling back to L-BFGS-B optimization...")
            
            # Fallback to simpler optimization
            from scipy.optimize import minimize
            
            result = minimize(
                self.objective_function,
                x0,
                args=(valid_detections,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': tolerance, 'disp': True}
            )
            
            success = result.success
            final_params = result.x
            final_error = result.fun
        
        if success:
            # Update calibration
            self.camera_matrix[0, 0] = final_params[0]  # fx
            self.camera_matrix[1, 1] = final_params[1]  # fy
            self.camera_matrix[0, 2] = final_params[2]  # cx
            self.camera_matrix[1, 2] = final_params[3]  # cy
            self.dist_coeffs[0] = final_params[4]       # k1
            self.dist_coeffs[1] = final_params[5]       # k2
            self.dist_coeffs[2] = final_params[6]       # k3
            self.dist_coeffs[3] = final_params[7]       # k4
            
            logger.info(f"Final error: {final_error:.4f}")
            logger.info(f"Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")
            
            # Store optimization history
            self.optimization_history.append({
                'initial_error': initial_error,
                'final_error': final_error,
                'iterations': iteration_count,
                'parameters': final_params.tolist()
            })
            
            return True
        
        logger.warning("Optimization failed to converge")
        return False
    
    def validate_with_geometry(self, april_detections, sample_size=100):
        """
        Validate calibration by checking geometric consistency across frames
        """
        validation_results = []
        
        for i, detections in enumerate(april_detections[:sample_size]):
            if not isinstance(detections, dict) or len(detections) < 3:
                continue
            
            # Estimate board pose
            rvec, tvec = self.estimate_board_pose(detections)
            if rvec is None:
                continue
            
            # Get 3D positions of detected tags
            detected_positions_3d = {}
            for tag_id in detections.keys():
                if tag_id in self.april_board.tag_ids:
                    # Transform tag center to camera coordinates
                    center = self.april_board.tag_positions_3d[tag_id]
                    center_cam = cv2.Rodrigues(rvec)[0] @ center + tvec.flatten()
                    detected_positions_3d[tag_id] = center_cam
            
            # Validate geometry
            validation = self.april_board.validate_detections(detected_positions_3d)
            validation['frame'] = i
            validation['reprojection_error'] = self.compute_reprojection_error(
                detections, rvec, tvec
            )
            validation_results.append(validation)
        
        # Summarize validation
        if validation_results:
            mean_reproj = np.mean([v['reprojection_error'] for v in validation_results])
            mean_geom = np.mean([v['mean_error'] for v in validation_results])
            valid_ratio = sum(1 for v in validation_results if v['valid']) / len(validation_results)
            
            logger.info(f"\nValidation Summary:")
            logger.info(f"  Mean reprojection error: {mean_reproj:.2f} pixels")
            logger.info(f"  Mean geometric error: {mean_geom:.2f} mm")
            logger.info(f"  Valid geometry ratio: {valid_ratio:.1%}")
            
            return {
                'mean_reprojection_error': mean_reproj,
                'mean_geometric_error': mean_geom,
                'valid_ratio': valid_ratio,
                'details': validation_results
            }
        
        return None


def extract_april_detections(corners, ids, april_board):
    """Extract structured detections from raw corners and ids"""
    detections = []
    
    for frame_corners, frame_ids in zip(corners, ids):
        if frame_ids is None:
            detections.append({})
            continue
        
        frame_detections = {}
        for i, tag_id in enumerate(frame_ids.flatten()):
            if tag_id in april_board.tag_ids:
                frame_detections[tag_id] = frame_corners[i][0]
        
        detections.append(frame_detections)
    
    return detections


def compare_calibrations(original_calib, refined_calib, april_detections, april_board):
    """Compare original and refined calibrations"""
    logger.info("\nComparing calibrations...")
    
    # Create validators for both calibrations
    original_validator = IterativeFisheyeCalibrator(
        original_calib, april_board, original_calib['img_size']
    )
    refined_validator = IterativeFisheyeCalibrator(
        refined_calib, april_board, refined_calib['img_size']
    )
    
    # Validate both
    logger.info("\nOriginal calibration:")
    orig_results = original_validator.validate_with_geometry(april_detections)
    
    logger.info("\nRefined calibration:")
    refined_results = refined_validator.validate_with_geometry(april_detections)
    
    # Compare pose estimation accuracy
    if orig_results and refined_results:
        improvement = {
            'reprojection': (orig_results['mean_reprojection_error'] - 
                           refined_results['mean_reprojection_error']) / 
                          orig_results['mean_reprojection_error'] * 100,
            'geometric': (orig_results['mean_geometric_error'] - 
                        refined_results['mean_geometric_error']) / 
                       orig_results['mean_geometric_error'] * 100,
            'validity': (refined_results['valid_ratio'] - 
                       orig_results['valid_ratio']) * 100
        }
        
        logger.info("\nImprovement Summary:")
        logger.info(f"  Reprojection error: {improvement['reprojection']:+.1f}%")
        logger.info(f"  Geometric accuracy: {improvement['geometric']:+.1f}%")
        logger.info(f"  Valid detections: {improvement['validity']:+.1f} pp")
        
        return improvement
    
    return None


def save_refined_calibration(calibrator, original_path, april_board, validation_results=None):
    """Save refined calibration with metadata"""
    import datetime
    
    data = {
        "calibration": {
            "camera_matrix": calibrator.camera_matrix.tolist(),
            "dist_coeffs": calibrator.dist_coeffs.tolist(),
            "refinement_method": "april_tag_geometry",
            "refinement_date": datetime.datetime.now().isoformat()
        },
        "april_board": {
            "tag_ids": april_board.tag_ids,
            "triangle_side_length": april_board.triangle_side,
            "marker_size": april_board.marker_size
        },
        "optimization_history": calibrator.optimization_history,
        "camera": {
            "resolution": list(calibrator.img_size)
        }
    }
    
    if validation_results:
        data["validation"] = {
            "mean_reprojection_error": validation_results['mean_reprojection_error'],
            "mean_geometric_error": validation_results['mean_geometric_error'],
            "valid_ratio": validation_results['valid_ratio']
        }
    
    # Save to new file
    base_name = os.path.splitext(original_path)[0]
    refined_path = f"{base_name}_refined_april.toml"
    
    with open(refined_path, 'w') as f:
        toml.dump(data, f)
    
    logger.info(f"Refined calibration saved to: {refined_path}")
    return refined_path


def debug_calibration_data(calib_data):
    """Debug function to check calibration data format"""
    logger.info("Debugging calibration data:")
    logger.info(f"Keys in calibration data: {list(calib_data.keys())}")
    
    if 'calibration' in calib_data:
        logger.info(f"Keys in calibration: {list(calib_data['calibration'].keys())}")
        
        if 'camera_matrix' in calib_data['calibration']:
            cam_mat = calib_data['calibration']['camera_matrix']
            logger.info(f"Camera matrix type: {type(cam_mat)}")
            logger.info(f"Camera matrix shape: {np.array(cam_mat).shape}")
            logger.info(f"Camera matrix:\n{np.array(cam_mat)}")
        
        if 'dist_coeffs' in calib_data['calibration']:
            dist = calib_data['calibration']['dist_coeffs']
            logger.info(f"Distortion coeffs type: {type(dist)}")
            logger.info(f"Distortion coeffs shape: {np.array(dist).shape}")
            logger.info(f"Distortion coeffs: {dist}")


def main():
    """Main iterative refinement pipeline"""
    import sys
    
    # Configuration with actual measured dimensions
    april_board = AprilTagBoard(
        tag_ids=[12, 14, 20],
        side_length=92.0,     # 9.2 cm equal sides
        base_length=82.0,     # 8.2 cm base
        marker_size=50.0      # Adjust based on your actual markers
    )
    
    # Load initial calibration
    if len(sys.argv) > 1:
        initial_calib_path = sys.argv[1]
    else:
        initial_calib_path = "enhanced_fisheye_calibration_20250804_151152.toml"
    
    logger.info(f"Loading initial calibration from: {initial_calib_path}")
    
    with open(initial_calib_path, 'r') as f:
        calib_data = toml.load(f)
    
    # Debug the calibration data structure
    debug_calibration_data(calib_data)
    
    # Extract calibration parameters with proper error handling
    try:
        if 'calibration' in calib_data:
            cam_mat = calib_data['calibration']['camera_matrix']
            dist_coeffs = calib_data['calibration']['dist_coeffs']
        else:
            # Try direct access
            cam_mat = calib_data['camera_matrix']
            dist_coeffs = calib_data['dist_coeffs']
        
        initial_calib = {
            'camera_matrix': np.array(cam_mat, dtype=np.float64),
            'dist_coeffs': np.array(dist_coeffs, dtype=np.float64),
            'img_size': tuple(calib_data.get('camera', {}).get('resolution', [1200, 800]))
        }
        
        logger.info(f"Successfully loaded calibration:")
        logger.info(f"  Camera matrix shape: {initial_calib['camera_matrix'].shape}")
        logger.info(f"  Distortion coeffs shape: {initial_calib['dist_coeffs'].shape}")
        logger.info(f"  Image size: {initial_calib['img_size']}")
        
    except Exception as e:
        logger.error(f"Error loading calibration data: {e}")
        logger.error("Please check the calibration file format")
        return
    
    # Load AprilTag data
    base_path = os.path.dirname(os.path.abspath(__file__))
    april_data_path = os.path.join(
        base_path, "data", 'recordings', "160_fov", 
        "3marker_complete_data", "3marker_april_mono_160fov_3", 
        "webcam_color.msgpack"
    )
    
    logger.info(f"Loading AprilTag data from: {april_data_path}")
    
    # Setup ArUco detector
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Load and detect AprilTags
    corners, ids = [], []
    with open(april_data_path, "rb") as f:
        unpacker = mp.Unpacker(f, object_hook=mpn.decode)
        for frame in tqdm(unpacker, desc="Detecting AprilTags"):
            _c, _i, _ = detector.detectMarkers(frame)
            corners.append(_c)
            ids.append(_i)
    
    # Extract structured detections
    april_detections = extract_april_detections(corners, ids, april_board)
    
    # Create calibrator
    calibrator = IterativeFisheyeCalibrator(initial_calib, april_board, initial_calib['img_size'])
    
    # Refine calibration - use simple version to avoid multiprocessing issues
    logger.info("\nStarting calibration refinement...")
    try:
        # Try the differential evolution with reduced iterations
        success = calibrator.refine_calibration(april_detections, max_iterations=10)
    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        logger.info("Falling back to simple optimization...")
        # Fall back to simple optimization
        success = calibrator.refine_calibration_simple(april_detections, max_iterations=50)
    
    if success:
        # Create refined calibration dict
        refined_calib = {
            'camera_matrix': calibrator.camera_matrix.copy(),
            'dist_coeffs': calibrator.dist_coeffs.copy(),
            'img_size': calibrator.img_size
        }
        
        # Compare calibrations
        improvement = compare_calibrations(
            initial_calib, refined_calib, april_detections, april_board
        )
        
        # Final validation
        logger.info("\nFinal validation of refined calibration:")
        validation_results = calibrator.validate_with_geometry(april_detections)
        
        # Save refined calibration
        save_refined_calibration(
            calibrator, initial_calib_path, april_board, validation_results
        )
        
        logger.info("\nRefinement complete!")
    else:
        logger.error("Refinement failed!")


if __name__ == "__main__":
    main()