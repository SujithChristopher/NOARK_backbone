"""
Fisheye calibration using k-means clustering to sample diverse calibration frames
Uses pre-extracted chessboard corners from msgpack file
"""

import numpy as np
import cv2
import msgpack
from sklearn.cluster import KMeans
from tqdm import tqdm
import toml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_numpy(obj):
    """Decode msgpack numpy format"""
    if isinstance(obj, dict) and b'nd' in obj:
        dtype = np.dtype(obj[b'kind'].decode() + str(obj[b'type']))
        return np.frombuffer(obj[b'data'], dtype=dtype).reshape(obj[b'shape'])
    return obj


def load_chessboard_corners(msgpack_path):
    """Load pre-extracted chessboard corners from msgpack file"""
    logger.info(f"Loading chessboard corners from {msgpack_path}")

    corners_list = []
    with open(msgpack_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for item in unpacker:
            decoded = decode_numpy(item)
            corners_list.append(decoded)

    corners_array = np.array(corners_list)
    logger.info(f"Loaded {len(corners_array)} frames with shape {corners_array[0].shape}")

    return corners_array


def compute_corner_features(corners_array):
    """
    Compute features for k-means clustering based on corner spread
    Features: mean position, std dev, and corner coverage of image space
    """
    logger.info("Computing features for k-means clustering")

    features = []
    for corners in corners_array:
        # Flatten corners to (N, 2)
        pts = corners.reshape(-1, 2)

        # Feature 1-2: Mean x, y position
        mean_x = pts[:, 0].mean()
        mean_y = pts[:, 1].mean()

        # Feature 3-4: Standard deviation of x, y (spread)
        std_x = pts[:, 0].std()
        std_y = pts[:, 1].std()

        # Feature 5-8: Min and max positions (coverage)
        min_x = pts[:, 0].min()
        max_x = pts[:, 0].max()
        min_y = pts[:, 1].min()
        max_y = pts[:, 1].max()

        # Feature 9-10: Range of corners
        range_x = max_x - min_x
        range_y = max_y - min_y

        features.append([mean_x, mean_y, std_x, std_y, min_x, max_x, min_y, max_y, range_x, range_y])

    return np.array(features)


def sample_diverse_frames_kmeans(corners_array, n_samples=200, random_state=42):
    """
    Use k-means clustering to sample diverse calibration frames
    Returns indices of selected frames
    """
    logger.info(f"Sampling {n_samples} diverse frames using k-means clustering (seed={random_state})")

    # Compute features for clustering
    features = compute_corner_features(corners_array)

    # Normalize features for better clustering
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Apply k-means clustering
    logger.info("Running k-means clustering...")
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state, n_init=10, verbose=0)
    kmeans.fit(features_normalized)

    # For each cluster, select the frame closest to cluster center
    selected_indices = []
    for cluster_id in range(n_samples):
        # Find all frames in this cluster
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) > 0:
            # Get features for this cluster
            cluster_features = features_normalized[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]

            # Find frame closest to cluster center
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

    logger.info(f"Selected {len(selected_indices)} frames from {n_samples} clusters")

    return np.array(selected_indices)


def construct_3d_points(pattern_size=(6, 4), square_size=40):
    """Construct 3D object points for chessboard"""
    X = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    X[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    X = X * square_size
    return X


def calibrate_fisheye(corners_subset, board_points, img_size=(1280, 800)):
    """
    Perform fisheye calibration on selected corner subset
    """
    logger.info(f"Performing fisheye calibration with {len(corners_subset)} frames")

    # Prepare world and image points
    world_points = []
    image_points = []

    for corners in corners_subset:
        image_points.append(corners)
        world_points.append(board_points)

    # Fisheye calibration flags
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
        ret, camera_matrix, distortion_coeffs, R, t = cv2.fisheye.calibrate(
            np.expand_dims(np.asarray(world_points), -2),
            image_points,
            img_size,
            None,
            None,
            flags=flags_calib,
            criteria=calibrate_criteria,
        )

        logger.info(f"Calibration successful! Reprojection error: {ret:.6f}")
        logger.info(f"\nCamera Matrix:\n{camera_matrix}")
        logger.info(f"\nDistortion Coefficients:\n{distortion_coeffs.flatten()}")

        # Calculate distortion percentage (k1 is primary radial distortion)
        k1 = distortion_coeffs[0, 0]
        distortion_percent = abs(k1) * 100
        logger.info(f"\nDistortion level: ~{distortion_percent:.1f}%")

        return {
            "reprojection_error": ret,
            "camera_matrix": camera_matrix,
            "dist_coeffs": distortion_coeffs,
            "rvecs": R,
            "tvecs": t
        }

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return None


def save_calibration_toml(calibration_result, output_path, pattern_size=(6, 4)):
    """Save calibration results to TOML file"""
    logger.info(f"Saving calibration to {output_path}")

    data = {
        "calibration": {
            "camera_matrix": calibration_result['camera_matrix'].tolist(),
            "dist_coeffs": calibration_result['dist_coeffs'].tolist(),
            "reprojection_error": float(calibration_result['reprojection_error']),
            "method": "fisheye",
            "pattern_size": list(pattern_size),
            "notes": "Calibrated using k-means sampled frames from OV9281 160° FOV camera"
        },
        "aruco": {
            "marker_length": 0.05,
            "marker_spacing": 0.01
        },
        "camera": {
            "resolution": [1280, 800],
            "model": "OV9281",
            "fov": 160
        },
        "stream_data": {
            "udp": False,
            "ip": "localhost",
            "port": 12345
        },
        "display": {
            "display": False
        }
    }

    with open(output_path, 'w') as f:
        toml.dump(data, f)

    logger.info(f"Calibration saved successfully!")


def generate_multiple_calibrations(corners_array, board_points, n_iterations=10, n_samples=200, img_size=(1280, 800)):
    """
    Generate multiple calibrations with different random samples to find the best one
    """
    logger.info(f"Generating {n_iterations} calibrations with different samplings")

    results = []

    for i in tqdm(range(n_iterations), desc="Calibration iterations"):
        # Sample diverse frames with different random seed each time
        random_seed = i * 100 + 42
        selected_indices = sample_diverse_frames_kmeans(corners_array, n_samples=n_samples, random_state=random_seed)
        corners_subset = corners_array[selected_indices]

        # Calibrate
        result = calibrate_fisheye(corners_subset, board_points, img_size)

        if result is not None:
            result['iteration'] = i
            result['selected_indices'] = selected_indices
            result['random_seed'] = random_seed
            results.append(result)

    logger.info(f"Generated {len(results)} successful calibrations")

    # Find best calibration (lowest reprojection error)
    if results:
        best_result = min(results, key=lambda x: x['reprojection_error'])
        logger.info(f"\nBest calibration from iteration {best_result['iteration']}")
        logger.info(f"Reprojection error: {best_result['reprojection_error']:.6f}")
        return best_result, results

    return None, []


def main():
    """Main calibration pipeline"""

    # Configuration
    msgpack_path = "chessboard_20251113_153002.msgpack"
    pattern_size = (6, 4)  # 6x4 chessboard
    square_size = 40  # 40mm (4cm) squares
    img_size = (1280, 800)
    n_samples = 200  # Number of frames to sample
    n_iterations = 10  # Number of calibration attempts with different samplings

    # Load pre-extracted corners
    corners_array = load_chessboard_corners(msgpack_path)

    if len(corners_array) == 0:
        logger.error("No corners loaded!")
        return

    # Construct 3D board points
    board_points = construct_3d_points(pattern_size, square_size)

    # Generate multiple calibrations and find the best
    best_result, all_results = generate_multiple_calibrations(
        corners_array,
        board_points,
        n_iterations=n_iterations,
        n_samples=n_samples,
        img_size=img_size
    )

    if best_result is None:
        logger.error("Calibration failed!")
        return

    # Save best calibration
    output_path = "calib_mono_1200_800_kmeans.toml"
    save_calibration_toml(best_result, output_path, pattern_size)

    # Print summary statistics
    if len(all_results) > 1:
        errors = [r['reprojection_error'] for r in all_results]
        logger.info(f"\nCalibration statistics across {len(all_results)} iterations:")
        logger.info(f"  Best error: {min(errors):.6f}")
        logger.info(f"  Worst error: {max(errors):.6f}")
        logger.info(f"  Mean error: {np.mean(errors):.6f}")
        logger.info(f"  Std dev: {np.std(errors):.6f}")

    logger.info("\nCalibration complete!")


if __name__ == "__main__":
    main()
