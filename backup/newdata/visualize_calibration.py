"""
Simple visualization script to verify fisheye calibration
Shows original vs undistorted chessboard corners
"""

import numpy as np
import cv2
import msgpack
import toml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_numpy(obj):
    """Decode msgpack numpy format"""
    if isinstance(obj, dict) and b'nd' in obj:
        dtype = np.dtype(obj[b'kind'].decode() + str(obj[b'type']))
        return np.frombuffer(obj[b'data'], dtype=dtype).reshape(obj[b'shape'])
    return obj


def load_calibration(toml_path):
    """Load calibration from TOML file"""
    logger.info(f"Loading calibration from {toml_path}")

    with open(toml_path, 'r') as f:
        data = toml.load(f)

    camera_matrix = np.array(data['calibration']['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(data['calibration']['dist_coeffs'], dtype=np.float64)
    img_size = tuple(data['camera']['resolution'])

    return camera_matrix, dist_coeffs, img_size


def load_sample_corners(msgpack_path, n_samples=5):
    """Load a few sample corner sets"""
    logger.info(f"Loading sample corners from {msgpack_path}")

    corners_list = []
    with open(msgpack_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for i, item in enumerate(unpacker):
            if i >= n_samples:
                break
            decoded = decode_numpy(item)
            corners_list.append(decoded)

    return corners_list


def undistort_corners(corners, camera_matrix, dist_coeffs, img_size=(1280, 800)):
    """Undistort corners using fisheye model"""
    # Get new camera matrix for undistortion
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, img_size, np.eye(3), balance=1
    )

    # Undistort corners
    corners_reshaped = corners.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.fisheye.undistortPoints(
        corners_reshaped, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    return undistorted.reshape(-1, 2)


def visualize_corners_comparison(corners_list, camera_matrix, dist_coeffs, img_size=(1280, 800)):
    """Create visualization comparing original and undistorted corners"""
    logger.info("Creating visualization...")

    n_samples = len(corners_list)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, corners in enumerate(corners_list):
        # Original corners
        ax_orig = axes[idx, 0]
        ax_orig.set_xlim(0, img_size[0])
        ax_orig.set_ylim(img_size[1], 0)
        ax_orig.set_aspect('equal')
        ax_orig.set_title(f'Sample {idx+1}: Original Corners')
        ax_orig.set_xlabel('X (pixels)')
        ax_orig.set_ylabel('Y (pixels)')
        ax_orig.grid(True, alpha=0.3)

        # Plot original corners
        pts_orig = corners.reshape(-1, 2)
        ax_orig.scatter(pts_orig[:, 0], pts_orig[:, 1], c='red', s=50, alpha=0.6, label='Original')

        # Draw lines connecting corners to show grid structure (6x4 grid)
        pts_grid = pts_orig.reshape(6, 4, 2)
        for i in range(6):
            ax_orig.plot(pts_grid[i, :, 0], pts_grid[i, :, 1], 'b-', alpha=0.3, linewidth=1)
        for j in range(4):
            ax_orig.plot(pts_grid[:, j, 0], pts_grid[:, j, 1], 'b-', alpha=0.3, linewidth=1)

        ax_orig.legend()

        # Undistorted corners
        ax_undist = axes[idx, 1]
        ax_undist.set_xlim(0, img_size[0])
        ax_undist.set_ylim(img_size[1], 0)
        ax_undist.set_aspect('equal')
        ax_undist.set_title(f'Sample {idx+1}: Undistorted Corners')
        ax_undist.set_xlabel('X (pixels)')
        ax_undist.set_ylabel('Y (pixels)')
        ax_undist.grid(True, alpha=0.3)

        # Undistort and plot
        pts_undist = undistort_corners(corners, camera_matrix, dist_coeffs, img_size)
        ax_undist.scatter(pts_undist[:, 0], pts_undist[:, 1], c='green', s=50, alpha=0.6, label='Undistorted')

        # Draw grid for undistorted corners
        pts_grid_undist = pts_undist.reshape(6, 4, 2)
        for i in range(6):
            ax_undist.plot(pts_grid_undist[i, :, 0], pts_grid_undist[i, :, 1], 'b-', alpha=0.3, linewidth=1)
        for j in range(4):
            ax_undist.plot(pts_grid_undist[:, j, 0], pts_grid_undist[:, j, 1], 'b-', alpha=0.3, linewidth=1)

        ax_undist.legend()

    plt.tight_layout()
    output_path = 'calibration_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def visualize_distortion_field(camera_matrix, dist_coeffs, img_size=(1280, 800)):
    """Create a visualization of the distortion field"""
    logger.info("Creating distortion field visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create a grid of points
    step = 50
    x = np.arange(0, img_size[0], step)
    y = np.arange(0, img_size[1], step)
    xx, yy = np.meshgrid(x, y)

    # Original points
    points_orig = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    # Undistort points
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, img_size, np.eye(3), balance=1
    )

    points_undist = cv2.fisheye.undistortPoints(
        points_orig.reshape(-1, 1, 2), camera_matrix, dist_coeffs, None, new_camera_matrix
    ).reshape(-1, 2)

    # Calculate displacement vectors
    displacement = points_undist - points_orig

    # Plot
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)
    ax.set_aspect('equal')
    ax.set_title('Distortion Correction Field\n(arrows show how pixels move during undistortion)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Plot displacement vectors
    scale = 5  # Scale factor for visibility
    for i in range(len(points_orig)):
        x0, y0 = points_orig[i]
        dx, dy = displacement[i] * scale
        ax.arrow(x0, y0, dx, dy, head_width=3, head_length=5, fc='blue', ec='blue', alpha=0.5, linewidth=0.5)

    # Add colorbar showing displacement magnitude
    mag = np.linalg.norm(displacement, axis=1).reshape(xx.shape)
    im = ax.contourf(xx, yy, mag, levels=20, alpha=0.3, cmap='hot')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Displacement (pixels)')

    ax.grid(True, alpha=0.3)

    output_path = 'distortion_field.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Distortion field saved to {output_path}")
    plt.close()


def print_calibration_summary(camera_matrix, dist_coeffs):
    """Print a summary of the calibration parameters"""
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("="*60)

    logger.info("\nCamera Matrix:")
    logger.info(f"  Focal Length X (fx): {camera_matrix[0, 0]:.2f} pixels")
    logger.info(f"  Focal Length Y (fy): {camera_matrix[1, 1]:.2f} pixels")
    logger.info(f"  Principal Point X (cx): {camera_matrix[0, 2]:.2f} pixels")
    logger.info(f"  Principal Point Y (cy): {camera_matrix[1, 2]:.2f} pixels")

    logger.info("\nDistortion Coefficients (Fisheye Model):")
    k = dist_coeffs.flatten()
    logger.info(f"  k1: {k[0]:.6f}")
    logger.info(f"  k2: {k[1]:.6f}")
    logger.info(f"  k3: {k[2]:.6f}")
    logger.info(f"  k4: {k[3]:.6f}")

    # Calculate approximate FOV
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    img_width = 1280
    img_height = 800

    fov_x = 2 * np.arctan(img_width / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(img_height / (2 * fy)) * 180 / np.pi

    logger.info("\nEstimated Field of View:")
    logger.info(f"  Horizontal FOV: {fov_x:.1f}°")
    logger.info(f"  Vertical FOV: {fov_y:.1f}°")

    logger.info("\n" + "="*60)


def main():
    """Main visualization pipeline"""

    # Paths
    toml_path = "calib_mono_1200_800_kmeans.toml"
    msgpack_path = "chessboard_20251113_153002.msgpack"

    # Load calibration
    camera_matrix, dist_coeffs, img_size = load_calibration(toml_path)

    # Print summary
    print_calibration_summary(camera_matrix, dist_coeffs)

    # Load sample corners
    corners_list = load_sample_corners(msgpack_path, n_samples=3)

    # Create visualizations
    visualize_corners_comparison(corners_list, camera_matrix, dist_coeffs, img_size)
    visualize_distortion_field(camera_matrix, dist_coeffs, img_size)

    logger.info("\nVisualization complete!")
    logger.info("Generated files:")
    logger.info("  - calibration_visualization.png")
    logger.info("  - distortion_field.png")


if __name__ == "__main__":
    main()
