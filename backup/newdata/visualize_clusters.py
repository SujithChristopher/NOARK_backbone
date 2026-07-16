"""
Visualize k-means clustering of chessboard frames
Shows how frames are distributed across clusters and which samples were selected
"""

import numpy as np
import msgpack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def perform_kmeans_clustering(corners_array, n_clusters=200, random_state=42):
    """
    Perform k-means clustering and return cluster assignments and selected samples
    """
    logger.info(f"Performing k-means clustering with {n_clusters} clusters")

    # Compute features
    features = compute_corner_features(corners_array)

    # Normalize features for better clustering
    features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Apply k-means clustering
    logger.info("Running k-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, verbose=0)
    kmeans.fit(features_normalized)

    # For each cluster, select the frame closest to cluster center
    selected_indices = []
    for cluster_id in range(n_clusters):
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

    logger.info(f"Selected {len(selected_indices)} frames from {n_clusters} clusters")

    return features_normalized, kmeans.labels_, selected_indices, kmeans.cluster_centers_


def visualize_clusters_2d(features_normalized, labels, selected_indices, cluster_centers, n_clusters=200):
    """
    Visualize clusters in 2D using PCA
    """
    logger.info("Creating 2D cluster visualization using PCA")

    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_normalized)
    centers_2d = pca.transform(cluster_centers)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: All frames colored by cluster
    ax1 = axes[0]
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab20', alpha=0.5, s=20, edgecolors='none')
    ax1.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2,
               label='Cluster Centers', zorder=5)
    ax1.set_title(f'All {len(features_2d)} Frames Colored by Cluster\n(PCA 2D projection)', fontsize=12)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Selected frames highlighted
    ax2 = axes[1]
    # Plot all frames in light gray
    ax2.scatter(features_2d[:, 0], features_2d[:, 1],
               c='lightgray', alpha=0.3, s=20, edgecolors='none', label='Not selected')

    # Highlight selected frames
    selected_features_2d = features_2d[selected_indices]
    scatter2 = ax2.scatter(selected_features_2d[:, 0], selected_features_2d[:, 1],
                          c=labels[selected_indices], cmap='tab20',
                          s=100, edgecolors='black', linewidths=1.5,
                          label='Selected samples', zorder=3)

    # Plot cluster centers
    ax2.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2,
               label='Cluster Centers', zorder=5)

    ax2.set_title(f'{len(selected_indices)} Selected Samples (1 per cluster)\n(PCA 2D projection)', fontsize=12)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'kmeans_clusters_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"2D cluster visualization saved to {output_path}")
    plt.close()


def visualize_cluster_statistics(labels, n_clusters=200):
    """
    Visualize cluster size distribution
    """
    logger.info("Creating cluster statistics visualization")

    # Count frames per cluster
    cluster_counts = np.bincount(labels, minlength=n_clusters)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Histogram of cluster sizes
    ax1 = axes[0]
    ax1.hist(cluster_counts, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(cluster_counts.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {cluster_counts.mean():.1f} frames/cluster')
    ax1.axvline(np.median(cluster_counts), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(cluster_counts):.1f} frames/cluster')
    ax1.set_xlabel('Number of frames per cluster')
    ax1.set_ylabel('Number of clusters')
    ax1.set_title('Distribution of Cluster Sizes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart of each cluster size
    ax2 = axes[1]
    cluster_ids = np.arange(n_clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.tile(colors, (n_clusters // 20 + 1, 1))[:n_clusters]

    ax2.bar(cluster_ids, cluster_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of frames in cluster')
    ax2.set_title(f'Frames per Cluster (Total: {len(labels)} frames, {n_clusters} clusters)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = 'cluster_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Cluster statistics saved to {output_path}")
    plt.close()


def visualize_selected_frames_spatial(corners_array, selected_indices, img_size=(1280, 800)):
    """
    Show spatial distribution of selected frames in image space
    """
    logger.info("Creating spatial distribution visualization")

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Plot all frames in light gray
    for corners in corners_array:
        pts = corners.reshape(-1, 2)
        mean_x = pts[:, 0].mean()
        mean_y = pts[:, 1].mean()
        ax.scatter(mean_x, mean_y, c='lightgray', s=10, alpha=0.3)

    # Highlight selected frames
    for idx in selected_indices:
        corners = corners_array[idx]
        pts = corners.reshape(-1, 2)
        mean_x = pts[:, 0].mean()
        mean_y = pts[:, 1].mean()
        ax.scatter(mean_x, mean_y, c='red', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)

    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)
    ax.set_aspect('equal')
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f'Spatial Distribution of Chessboard Centers\nGray: All {len(corners_array)} frames | Red: {len(selected_indices)} selected samples')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'spatial_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Spatial distribution saved to {output_path}")
    plt.close()


def print_clustering_summary(labels, selected_indices, n_clusters):
    """Print summary statistics about the clustering"""
    cluster_counts = np.bincount(labels, minlength=n_clusters)

    logger.info("\n" + "="*60)
    logger.info("K-MEANS CLUSTERING SUMMARY")
    logger.info("="*60)
    logger.info(f"\nTotal frames: {len(labels)}")
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Selected samples: {len(selected_indices)}")

    logger.info(f"\nCluster size statistics:")
    logger.info(f"  Mean frames per cluster: {cluster_counts.mean():.1f}")
    logger.info(f"  Median frames per cluster: {np.median(cluster_counts):.1f}")
    logger.info(f"  Min frames in a cluster: {cluster_counts.min()}")
    logger.info(f"  Max frames in a cluster: {cluster_counts.max()}")
    logger.info(f"  Std dev: {cluster_counts.std():.1f}")

    logger.info("\n" + "="*60)


def main():
    """Main visualization pipeline"""

    # Configuration
    msgpack_path = "chessboard_20251113_153002.msgpack"
    n_clusters = 200
    random_state = 42  # Use same seed as first calibration

    # Load corners
    corners_array = load_chessboard_corners(msgpack_path)

    # Perform k-means clustering
    features_normalized, labels, selected_indices, cluster_centers = perform_kmeans_clustering(
        corners_array, n_clusters=n_clusters, random_state=random_state
    )

    # Print summary
    print_clustering_summary(labels, selected_indices, n_clusters)

    # Create visualizations
    visualize_clusters_2d(features_normalized, labels, selected_indices, cluster_centers, n_clusters)
    visualize_cluster_statistics(labels, n_clusters)
    visualize_selected_frames_spatial(corners_array, selected_indices)

    logger.info("\nVisualization complete!")
    logger.info("Generated files:")
    logger.info("  - kmeans_clusters_2d.png")
    logger.info("  - cluster_statistics.png")
    logger.info("  - spatial_distribution.png")


if __name__ == "__main__":
    main()
