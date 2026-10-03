import numpy as np
import cv2
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from typing import Tuple, List

class CameraOptimizer:
    def __init__(self, image_width: int = 1200, image_height: int = 800):
        """Initialize camera optimizer for specific image dimensions."""
        self.width = image_width
        self.height = image_height
        
        # Your existing calibration parameters
        self.current_K = np.array([
            [583.74894761, 0, 655.62169261],
            [0, 587.45257362, 334.63280683],
            [0, 0, 1]
        ])
        
        self.current_dist = np.array([
            [-0.0454573],
            [0.04474313],
            [-0.06101144],
            [0.0247249]
        ]).flatten()
        
    def generate_grid_points(self, grid_size: int = 20) -> np.ndarray:
        """Generate a uniform grid of points across the image."""
        x = np.linspace(0, self.width, grid_size)
        y = np.linspace(0, self.height, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        return points.astype(np.float32)
    
    def analyze_distortion_pattern(self, undistorted_points: np.ndarray, 
                                  distorted_points: np.ndarray) -> dict:
        """Analyze the distortion pattern from the plot data."""
        # Calculate displacement vectors
        displacement = distorted_points - undistorted_points
        
        # Calculate radial distances from image center
        center = np.array([self.width/2, self.height/2])
        radial_dist = np.linalg.norm(undistorted_points - center, axis=1)
        
        # Analyze radial distortion characteristics
        radial_displacement = np.linalg.norm(displacement, axis=1)
        
        # Fit polynomial to radial distortion
        coeffs = np.polyfit(radial_dist, radial_displacement, 4)
        
        return {
            'displacement': displacement,
            'radial_dist': radial_dist,
            'radial_displacement': radial_displacement,
            'polynomial_coeffs': coeffs,
            'max_distortion': np.max(radial_displacement),
            'mean_distortion': np.mean(radial_displacement)
        }
    
    def optimize_uniform_scaling(self) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize for uniform scaling to minimize distortion effects."""
        
        # Generate test points
        test_points = self.generate_grid_points(30)
        
        def objective(params):
            """Objective function to minimize distortion non-uniformity."""
            fx, fy, cx, cy, k1, k2, p1, p2 = params
            
            # Construct camera matrix
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # Distortion coefficients
            dist = np.array([k1, k2, p1, p2, 0])
            
            # Apply distortion
            points_3d = cv2.convertPointsToHomogeneous(test_points)
            points_3d = points_3d.reshape(-1, 1, 3)
            
            # Project points
            projected, _ = cv2.projectPoints(
                points_3d, 
                np.zeros(3), 
                np.zeros(3), 
                K, 
                dist
            )
            projected = projected.reshape(-1, 2)
            
            # Calculate uniformity metric
            center = np.array([cx, cy])
            
            # Original distances from center
            orig_dist = np.linalg.norm(test_points - center, axis=1)
            
            # Projected distances from center
            proj_dist = np.linalg.norm(projected - center, axis=1)
            
            # We want uniform scaling, so minimize variance in scale factors
            scale_factors = proj_dist / (orig_dist + 1e-10)
            
            # Penalize non-uniform scaling
            uniformity_loss = np.var(scale_factors)
            
            # Also penalize points going outside image bounds
            out_of_bounds = np.sum(
                (projected[:, 0] < 0) | (projected[:, 0] > self.width) |
                (projected[:, 1] < 0) | (projected[:, 1] > self.height)
            )
            
            # Penalize deviation from expected FOV coverage
            coverage_loss = np.abs(np.mean(scale_factors) - 1.0)
            
            return uniformity_loss + 0.1 * out_of_bounds + 0.5 * coverage_loss
        
        # Initial parameters from current calibration
        fx_init = self.current_K[0, 0]
        fy_init = self.current_K[1, 1]
        
        # Start with centered principal point for better uniformity
        cx_init = self.width / 2
        cy_init = self.height / 2
        
        initial_params = [
            fx_init, fy_init, cx_init, cy_init,
            self.current_dist[0], self.current_dist[1],
            self.current_dist[2], self.current_dist[3]
        ]
        
        # Bounds for optimization
        bounds = [
            (400, 800),  # fx
            (400, 800),  # fy
            (self.width * 0.4, self.width * 0.6),   # cx
            (self.height * 0.4, self.height * 0.6),  # cy
            (-0.5, 0.5),  # k1
            (-0.5, 0.5),  # k2
            (-0.1, 0.1),  # p1
            (-0.1, 0.1),  # p2
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Extract optimized parameters
        fx, fy, cx, cy, k1, k2, p1, p2 = result.x
        
        optimized_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        optimized_dist = np.array([k1, k2, p1, p2, 0])
        
        return optimized_K, optimized_dist
    
    def optimize_from_distortion_pattern(self, 
                                        enforce_equal_focal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize camera parameters to achieve more uniform distortion."""
        
        def create_uniform_params():
            """Create parameters that encourage uniform scaling."""
            
            # For 160-degree FOV, we need appropriate focal length
            # Using FOV formula: FOV = 2 * arctan(sensor_size / (2 * focal_length))
            # Rearranging: focal_length = sensor_size / (2 * tan(FOV/2))
            
            fov_rad = np.radians(160)
            
            # Estimate focal length for uniform coverage
            # Assuming the image diagonal should cover the FOV
            diagonal = np.sqrt(self.width**2 + self.height**2)
            focal_estimate = diagonal / (2 * np.tan(fov_rad / 2))
            
            if enforce_equal_focal:
                # Use same focal length for both axes
                fx = fy = focal_estimate * 0.7  # Adjust factor based on sensor
            else:
                # Allow slight difference but keep them close
                fx = focal_estimate * 0.7
                fy = focal_estimate * 0.72
            
            # Center the principal point for better uniformity
            cx = self.width / 2
            cy = self.height / 2
            
            # For wide-angle lens, we expect negative k1 (barrel distortion)
            # Start with moderate values
            k1 = -0.1
            k2 = 0.05
            p1 = 0.0  # Start with no tangential distortion
            p2 = 0.0
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            dist = np.array([k1, k2, p1, p2, 0])
            
            return K, dist
        
        # Generate initial uniform parameters
        uniform_K, uniform_dist = create_uniform_params()
        
        # Fine-tune using optimization
        optimized_K, optimized_dist = self.optimize_uniform_scaling()
        
        return optimized_K, optimized_dist
    
    def visualize_comparison(self, new_K: np.ndarray, new_dist: np.ndarray):
        """Visualize the distortion patterns for comparison."""
        
        # Generate grid points
        grid_points = self.generate_grid_points(15)
        
        # Apply current distortion
        current_distorted = cv2.undistortPoints(
            grid_points.reshape(-1, 1, 2),
            self.current_K,
            self.current_dist,
            P=self.current_K
        ).reshape(-1, 2)
        
        # Apply new distortion
        new_distorted = cv2.undistortPoints(
            grid_points.reshape(-1, 1, 2),
            new_K,
            new_dist,
            P=new_K
        ).reshape(-1, 2)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original distortion
        axes[0].scatter(grid_points[:, 0], grid_points[:, 1], c='blue', marker='o', s=20, label='Original')
        axes[0].scatter(current_distorted[:, 0], current_distorted[:, 1], c='red', marker='x', s=20, label='Current Distorted')
        axes[0].set_title('Current Calibration')
        axes[0].set_xlim(0, self.width)
        axes[0].set_ylim(0, self.height)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].invert_yaxis()
        
        # New distortion
        axes[1].scatter(grid_points[:, 0], grid_points[:, 1], c='blue', marker='o', s=20, label='Original')
        axes[1].scatter(new_distorted[:, 0], new_distorted[:, 1], c='green', marker='x', s=20, label='Optimized Distorted')
        axes[1].set_title('Optimized Calibration')
        axes[1].set_xlim(0, self.width)
        axes[1].set_ylim(0, self.height)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()
        
        # Displacement comparison
        current_displacement = np.linalg.norm(current_distorted - grid_points, axis=1)
        new_displacement = np.linalg.norm(new_distorted - grid_points, axis=1)
        
        x_pos = np.arange(len(current_displacement))
        axes[2].bar(x_pos - 0.2, current_displacement, 0.4, label='Current', alpha=0.7)
        axes[2].bar(x_pos + 0.2, new_displacement, 0.4, label='Optimized', alpha=0.7)
        axes[2].set_title('Displacement Magnitude Comparison')
        axes[2].set_xlabel('Point Index')
        axes[2].set_ylabel('Displacement (pixels)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nDistortion Statistics:")
        print(f"Current - Mean displacement: {np.mean(current_displacement):.2f} pixels")
        print(f"Current - Std displacement: {np.std(current_displacement):.2f} pixels")
        print(f"Optimized - Mean displacement: {np.mean(new_displacement):.2f} pixels")
        print(f"Optimized - Std displacement: {np.std(new_displacement):.2f} pixels")
    
    def run_optimization(self):
        """Run the complete optimization process."""
        
        print("Starting camera matrix optimization...")
        print(f"Image size: {self.width}x{self.height}")
        print(f"Target: 160-degree FOV with uniform distortion")
        
        print("\nCurrent Camera Matrix:")
        print(self.current_K)
        print("\nCurrent Distortion Coefficients:")
        print(self.current_dist)
        
        # Optimize for uniform scaling
        print("\n1. Optimizing for uniform scaling...")
        uniform_K, uniform_dist = self.optimize_uniform_scaling()
        
        print("\nOptimized Camera Matrix (Uniform Scaling):")
        print(uniform_K)
        print("\nOptimized Distortion Coefficients:")
        print(uniform_dist)
        
        # Alternative: Create from distortion pattern
        print("\n2. Creating parameters from distortion pattern...")
        pattern_K, pattern_dist = self.optimize_from_distortion_pattern()
        
        print("\nPattern-based Camera Matrix:")
        print(pattern_K)
        print("\nPattern-based Distortion Coefficients:")
        print(pattern_dist)
        
        # Visualize comparison
        print("\n3. Visualizing results...")
        self.visualize_comparison(uniform_K, uniform_dist)
        
        return uniform_K, uniform_dist, pattern_K, pattern_dist


# Main execution
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = CameraOptimizer(image_width=1200, image_height=800)
    
    # Run optimization
    uniform_K, uniform_dist, pattern_K, pattern_dist = optimizer.run_optimization()
    
    # Save results
    print("\n" + "="*50)
    print("FINAL RECOMMENDED PARAMETERS")
    print("="*50)
    print("\nOption 1 - Optimized Uniform Scaling:")
    print("Camera Matrix:")
    print(uniform_K)
    print("\nDistortion Coefficients:")
    print(uniform_dist)
    
    print("\nOption 2 - Pattern-based Parameters:")
    print("Camera Matrix:")
    print(pattern_K)
    print("\nDistortion Coefficients:")
    print(pattern_dist)
    
    # Save to file
    np.savez('optimized_camera_params.npz',
             uniform_K=uniform_K,
             uniform_dist=uniform_dist,
             pattern_K=pattern_K,
             pattern_dist=pattern_dist,
             original_K=optimizer.current_K,
             original_dist=optimizer.current_dist)
    
    print("\nParameters saved to 'optimized_camera_params.npz'")