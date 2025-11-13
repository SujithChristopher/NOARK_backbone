"""
Calibration Manager - Handles camera calibration data storage and retrieval
Stores calibration in ~/Documents/NOARK/calibration/calibration.json
"""

import json
import os
import platform
import numpy as np
from pathlib import Path


class CalibrationManager:
    def __init__(self):
        # Determine base path based on platform
        if platform.system() == "Linux":
            self.base_path = Path.home() / "Documents" / "NOARK" / "calibration"
        else:
            self.base_path = Path.home() / "Documents" / "NOARK" / "calibration"

        self.calib_file = self.base_path / "calibration.json"

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def load_calibration(self, camera_name="default"):
        """
        Load calibration data from JSON file

        Args:
            camera_name: Name of the camera configuration to load

        Returns:
            dict: Calibration data with 'camera_matrix' and 'dist_coeffs'

        Raises:
            FileNotFoundError: If calibration file doesn't exist
            KeyError: If camera_name not found in calibration file
        """
        if not self.calib_file.exists():
            raise FileNotFoundError(
                f"Calibration file not found at: {self.calib_file}\n"
                f"Please create calibration data using create_default_calibration() "
                f"or run camera calibration."
            )

        with open(self.calib_file, 'r') as f:
            all_calibrations = json.load(f)

        if camera_name not in all_calibrations:
            available = list(all_calibrations.keys())
            raise KeyError(
                f"Camera '{camera_name}' not found in calibration file.\n"
                f"Available cameras: {available}"
            )

        calib_data = all_calibrations[camera_name]

        # Convert lists back to numpy arrays
        return {
            'camera_matrix': np.array(calib_data['camera_matrix']),
            'dist_coeffs': np.array(calib_data['dist_coeffs'])
        }

    def save_calibration(self, camera_matrix, dist_coeffs, camera_name="default",
                        resolution=None, notes=None):
        """
        Save calibration data to JSON file

        Args:
            camera_matrix: 3x3 numpy array
            dist_coeffs: 1x5 or 1x4 numpy array of distortion coefficients
            camera_name: Name for this camera configuration
            resolution: Optional tuple (width, height)
            notes: Optional notes about this calibration
        """
        # Load existing calibrations or create new dict
        if self.calib_file.exists():
            with open(self.calib_file, 'r') as f:
                all_calibrations = json.load(f)
        else:
            all_calibrations = {}

        # Convert numpy arrays to lists for JSON serialization
        calib_data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'resolution': resolution,
            'notes': notes,
            'created_date': self._get_timestamp()
        }

        all_calibrations[camera_name] = calib_data

        # Save to file
        with open(self.calib_file, 'w') as f:
            json.dump(all_calibrations, f, indent=2)

        print(f"Calibration saved to: {self.calib_file}")
        print(f"Camera name: {camera_name}")

    def list_calibrations(self):
        """
        List all available calibrations in the file

        Returns:
            list: List of camera names
        """
        if not self.calib_file.exists():
            return []

        with open(self.calib_file, 'r') as f:
            all_calibrations = json.load(f)

        return list(all_calibrations.keys())

    def delete_calibration(self, camera_name):
        """Delete a specific calibration"""
        if not self.calib_file.exists():
            print("No calibration file exists")
            return

        with open(self.calib_file, 'r') as f:
            all_calibrations = json.load(f)

        if camera_name in all_calibrations:
            del all_calibrations[camera_name]

            with open(self.calib_file, 'w') as f:
                json.dump(all_calibrations, f, indent=2)

            print(f"Deleted calibration: {camera_name}")
        else:
            print(f"Calibration '{camera_name}' not found")

    def create_default_calibration(self):
        """
        Create default calibration data from existing TOML files
        This is a one-time migration helper
        """
        import toml

        # Try to find existing calibration files
        project_root = Path(__file__).parent

        calibrations_to_migrate = []

        # Look for webcam calibration
        webcam_calib = project_root / "old_calibration" / "webcam_calib.toml"
        if webcam_calib.exists():
            calibrations_to_migrate.append(("webcam", webcam_calib))

        # Look for RPi calibration
        rpi_calib = project_root / "old_calibration" / "calib_mono_faith3D.toml"
        if rpi_calib.exists():
            calibrations_to_migrate.append(("rpi_faith3D", rpi_calib))

        # Look for other common calibrations
        mono_calib = project_root / "calib_mono_1200_800.toml"
        if mono_calib.exists():
            calibrations_to_migrate.append(("mono_1200_800", mono_calib))

        if not calibrations_to_migrate:
            # Create a basic default if no files found
            print("No existing calibration files found. Creating placeholder calibration.")
            print("WARNING: This is NOT a real calibration! Please calibrate your camera.")

            # Very basic placeholder (NOT ACCURATE - user must calibrate!)
            camera_matrix = np.array([
                [671.25, 0.0, 678.00],
                [0.0, 692.23, 443.37],
                [0.0, 0.0, 1.0]
            ])
            dist_coeffs = np.array([[-0.0426, 0.0039, 0.0057, -0.0003, -0.0034]])

            self.save_calibration(
                camera_matrix,
                dist_coeffs,
                camera_name="default",
                resolution=(1280, 720),
                notes="PLACEHOLDER - Please calibrate your camera!"
            )
        else:
            # Migrate existing calibrations
            print(f"Found {len(calibrations_to_migrate)} calibration files to migrate:")

            for name, toml_path in calibrations_to_migrate:
                try:
                    calib_data = toml.load(toml_path)
                    camera_matrix = np.array(
                        calib_data["calibration"]["camera_matrix"]
                    ).reshape(3, 3)
                    dist_coeffs = np.array(calib_data["calibration"]["dist_coeffs"])

                    # Get resolution if available
                    resolution = None
                    if "camera" in calib_data and "resolution" in calib_data["camera"]:
                        resolution = calib_data["camera"]["resolution"]

                    self.save_calibration(
                        camera_matrix,
                        dist_coeffs,
                        camera_name=name,
                        resolution=resolution,
                        notes=f"Migrated from {toml_path.name}"
                    )
                    print(f"  ✓ Migrated: {name}")
                except Exception as e:
                    print(f"  ✗ Failed to migrate {name}: {e}")

        print(f"\nCalibration file created at: {self.calib_file}")
        print(f"Available calibrations: {self.list_calibrations()}")

    def get_calibration_info(self, camera_name="default"):
        """
        Get metadata about a calibration

        Returns:
            dict: Calibration metadata
        """
        if not self.calib_file.exists():
            return None

        with open(self.calib_file, 'r') as f:
            all_calibrations = json.load(f)

        if camera_name not in all_calibrations:
            return None

        calib = all_calibrations[camera_name]

        return {
            'camera_name': camera_name,
            'resolution': calib.get('resolution'),
            'notes': calib.get('notes'),
            'created_date': calib.get('created_date'),
            'camera_matrix_shape': np.array(calib['camera_matrix']).shape,
            'dist_coeffs_shape': np.array(calib['dist_coeffs']).shape
        }

    def _get_timestamp(self):
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """CLI interface for calibration manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Manage camera calibrations")
    parser.add_argument('action', choices=['list', 'create', 'info', 'delete'],
                       help="Action to perform")
    parser.add_argument('--name', default='default',
                       help="Camera name (for info/delete actions)")

    args = parser.parse_args()

    manager = CalibrationManager()

    if args.action == 'list':
        calibrations = manager.list_calibrations()
        if calibrations:
            print("Available calibrations:")
            for name in calibrations:
                info = manager.get_calibration_info(name)
                print(f"  - {name}")
                print(f"    Resolution: {info['resolution']}")
                print(f"    Created: {info['created_date']}")
                print(f"    Notes: {info['notes']}")
        else:
            print("No calibrations found. Run 'create' to initialize.")

    elif args.action == 'create':
        manager.create_default_calibration()

    elif args.action == 'info':
        info = manager.get_calibration_info(args.name)
        if info:
            print(f"Calibration: {info['camera_name']}")
            print(f"Resolution: {info['resolution']}")
            print(f"Created: {info['created_date']}")
            print(f"Notes: {info['notes']}")
            print(f"Camera matrix: {info['camera_matrix_shape']}")
            print(f"Distortion coeffs: {info['dist_coeffs_shape']}")
        else:
            print(f"Calibration '{args.name}' not found")

    elif args.action == 'delete':
        manager.delete_calibration(args.name)


if __name__ == "__main__":
    main()
