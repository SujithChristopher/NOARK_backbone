import numpy as np
from scipy.spatial.transform import Rotation as R


class ExponentialMovingAverageFilter3D:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema_x = None
        self.ema_y = None
        self.ema_z = None

    def update(self, ema):
        if self.ema_x is None:
            self.ema_x = ema[0]
            self.ema_y = ema[1]
            self.ema_z = ema[2]
        else:
            self.ema_x = self.alpha * ema[0] + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * ema[1] + (1 - self.alpha) * self.ema_y
            self.ema_z = self.alpha * ema[2] + (1 - self.alpha) * self.ema_z
        return np.array([self.ema_x, self.ema_y, self.ema_z])


class CoordinateTransform:
    def __init__(self, offsets):
        self.offsets = offsets
        self.rotation_matrix = None
        self.translation_vector = None
        self.transformed = None

    def compute_transformed_coordinates(self, coordinate):
        self.rotation_matrix = R.from_rotvec(
            np.array([coordinate["rx"], coordinate["ry"], coordinate["rz"]]).T
        ).as_matrix()
        self.translation_vector = np.array(
            [coordinate["x"], coordinate["y"], coordinate["z"]]
        )
        self.transformed = np.array(
            [
                (_r @ self.offsets.reshape(3, 1) + _t.reshape(3, 1)).T[0]
                for _r, _t in zip(self.rotation_matrix, self.translation_vector.T)
            ]
        )
        return self.rotation_matrix, self.translation_vector, self.transformed

    def transform_to_world_coordinates(self):
        _r_88_inv = self.rotation_matrix[0].T
        _tvec_88_0 = self.translation_vector.T[0]
        world_coordinates = {}
        for key in [12, 14, 20, 88, 89]:
            world_coordinates[key] = _r_88_inv @ (self.transformed - self.offsets).T
        return world_coordinates


# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # Define offsets
# offsets = {
#     12: np.array([-0.054, 0.031, -0.069]),
#     14: np.array([0.00, 0.1025, -0.069]),
#     20: np.array([0.00, 0.01, -0.069]),
#     88: np.array([0.00, 0.031, -0.1075]),
#     89: np.array([0.054, 0.031, -0.069])
# }

# # Function to compute transformed coordinates
# def compute_transformed_coordinates(coordinate, id_offset):
#     rotation_matrix = R.from_rotvec(np.array([coordinate['rx'], coordinate['ry'], coordinate['rz']]).T).as_matrix()
#     translation_vector = np.array([coordinate['x'], coordinate['y'], coordinate['z']])
#     transformed = np.array([(_r @ id_offset.reshape(3,1) + _t.reshape(3,1)).T[0] for _r, _t in zip(rotation_matrix, translation_vector.T)])
#     return rotation_matrix, translation_vector, transformed

# # Compute transformations
# results = {}
# for key in [12, 14, 20, 88, 89]:
#     results[key] = compute_transformed_coordinates(coordinate[str(key)], offsets[key])

# # Transform to world coordinates
# _r_88_inv = results[88][0][0].T
# _tvec_88_0 = results[88][1].T[0]

# world_coordinates = {}
# for key in [12, 14, 20, 88, 89]:
#     world_coordinates[key] = _r_88_inv @ (results[key][2] - _tvec_88_0).T

# _gt_12, _gt_14, _gt_20, _gt_88, _gt_89 = [world_coordinates[key] for key in [12, 14, 20, 88, 89]]
