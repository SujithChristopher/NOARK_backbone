import numpy as np


# from numba import njit


def calculate_rotmat(xdir, zdir, org):
    """
    this function calculates rotation matrix
    """
    v1 = xdir - org  # v1
    v2 = zdir - org  # v2

    vxnorm = v1 / np.linalg.norm(v1)

    vzcap = v2 - (vxnorm.T @ v2) * vxnorm
    vznorm = vzcap / np.linalg.norm(vzcap)

    vynorm = np.cross(vznorm.T[0], vxnorm.T[0]).reshape(3, 1)
    rotMat = np.hstack((vxnorm, vynorm, vznorm))
    return rotMat


# @njit
def calculate_rotmat_from_xyo(xdir, ydir, org):
    """
    this function calculates rotation matrix
    """
    v1 = xdir - org  # v1
    v2 = ydir - org  # v2

    vxnorm = v1 / np.linalg.norm(v1)

    vycap = v2 - np.dot(vxnorm.T, v2) * vxnorm
    vynorm = vycap / np.linalg.norm(vycap)

    vznorm = np.cross(vynorm.T[0], vxnorm.T[0]).reshape(3, 1)
    rotMat = np.hstack((vxnorm, vynorm, vznorm))
    return rotMat


def rotmat_from_x_z(x_vectors, z_vectors):
    """Right-handed basis from an x edge and an approximate z edge.

    Same convention as :func:`calculate_rotmat` -- columns are ``[x, y, z]``,
    the x edge is kept exact, the z edge is orthogonalized against it, and
    ``y = z x x`` -- but it takes the edge vectors directly instead of three
    points, and is vectorized. Accepts ``(3,)`` or ``(N, 3)`` and returns
    ``(3, 3)`` or ``(N, 3, 3)``.
    """
    x_vectors = np.atleast_2d(np.asarray(x_vectors, dtype=np.float64))
    z_vectors = np.atleast_2d(np.asarray(z_vectors, dtype=np.float64))
    single = x_vectors.shape[0] == 1 and np.ndim(x_vectors) == 2

    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return vectors / norms

    x_axis = normalize(x_vectors)
    z_projected = z_vectors - np.sum(z_vectors * x_axis, axis=1, keepdims=True) * x_axis
    z_axis = normalize(z_projected)
    y_axis = np.cross(z_axis, x_axis)
    matrices = np.stack([x_axis, y_axis, z_axis], axis=2)
    return matrices[0] if single else matrices
