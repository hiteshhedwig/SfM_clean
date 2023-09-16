import cv2
import numpy as np
from scipy.optimize import least_squares

def project(points_3D, P):
    """Project 3D points into 2D using a camera matrix P."""
    points_3D_homogeneous = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    points_2D_homogeneous = points_3D_homogeneous @ P.T
    return points_2D_homogeneous[:, :2] / points_2D_homogeneous[:, 2:]


def reprojection_error(params, x1_obs, x2_obs, P1, P2):
    """Compute the reprojection error."""
    num_points = x1_obs.shape[0]
    X = np.array(params).reshape((num_points, 3))
    
    x1_proj = project(X, P1)
    x2_proj = project(X, P2)
    
    error_x1 = x1_obs - x1_proj
    error_x2 = x2_obs - x2_proj
    
    return np.hstack([error_x1.ravel(), error_x2.ravel()])

def bundle_adjustments(Rotation_mat, translation_mat, Intrinsic_mat, X_initial, x1_observed, x2_observed, P1, P2):

    res = least_squares(reprojection_error, X_initial.ravel(), args=(x1_observed, x2_observed, P1, P2))

    # Reshape the optimized result to get the refined 3D points
    X_optimized = res.x.reshape((-1, 3))
    print(X_optimized)