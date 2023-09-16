import cv2
import numpy as np
from scipy.linalg import svd

# 3. **Estimate Motion**: Using the matched points, estimate the camera's motion between the two images. 
# This is typically done using techniques like the eight-point algorithm for fundamental matrix estimation, 
# which can be followed by a triangulation method to determine 3D points in space.

def estimate_essential_matrix(FundamentalMatrix, left_intrinsic_matrix, right_intrinsic_matrix):
    # since, we know (left_intrinsic_mat(transpose)*F*right_intrinsic_mat 
    return (left_intrinsic_matrix.T)@FundamentalMatrix@right_intrinsic_matrix

def decompose_essential_matrix(essential_matrix):
    # Singular Value Decomposition (SVD)
    U, S, Vt = svd(essential_matrix)

    # define W 
    # The Z matrix is a skew-symmetric matrix. A skew-symmetric matrix is a matrix that is the negative of its transpose.
    # The product UWVT gives one of the possible rotation matrices. 
    # This is because W represents a rotation of 90 degrees about the z-axis. 
    # When you multiply U and VT with W in between, 
    # you're effectively combining the rotations from the two camera views with this 90-degree rotation. 
    # This results in one of the possible rotation matrices that align the two camera views based on the epipolar geometry.
    W = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
    ])

    # two possible rotation matrices
    Rot_1 = U@W@Vt
    Rot_2 = U@W.T@Vt

    # Ensure the rotation matrices are proper rotations (det(R) = 1)
    if np.linalg.det(Rot_1) < 0:
        Rot_1 = -Rot_1
    if np.linalg.det(Rot_2) < 0:
        Rot_2 = -Rot_2

    # Two possible translation vectors
    tra_1 = U[:, 2].reshape(3,1)
    tra_2 = -U[:, 2].reshape(3,1)

    return [Rot_1, Rot_2], [tra_1, tra_2]

def get_3d_triangulated_points(K, rot, tra, pts1, pts2):
    Projection_1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    Projection_2 = K @ np.hstack([rot, tra])

    points3d = []
    for pt1, pt2 in zip(pts1, pts2):
        points_3D_homogeneous = cv2.triangulatePoints(Projection_1, Projection_2, pt1, pt2)
        # Convert from homogeneous to regular 3D coordinates
        points_3D = points_3D_homogeneous[:3] / points_3D_homogeneous[3]
        points3d.append(points_3D)
    return np.array(points3d), Projection_1, Projection_2



def validate_cheirality_condition(K, rotations_arr, translation_arr, pt1, pt2):
    for R, t in zip(rotations_arr, translation_arr):
        print("=======")
        # Triangulate a Point:
        # For each of the four possible (R, t) combinations, triangulate a 3D point using the matched 2D points from the two images.
        Projection_1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        print("R ", R.shape)
        print("t ", t.shape)
        Projection_2 = K @ np.hstack([R, t])

        # Triangulate using OpenCV
        X_homogeneous = cv2.triangulatePoints(Projection_1, Projection_2, pt1, pt2)
        X = X_homogeneous[:3] / X_homogeneous[3]

        print("Triangulated 3D point:", X)
        # Check if the point is in front of the first camera
        if X[2] <= 0:
            continue

        # Transform the point to the second camera's coordinate system
        X_transformed = R @ X + t

        # Check if the point is in front of the second camera
        if X_transformed[2] > 0:
            return R, t
        
    raise Exception("Couldnt find any valid projection - check your pt1 and pt2 points")


