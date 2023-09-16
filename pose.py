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

    # two possible translation vectors
    tra_1 = U[:2]
    tra_2 = (-1)*U[:2]

    return [Rot_1, Rot_2], [tra_1,tra_2]


