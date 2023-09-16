import cv2
import numpy as np

# 3. **Estimate Motion**: Using the matched points, estimate the camera's motion between the two images. 
# This is typically done using techniques like the eight-point algorithm for fundamental matrix estimation, 
# which can be followed by a triangulation method to determine 3D points in space.

def estimate_essential_matrix(FundamentalMatrix, left_intrinsic_matrix, right_intrinsic_matrix):
    # since, we know (left_intrinsic_mat(transpose)*F*right_intrinsic_mat 
    return (left_intrinsic_matrix.T)@FundamentalMatrix@right_intrinsic_matrix