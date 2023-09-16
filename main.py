import cv2
import numpy as np
### Steps in SfM:
# 1. **Feature Extraction**: Identify points of interest in each image. This can be done using feature detection algorithms such as SIFT, SURF, or ORB.
# 2. **Feature Matching**: Find the same points of interest in multiple images. If two images have a set of points in common, it's possible they are observing the same part of a scene. 
from feature import feature_extraction_and_matching,demo_matched_image
# 3. **Estimate Motion**: Using the matched points, estimate the camera's motion between the two images. This is typically done using techniques like the eight-point algorithm for fundamental matrix estimation, which can be followed by a triangulation method to determine 3D points in space.
from pose import estimate_essential_matrix, decompose_essential_matrix, \
                validate_cheirality_condition, get_3d_triangulated_points
# 4. **Bundle Adjustment**: Refine the estimated camera poses and 3D point positions simultaneously. This step minimizes the reprojection error, which is the difference between the observed position of a point in an image and the projected position of the estimated 3D point using the camera pose.
from bundle_adjust import bundle_adjustments
# 5. **Dense Reconstruction**: After determining camera poses and sparse 3D points from the previous steps, you can proceed to reconstruct the scene densely. This involves understanding which parts of the images correspond to which 3D points (i.e., establishing a dense correspondence between multiple images).

# 6. **Meshing and Texturing**: Once you have a dense cloud of 3D points, you can create a mesh or surface from them. This is often followed by "draping" the images over this surface to produce a textured 3D model.


INTRINSIC_MATRIX = np.array([
                                [1733.74,  0,     792.27],
                                [0,      1733.74, 541.89],
                                [0,        0,     1]
                            ])

def load_image(filename):
    return cv2.imread(filename)

def draw_epilines(img1, img2, lines, pts1, pts2):
    """Draw epipolar lines and points."""
    r, c, _ = img2.shape  # Get the shape with channels
    for r, pt1, pt2 in zip(lines, pts1.astype(int), pts2.astype(int)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2



def main():
    DEBUG = False

    img_0_path = "data/artroom1/im0.png"
    img_1_path = "data/artroom1/im1.png"

    img0 = load_image(img_0_path)
    img1 = load_image(img_1_path)

    fundamental_mat, kp1, kp2, ransac_matches, src_pts, dst_pts = feature_extraction_and_matching(img0, img1)

    if DEBUG:
        demo_matched_image(img0, img1, kp1, kp2, ransac_matches)

    # estimate essential matrices
    essential_mat = estimate_essential_matrix(fundamental_mat, INTRINSIC_MATRIX, INTRINSIC_MATRIX)
    print("Essential Matrix " , essential_mat)

    # get possible rotation and translation matrices
    possible_rotations, possible_translations = decompose_essential_matrix(essential_matrix=essential_mat)
    print(possible_rotations[0].shape)

    # cheirality verification
    R,t = validate_cheirality_condition(INTRINSIC_MATRIX, possible_rotations, possible_translations, src_pts[0], dst_pts[0])
    print("Valid rotation and translation matrices \n", R, "\n\n", t)

    # get 3d points of all the matches -
    points_3d, P1, P2 = get_3d_triangulated_points(INTRINSIC_MATRIX, R, t, src_pts, dst_pts)
    print(len(points_3d))

    # bundle_adjustments
    # bundle_adjustments(R, t, INTRINSIC_MATRIX, points_3d, src_pts, dst_pts, P1, P2)


    import matplotlib.pyplot as plt

    # Rectify images
    valid, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(src_pts), np.float32(dst_pts), fundamental_mat, img1.shape[1::-1])
    if valid:
        print("Valid hurray~")

    img0_rectified = cv2.warpPerspective(img0, H1, img0.shape[1::-1])
    img1_rectified = cv2.warpPerspective(img1, H2, img1.shape[1::-1])


    # Draw epipolar lines
    lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, fundamental_mat).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, fundamental_mat).reshape(-1, 3)

    img0_epilines, img1_epilines = draw_epilines(img0_rectified, img1_rectified, lines1, src_pts, dst_pts)

    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img0_epilines)
    plt.subplot(1, 2, 2), plt.imshow(img1_epilines)
    plt.show()


if __name__ == '__main__':
    main()