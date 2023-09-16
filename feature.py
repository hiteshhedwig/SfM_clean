import cv2
import numpy as np

# 1. **Feature Extraction**: Identify points of interest in each image. This can be done using feature detection algorithms such as SIFT, SURF, or ORB.
# 2. **Feature Matching**: Find the same points of interest in multiple images. If two images have a set of points in common, it's possible they are observing the same part of a scene. 

def feature_extraction_and_matching(img0, img1):
    # Create a SIFT object
    sift = cv2.SIFT_create()
    # Detect and compute SIFT features for both images
    kp1, des1 = sift.detectAndCompute(img0, None)
    kp2, des2 = sift.detectAndCompute(img1, None)

    # Perform brute-force matching with ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # For each pair of matches (m, n) obtained:
    # m is the best match and n is the second-best match.
    good_matches = []
    for m, n in matches:
        # The ratio test checks the quality of the matches.
        # If the distance of the best match (m.distance) is significantly smaller than that of the second-best match (n.distance),
        # then the match is considered to be "good".
        # In this case, if m's distance is less than 75% of n's distance, it's added to the good_matches list.
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract the coordinates of matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Use RANSAC to identify inliers
    # The fundamental matrix encapsulates the epipolar geometry between two views and is a fundamental concept in stereo vision and structure from motion.
    fundamental_mat, inliers = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    if inliers is None:
        exit(0)

    # Filter matches using the inliers
    ransac_matches = [good_matches[i] for i, val in enumerate(inliers) if val == 1]
    return fundamental_mat, kp1, kp2, ransac_matches, src_pts, dst_pts

def demo_matched_image(img0, img1, kp1, kp2, ransac_matches):
    img_matches = cv2.drawMatches(img0, kp1, img1, kp2, ransac_matches, None)
    img_matches = cv2.resize(img_matches, (1920,1080))
    cv2.imshow("Matches " , img_matches)
    cv2.waitKey(0)