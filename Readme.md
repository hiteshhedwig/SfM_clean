# SfM - 

An attempt to write clean and readable code for SfM pipeline.

## Steps in SfM:

1. **Feature Extraction**: Identify points of interest in each image. This can be done using feature detection algorithms such as SIFT, SURF, or ORB.

2. **Feature Matching**: Find the same points of interest in multiple images. If two images have a set of points in common, it's possible they are observing the same part of a scene. 

3. **Estimate Motion**: Using the matched points, estimate the camera's motion between the two images. This is typically done using techniques like the eight-point algorithm for fundamental matrix estimation, which can be followed by a triangulation method to determine 3D points in space.

4. **Bundle Adjustment**: Refine the estimated camera poses and 3D point positions simultaneously. This step minimizes the reprojection error, which is the difference between the observed position of a point in an image and the projected position of the estimated 3D point using the camera pose.

5. **Dense Reconstruction**: After determining camera poses and sparse 3D points from the previous steps, you can proceed to reconstruct the scene densely. This involves understanding which parts of the images correspond to which 3D points (i.e., establishing a dense correspondence between multiple images).

6. **Meshing and Texturing**: Once you have a dense cloud of 3D points, you can create a mesh or surface from them. This is often followed by "draping" the images over this surface to produce a textured 3D model.