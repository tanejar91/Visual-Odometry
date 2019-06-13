## Visual-Odometry
This repository contians the code and results to estimate an autonomous vehicle trajectory by images taken with a monocular camera 
set up on the vehicle. Data is taken from the Coursera course 'Visual Perception for Self-Driving Cars'.

## Procedure
1. Generate keypoints and descriptors for every image in the dataset. ORB is used for feature detection and descriptor generation.
Code for SURF is also provided(commented), as SURF is not available for python 3.6 in Windows.
2. Match features between subsequent pairs of images. Use Lowe's ratio test to filter the mateches.
3. Estimate camera motion from subsequent images. Camera motion here means rotation and translation of the camera. Function 
estimate_motion_EssentialMat is used for this task. This task is achieved by Essential Matrix Decomposition. Essential matrix can be 
found using "cv2.findEssentialMat()"; this function takes matched keypoints in subsequent pair of images and camera matrix as arguments.
4. Estimate trajectory using Inverse Transformation 
