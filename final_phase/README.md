# Instructions

1. Calibrate the two cameras (eye-to-hand) by `hand_eye__calib_apriltag.py`, save trajectory/extrinsics.
2. Label ground truth scan spots on the subject's body (target 1, target 2, target 4) with a mark pen.
3. Change the `SUBJECT_NAME` and `SCAN_POSE` in `subject_info.py` and run `python subject_info.py` to create a new folder for the new subject/pose.
   - `SCAN_POSE` is either 'front' or 'side'.
4. Run `python run_pipeline.py` to run the entire pipeline, for both front and side poses.
5. Run `imshow.py` and manually measure and record the pixel coordinates of the three targets in two color images.
   - For target 1 and 2, use the images taken for the front pose.
   - For target 4, use the images taken for the side pose.
6. In `compute_ground_truth.py`, manually input the corresponding pixel coordinates and target name, and run the script. Repeat for all three targets.

# Ratio Regression

1. Run `optimize.py` to compute the optimal parameter values for both front and side models.
