# Project Source Code

## Instructions 

1. Calibrate the two cameras (eye-to-hand) by `hand_eye_calib_apriltag.py`, save trajectory/extrinsics.
2. Label ground truth scan spots on the subject's body (target 1, target 2, target 4) with a mark pen.
3. Change the `SUBJECT_NAME` and `SCAN_POSE` in `subject_info.py` and run `python subject_info.py` to create a new folder for the new subject/pose.
   - `SCAN_POSE` is either 'front' or 'side'.
4. Have the subject lied on the stretcher.
5. `cd ..`, and run `python run_pipeline.py` to run the entire pipeline, for both front and side poses.
6. Manually measure and record the pixel coordinates of the three targets in two color images.
   - For target 1 and 2, use the images taken for the front pose.
   - For target 4, use the images taken for the side pose.

## Ratio Optimization

1. Run `python evaluation/compute_gt_position.py` to compute the ground-truth 3D positions:
2. Run `python optimize.py` to optimize the ratio parameters for both front and side postures.
