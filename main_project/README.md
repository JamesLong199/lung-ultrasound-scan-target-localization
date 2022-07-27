# Project Source Code

## Instructions

1. Calibrate the two cameras (eye-to-hand) by `hand_eye__calib_apriltag.py`, save trajectory/extrinsics.
2. Label ground truth scan spots on the subject's body (target 1, target 2, target 4) with a mark pen.
3. Change the `SUBJECT_NAME` and `SCAN_POSE` in `subject_info.py` and run `python subject_info.py` to create a new folder for the new subject/pose.
   - `SCAN_POSE` is either 'front' or 'side'.
4. `cd ..`, and run `python run_pipeline.py` to run the entire pipeline, for both front and side poses.
5. Manually measure and record the pixel coordinates of the three targets in two color images.
   - For target 1 and 2, use the images taken for the front pose.
   - For target 4, use the images taken for the side pose.

## Ratio Optimization

1. Run `compute_gt_position.py` and `compute_gt_normal` to compute the ground-truth 6D pose with three methods:
   1. Method 1: Use Camera 1 only.
   2. Method 2: Use Camera 2 only.
   3. Method 3: Use both Cameras.
2. Run `optimize.py` to optimize the ratio parameters for both front and side postures.

## Note:
- Make sure to download ViTPose models in advance from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose).