# Error Evaluation

### 2D Ground-Truth Data:
- Prepare the ground-truth pixel coordinates in a file `subject_2D_gt.xlsx`.
- Data should include subject name, three targets' pixel coordinates in both cameras.
- Data should be formatted as follows, for example:

| Name | target1_cam1 | target1_cam2 | target2_cam1 | target2_cam2 |target4_cam1 | target4_cam2 |
| :----:| :----: | :----: | :----: |  :----: |  :----: |  :----: |
| John Doe | (100, 100) | (100, 100) | (200, 200) |  (200, 200) |  (400, 400) |  (400, 400) |

### Compute Ground-Truth Pose:
- `compute_gt_position.py`: compute three sets of 3D ground-truth position with three methods:
  1. Method 1: Use camera 1's data only.
  2. Method 2: Use camera 2's data only.
  3. Method 3: Use both cameras' data. 
- `compute_gt_position_diff.py`: compute the euclidean distance between each pair of corresponding 3D ground-truth positions computed by the above three methods.
- `compute_gt_normal.py`: compute the ground-truth normal vectors based on the 3D ground-truth positions.

### Compute Error:
- `compute_target_normal.py`: compute the predicted target normal vector, based on the predicted target positions.
- `compute_position_error.py`: compute 3D position error, by the euclidean distance between the predicted position and the ground-truth position.
- `compute_normal_error.py`: compute 3D orientation error, by the angle difference between the predicted normal vector and the ground-truth normal vector.

### K-fold Evaluation:
- `k_fold_success_rate.py`: Compute success rate mean & std for all thresholds with K-fold validation.
- `k_fold_validation.py`: K-fold cross validation for the ratio model