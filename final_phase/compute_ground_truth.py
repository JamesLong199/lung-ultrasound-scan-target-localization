from subject_info import SUBJECT_NAME, SCAN_POSE
from triangulation import *
from utils.trajectory_io import *
import pickle

# TODO: manually modify the ground truth 2D coordinates. Need to use imshow.py to manually find the point.
GT_cam1 = np.array([325.0, 224.0])
GT_cam2 = np.array([282.8, 244.9])
target = 'target_4'  # 'target_1', 'target_2', 'target_4'

folder_path = 'ViTPose_UR_data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'

# read intrinsics
with open(folder_path + 'intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
    cam1_intr = pickle.load(f)
    print("cam1_intr: \n", cam1_intr)

with open(folder_path + 'intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
    cam2_intr = pickle.load(f)
    print("cam2_intr: \n", cam2_intr)

# read extrinsics
camera_poses = read_trajectory(folder_path + "odometry.log")
T_cam1_base = camera_poses[0].pose
T_base_cam1 = np.linalg.inv(T_cam1_base)
print("T_base_cam1: \n", T_base_cam1)
T_cam2_base = camera_poses[1].pose
T_base_cam2 = np.linalg.inv(T_cam2_base)
print("T_base_cam2: \n", T_base_cam2)

GT_3D = reconstruct(GT_cam1, GT_cam2, cam1_intr, cam2_intr, T_base_cam1, T_base_cam2)
GT_2D_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3,:] @ np.vstack([GT_3D,[1]]))
GT_2D_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3,:] @ np.vstack([GT_3D,[1]]))
print('GT_2D_cam1:\n', GT_2D_cam1)
print('GT_2D_cam2:\n', GT_2D_cam2)

print("GT_3D: \n", GT_3D)

with open(folder_path + 'ground_truth.pickle', 'rb') as f:
    GT_dict = pickle.load(f)

GT_dict[target] = GT_3D
print("GT_dict: ", GT_dict)

with open(folder_path + 'ground_truth.pickle', 'wb') as f:
    pickle.dump(GT_dict, f)

