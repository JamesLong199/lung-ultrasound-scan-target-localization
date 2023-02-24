# Compute the ground-truth error of three different methods (cam1 only vs cam2 only cs two cams):

import pandas as pd
import os
import pickle
import numpy as np
from utils.triangulation import *
from utils.trajectory_io import *

excel_data_df = pd.read_excel("subject_2D_gt.xlsx")
data = excel_data_df.to_numpy()

# three methods:
cam1_gt_error = {'target1': [], 'target2': [], 'target4': []}
cam2_gt_error = {'target1': [], 'target2': [], 'target4': []}
two_cam_gt_error_cam1 = {'target1': [], 'target2': [], 'target4': []}
two_cam_gt_error_cam2 = {'target1': [], 'target2': [], 'target4': []}

data_path = '../data'

for subject in data:
    SUBJECT_NAME = subject[0]
    # print("subject name: ", SUBJECT_NAME)

    # Convert String to Tuple using map() + tuple() + int + split()
    cam1_tar1_gt_2d_ori = np.array(tuple(map(float, subject[1][1:-2].split(', '))))
    cam2_tar1_gt_2d_ori = np.array(tuple(map(float, subject[2][1:-2].split(', '))))
    cam1_tar2_gt_2d_ori = np.array(tuple(map(float, subject[3][1:-2].split(', '))))
    cam2_tar2_gt_2d_ori = np.array(tuple(map(float, subject[4][1:-2].split(', '))))
    cam1_tar4_gt_2d_ori = np.array(tuple(map(float, subject[5][1:-2].split(', '))))
    cam2_tar4_gt_2d_ori = np.array(tuple(map(float, subject[6][1:-2].split(', '))))

    subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        print("No such subject!!")
        continue

    # frontal points: target 1 & target 2
    front_folder_path = os.path.join(subject_folder_path, 'front')
    with open(front_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(front_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(front_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam1_tar1_gt_3d, cam1_tar2_gt_3d = cam_1_gt['target_1'], cam_1_gt['target_2']
    cam2_tar1_gt_3d, cam2_tar2_gt_3d = cam_2_gt['target_1'], cam_2_gt['target_2']
    two_cam_tar1_gt_3d, two_cam_tar2_gt_3d = two_cam_gt['target_1'], two_cam_gt['target_2']

    # read RGB intrinsics
    with open(front_folder_path + '/intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
        cam1_intr = pickle.load(f)

    with open(front_folder_path + '/intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
        cam2_intr = pickle.load(f)

    # read extrinsics
    camera_poses_front = read_trajectory(front_folder_path + "/odometry.log")
    T_base_cam1_front = np.linalg.inv(camera_poses_front[0].pose)
    T_base_cam2_front = np.linalg.inv(camera_poses_front[1].pose)

    # side point: target 4
    side_folder_path = os.path.join(subject_folder_path, 'side')
    with open(side_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(side_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(side_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam1_tar4_gt_3d = cam_1_gt['target_4']
    cam2_tar4_gt_3d = cam_2_gt['target_4']
    two_cam_tar4_gt_3d = two_cam_gt['target_4']

    # read extrinsics
    camera_poses_side = read_trajectory(side_folder_path + "/odometry.log")
    T_base_cam1_side = np.linalg.inv(camera_poses_side[0].pose)
    T_base_cam2_side = np.linalg.inv(camera_poses_side[1].pose)

    # project 3D ground truth back to 2D

    # cam1
    cam1_tar1_gt_2d_pred = from_homog(cam1_intr @ T_base_cam1_front[0:3, :] @ to_homog(cam1_tar1_gt_3d)).squeeze()
    cam1_tar2_gt_2d_pred = from_homog(cam1_intr @ T_base_cam1_front[0:3, :] @ to_homog(cam1_tar2_gt_3d)).squeeze()
    cam1_tar4_gt_2d_pred = from_homog(cam1_intr @ T_base_cam1_side[0:3, :] @ to_homog(cam1_tar4_gt_3d)).squeeze()

    cam1_gt_error['target1'].append(np.linalg.norm(cam1_tar1_gt_2d_ori - cam1_tar1_gt_2d_pred))
    cam1_gt_error['target2'].append(np.linalg.norm(cam1_tar2_gt_2d_ori - cam1_tar2_gt_2d_pred))
    cam1_gt_error['target4'].append(np.linalg.norm(cam1_tar4_gt_2d_ori - cam1_tar4_gt_2d_pred))

    # cam2
    cam2_tar1_gt_2d_pred = from_homog(cam2_intr @ T_base_cam2_front[0:3, :] @ to_homog(cam2_tar1_gt_3d)).squeeze()
    cam2_tar2_gt_2d_pred = from_homog(cam2_intr @ T_base_cam2_front[0:3, :] @ to_homog(cam2_tar2_gt_3d)).squeeze()
    cam2_tar4_gt_2d_pred = from_homog(cam2_intr @ T_base_cam2_side[0:3, :] @ to_homog(cam2_tar4_gt_3d)).squeeze()

    cam2_gt_error['target1'].append(np.linalg.norm(cam2_tar1_gt_2d_ori - cam2_tar1_gt_2d_pred))
    cam2_gt_error['target2'].append(np.linalg.norm(cam2_tar2_gt_2d_ori - cam2_tar2_gt_2d_pred))
    cam2_gt_error['target4'].append(np.linalg.norm(cam2_tar4_gt_2d_ori - cam2_tar4_gt_2d_pred))

    # two cam
    # -- method 1: average error from 2 views
    # -- method 2: don't average, compare with one cam separately
    two_cam_tar1_gt_2d_cam1_pred = from_homog(
        cam1_intr @ T_base_cam1_front[0:3, :] @ to_homog(two_cam_tar1_gt_3d)).squeeze()
    two_cam_tar2_gt_2d_cam1_pred = from_homog(
        cam1_intr @ T_base_cam1_front[0:3, :] @ to_homog(two_cam_tar2_gt_3d)).squeeze()
    two_cam_tar4_gt_2d_cam1_pred = from_homog(
        cam1_intr @ T_base_cam1_side[0:3, :] @ to_homog(two_cam_tar4_gt_3d)).squeeze()

    two_cam_gt_error_cam1['target1'].append(np.linalg.norm(cam1_tar1_gt_2d_ori - two_cam_tar1_gt_2d_cam1_pred))
    two_cam_gt_error_cam1['target2'].append(np.linalg.norm(cam1_tar2_gt_2d_ori - two_cam_tar2_gt_2d_cam1_pred))
    two_cam_gt_error_cam1['target4'].append(np.linalg.norm(cam1_tar4_gt_2d_ori - two_cam_tar4_gt_2d_cam1_pred))

    two_cam_tar1_gt_2d_cam2_pred = from_homog(
        cam2_intr @ T_base_cam2_front[0:3, :] @ to_homog(two_cam_tar1_gt_3d)).squeeze()
    two_cam_tar2_gt_2d_cam2_pred = from_homog(
        cam2_intr @ T_base_cam2_front[0:3, :] @ to_homog(two_cam_tar2_gt_3d)).squeeze()
    two_cam_tar4_gt_2d_cam2_pred = from_homog(
        cam2_intr @ T_base_cam2_side[0:3, :] @ to_homog(two_cam_tar4_gt_3d)).squeeze()

    two_cam_gt_error_cam2['target1'].append(np.linalg.norm(cam2_tar1_gt_2d_ori - two_cam_tar1_gt_2d_cam2_pred))
    two_cam_gt_error_cam2['target2'].append(np.linalg.norm(cam2_tar2_gt_2d_ori - two_cam_tar2_gt_2d_cam2_pred))
    two_cam_gt_error_cam2['target4'].append(np.linalg.norm(cam2_tar4_gt_2d_ori - two_cam_tar4_gt_2d_cam2_pred))

gt_error_dict = {'cam1_gt_error': cam1_gt_error, 'cam2_gt_error': cam2_gt_error,
                 'two_cam_gt_error_cam1': two_cam_gt_error_cam1, 'two_cam_gt_error_cam2': two_cam_gt_error_cam2}
with open('gt_error_dict.pickle', 'wb') as f:
    pickle.dump(gt_error_dict, f)
