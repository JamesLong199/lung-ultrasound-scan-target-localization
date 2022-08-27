# Compute the ground-truth difference between two different methods:
# It wasn't used in our project in the end.

import os
import pickle
import numpy as np

# three methods:
cam1_cam2_diff = {'target1': {'x': [], 'y': [], 'z': []}, 'target2': {'x': [], 'y': [], 'z': []},
                  'target4': {'x': [], 'y': [], 'z': []}}
two_cam_cam1_diff = {'target1': {'x': [], 'y': [], 'z': []}, 'target2': {'x': [], 'y': [], 'z': []},
                     'target4': {'x': [], 'y': [], 'z': []}}
two_cam_cam2_diff = {'target1': {'x': [], 'y': [], 'z': []}, 'target2': {'x': [], 'y': [], 'z': []},
                     'target4': {'x': [], 'y': [], 'z': []}}

data_path = '../data'

# loop through the data folder
for SUBJECT_NAME in os.listdir(data_path):
    subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        continue

    # frontal points: target 1 & target 2
    front_folder_path = os.path.join(subject_folder_path, 'front')
    with open(front_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(front_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(front_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam_1_tar1_gt, cam_1_tar2_gt = cam_1_gt['target_1'].flatten(), cam_1_gt['target_2'].flatten()
    cam_2_tar1_gt, cam_2_tar2_gt = cam_2_gt['target_1'].flatten(), cam_2_gt['target_2'].flatten()
    two_cam_tar1_gt, two_cam_tar2_gt = two_cam_gt['target_1'].flatten(), two_cam_gt['target_2'].flatten()

    # side point: target 4
    side_folder_path = os.path.join(subject_folder_path, 'side')
    with open(side_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(side_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(side_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam_1_tar4_gt = cam_1_gt['target_4'].flatten()
    cam_2_tar4_gt = cam_2_gt['target_4'].flatten()
    two_cam_tar4_gt = two_cam_gt['target_4'].flatten()

    for target, cam1_tar, cam2_tar, two_cam_tar in zip(cam1_cam2_diff.keys(),
                                                       [cam_1_tar1_gt, cam_1_tar2_gt, cam_1_tar4_gt],
                                                       [cam_2_tar1_gt, cam_2_tar2_gt, cam_2_tar4_gt],
                                                       [two_cam_tar1_gt, two_cam_tar2_gt, two_cam_tar4_gt]):
        for i, axis in enumerate(cam1_cam2_diff[target].keys()):
            cam1_cam2_diff[target][axis].append(cam1_tar[i] - cam2_tar[i])
            two_cam_cam1_diff[target][axis].append(two_cam_tar[i] - cam1_tar[i])
            two_cam_cam2_diff[target][axis].append(two_cam_tar[i] - cam2_tar[i])

gt_diff_dict = {'cam1_cam2_diff': cam1_cam2_diff, 'two_cam_cam1_diff': two_cam_cam1_diff, 'two_cam_cam2_diff': two_cam_cam2_diff}
with open('gt_diff.pickle', 'wb') as f:
    pickle.dump(gt_diff_dict, f)

# Compute difference mean and std
for target in cam1_cam2_diff.keys():
    print("========================================================================")
    for gt_diff_name, gt_diff in zip(gt_diff_dict.keys(), gt_diff_dict.values()):
        for axis in cam1_cam2_diff[target].keys():
            print(f"{target}, {gt_diff_name}, {axis}-axis: mean = {np.mean(gt_diff[target][axis]) * 1000: .1f}")
            print(f"{target}, {gt_diff_name}, {axis}-axis: std = {np.std(gt_diff[target][axis]) * 1000: .1f}")


