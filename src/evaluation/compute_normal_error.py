# Compute the error mean and std of the two-cam method
# Compare the error among three HPR models

import os
import pickle
import numpy as np

vit_large_err = {'target1': [], 'target2': [], 'target4': []}
vit_base_err = {'target1': [], 'target2': [], 'target4': []}
open_pose_err = {'target1': [], 'target2': [], 'target4': []}

data_path = '../data'


def angle_difference(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if dot_product > 1:
        dot_product = 1
    angle = np.degrees(np.arccos(dot_product))
    return angle

for SUBJECT_NAME in os.listdir(data_path):
    subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        continue

    # frontal points: target 1 & target 2
    front_folder_path = os.path.join(subject_folder_path, 'front')
    with open(front_folder_path + '/ViTPose_large/final_target_normal_opt.pickle', 'rb') as f:
        vit_large_pred = pickle.load(f)
    with open(front_folder_path + '/ViTPose_base/final_target_normal_opt.pickle', 'rb') as f:
        vit_base_pred = pickle.load(f)
    with open(front_folder_path + '/OpenPose/final_target_normal_opt.pickle', 'rb') as f:
        open_pose_pred = pickle.load(f)
    with open(front_folder_path + '/two_cam_gt_normal.pickle', 'rb') as f:
        gt = pickle.load(f)

    vit_large_pred_tar1, vit_large_pred_tar2 = vit_large_pred['target1_normal'], vit_large_pred['target2_normal']
    vit_base_pred_tar1, vit_base_pred_tar2 = vit_base_pred['target1_normal'], vit_base_pred['target2_normal']
    open_pose_pred_tar1, open_pose_pred_tar2 = open_pose_pred['target1_normal'], open_pose_pred['target2_normal']
    gt_tar1, gt_tar2 = gt['target1_normal'], gt['target2_normal']

    side_folder_path = os.path.join(subject_folder_path, 'side')
    with open(side_folder_path + '/ViTPose_large/final_target_normal_opt.pickle', 'rb') as f:
        vit_large_pred = pickle.load(f)
    with open(side_folder_path + '/ViTPose_base/final_target_normal_opt.pickle', 'rb') as f:
        vit_base_pred = pickle.load(f)
    with open(side_folder_path + '/OpenPose/final_target_normal_opt.pickle', 'rb') as f:
        open_pose_pred = pickle.load(f)
    with open(side_folder_path + '/two_cam_gt_normal.pickle', 'rb') as f:
        gt = pickle.load(f)

    vit_large_pred_tar4 = vit_large_pred['target4_normal']
    vit_base_pred_tar4 = vit_base_pred['target4_normal']
    open_pose_pred_tar4 = open_pose_pred['target4_normal']
    gt_tar4 = gt['target4_normal']

    vit_large_pred_tar_list = [vit_large_pred_tar1, vit_large_pred_tar2, vit_large_pred_tar4]
    vit_base_pred_tar_list = [vit_base_pred_tar1, vit_base_pred_tar2, vit_base_pred_tar4]
    open_pose_pred_tar_list = [open_pose_pred_tar1, open_pose_pred_tar2, open_pose_pred_tar4]
    gt_tar_list = [gt_tar1, gt_tar2, gt_tar4]

    for target, vit_large_pred_tar, vit_base_pred_tar, open_pose_pred_tar, gt_tar in zip(vit_large_err.keys(),
                                                                                         vit_large_pred_tar_list,
                                                                                         vit_base_pred_tar_list,
                                                                                         open_pose_pred_tar_list,
                                                                                         gt_tar_list):
        vit_large_err[target].append(angle_difference(vit_large_pred_tar, gt_tar))
        vit_base_err[target].append(angle_difference(vit_base_pred_tar, gt_tar))

        # skip outlier for openpose target 4:
        if target == 'target4' and (SUBJECT_NAME == 'charles_xu' or SUBJECT_NAME == 'jingyu_wu'):
            continue
        open_pose_err[target].append(angle_difference(open_pose_pred_tar, gt_tar))

err_dict = {'vit_large_err': vit_large_err, 'vit_base_err': vit_base_err, 'open_pose_err': open_pose_err}
with open('normal_error_dict_opt.pickle', 'wb') as f:
    pickle.dump(err_dict, f)

# Compute difference mean and std
for target in vit_large_err.keys():
    print("========================================================================")
    for model_name, model_err in zip(err_dict.keys(), err_dict.values()):
        print(f"{target}, {model_name}, mean = {np.mean(model_err[target]): .1f}")
        print(f"{target}, {model_name}, std = {np.std(model_err[target]): .1f}")
