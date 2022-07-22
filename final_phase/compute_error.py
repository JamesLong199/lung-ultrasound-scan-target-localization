# Compute the error mean and std of the two-cam method
# Compare the error among three HPR models

import os
import pickle
import numpy as np

vit_large_err = {'target1': [], 'target2': [], 'target4': []}
vit_base_err = {'target1': [], 'target2': [], 'target4': []}
open_pose_err = {'target1': [], 'target2': [], 'target4': []}

for SUBJECT_NAME in os.listdir('data'):
    subject_folder_path = os.path.join('data', SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        continue

    # frontal points: target 1 & target 2
    front_folder_path = os.path.join(subject_folder_path, 'front')
    with open(front_folder_path + '/ViTPose_large/final_target.pickle', 'rb') as f:
        vit_large_pred = pickle.load(f)
    with open(front_folder_path + '/ViTPose_base/final_target.pickle', 'rb') as f:
        vit_base_pred = pickle.load(f)
    with open(front_folder_path + '/OpenPose/final_target.pickle', 'rb') as f:
        open_pose_pred = pickle.load(f)
    with open(front_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        gt = pickle.load(f)

    vit_large_pred_tar1, vit_large_pred_tar2 = vit_large_pred[0].flatten(), vit_large_pred[1].flatten()
    vit_base_pred_tar1, vit_base_pred_tar2 = vit_base_pred[0].flatten(), vit_base_pred[1].flatten()
    open_pose_pred_tar1, open_pose_pred_tar2 = open_pose_pred[0].flatten(), open_pose_pred[1].flatten()
    gt_tar1, gt_tar2 = gt['target_1'].flatten(), gt['target2_3d'].flatten()

    side_folder_path = os.path.join(subject_folder_path, 'side')
    with open(side_folder_path + '/ViTPose_large/final_target.pickle', 'rb') as f:
        vit_large_pred = pickle.load(f)
    with open(side_folder_path + '/ViTPose_base/final_target.pickle', 'rb') as f:
        vit_base_pred = pickle.load(f)
    with open(side_folder_path + '/OpenPose/final_target.pickle', 'rb') as f:
        open_pose_pred = pickle.load(f)
    with open(side_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        gt = pickle.load(f)

    vit_large_pred_tar4 = vit_large_pred[0].flatten()
    vit_base_pred_tar4 = vit_base_pred[0].flatten()
    open_pose_pred_tar4 = open_pose_pred[0].flatten()
    gt_tar4 = gt['target_4'].flatten()

    vit_large_pred_tar_list = [vit_large_pred_tar1, vit_large_pred_tar2, vit_large_pred_tar4]
    vit_base_pred_tar_list = [vit_base_pred_tar1, vit_base_pred_tar2, vit_base_pred_tar4]
    open_pose_pred_tar_list = [open_pose_pred_tar1, open_pose_pred_tar2, open_pose_pred_tar4]
    gt_tar_list = [gt_tar1, gt_tar2, gt_tar4]

    for target, vit_large_pred_tar, vit_base_pred_tar, open_pose_pred_tar, gt_tar in zip(vit_large_err.keys(),
                                                                                         vit_large_pred_tar_list,
                                                                                         vit_base_pred_tar_list,
                                                                                         open_pose_pred_tar_list,
                                                                                         gt_tar_list):
        vit_large_err[target].append(np.linalg.norm(vit_large_pred_tar - gt_tar))
        vit_base_err[target].append(np.linalg.norm(vit_base_pred_tar - gt_tar))
        open_pose_err[target].append(np.linalg.norm(open_pose_pred_tar - gt_tar))

err_dict = {'vit_large_err': vit_large_err, 'vit_base_err': vit_base_err, 'open_pose_err': open_pose_err}
with open('error_dict.pickle', 'wb') as f:
    pickle.dump(err_dict, f)

# Compute difference mean and std
for target in vit_large_err.keys():
    print("========================================================================")
    for model_name, model_err in zip(err_dict.keys(), err_dict.values()):
        print(f"{target}, {model_name}, mean = {np.mean(model_err[target]) * 1000: .1f}")
        print(f"{target}, {model_name}, std = {np.std(model_err[target]) * 1000: .1f}")
