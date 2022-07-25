import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

cam1_tar1_gt_list = []
cam1_tar2_gt_list = []
cam1_tar4_gt_list = []

cam2_tar1_gt_list = []
cam2_tar2_gt_list = []
cam2_tar4_gt_list = []

two_cam_tar1_gt_list = []
two_cam_tar2_gt_list = []
two_cam_tar4_gt_list = []

for SUBJECT_NAME in os.listdir('../data'):
    subject_folder_path = '../data/' + SUBJECT_NAME
    if os.path.isfile(subject_folder_path):
        continue

    # frontal points: target 1 & target 2
    front_folder_path = subject_folder_path + '/front'
    with open(front_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(front_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(front_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam_1_tar1_gt, cam_1_tar2_gt = cam_1_gt['target_1'].flatten(), cam_1_gt['target2_3d'].flatten()
    cam_2_tar1_gt, cam_2_tar2_gt = cam_2_gt['target_1'].flatten(), cam_2_gt['target2_3d'].flatten()
    two_cam_tar1_gt, two_cam_tar2_gt = two_cam_gt['target_1'].flatten(), two_cam_gt['target2_3d'].flatten()

    cam1_tar1_gt_list.append(cam_1_tar1_gt)
    cam1_tar2_gt_list.append(cam_1_tar2_gt)
    cam2_tar1_gt_list.append(cam_2_tar1_gt)
    cam2_tar2_gt_list.append(cam_2_tar2_gt)
    two_cam_tar1_gt_list.append(two_cam_tar1_gt)
    two_cam_tar2_gt_list.append(two_cam_tar2_gt)

    # side point: target 4
    side_folder_path = subject_folder_path + '/side'
    with open(side_folder_path + '/cam_1_gt.pickle', 'rb') as f:
        cam_1_gt = pickle.load(f)
    with open(side_folder_path + '/cam_2_gt.pickle', 'rb') as f:
        cam_2_gt = pickle.load(f)
    with open(side_folder_path + '/two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)

    cam_1_tar4_gt = cam_1_gt['target_4'].flatten()
    cam_2_tar4_gt = cam_2_gt['target_4'].flatten()
    two_cam_tar4_gt = two_cam_gt['target_4'].flatten()

    cam1_tar4_gt_list.append(cam_1_tar4_gt)
    cam2_tar4_gt_list.append(cam_2_tar4_gt)
    two_cam_tar4_gt_list.append(two_cam_tar4_gt)


# separate target
BIG_INTERVAL = 4.5
SMALL_INTERVAL = 1

# # Separate targets
# fig, ax = plt.subplots(1,3, figsize=(20, 5))
# for i, (target, target_name, cam1_tar_list, cam2_tar_list, two_cam_tar_list) in enumerate(zip(['target1', 'target2', 'target4'],
#                                                         ['Target1', 'Target2', 'Target4'],
#                                                         [cam1_tar1_gt_list, cam1_tar2_gt_list, cam1_tar4_gt_list],
#                                                         [cam2_tar1_gt_list, cam2_tar2_gt_list, cam2_tar4_gt_list],
#                                                         [two_cam_tar1_gt_list, two_cam_tar2_gt_list, two_cam_tar4_gt_list] )):
#     bp_list = []
#     for j, axis in enumerate(['X', 'Y', 'Z']):
#         for w, (method, method_name, edge_color, fill_color) in enumerate(zip([cam1_tar_list, two_cam_tar_list, cam2_tar_list, ],
#                                                                               ['Cam1 Only', 'Two Cams', 'Cam2 Only'],
#                                                                               ['blue', 'red', 'green'],
#                                                                               ['cyan', 'yellow', 'orange']                                                                              )):
#             axis_value = [gt[j] for gt in method]  # all gt values of a particular axis
#
#             bp = ax[i].boxplot([axis_value], positions=[(j + 1) * BIG_INTERVAL + w * SMALL_INTERVAL], patch_artist=True,
#                              widths=0.3, whis=99)
#
#             for patch in bp['boxes']:
#                 patch.set(facecolor=fill_color)
#                 patch.set(edgecolor=edge_color)
#             for patch in bp['caps']:
#                 patch.set(color=edge_color)
#             for patch in bp['whiskers']:
#                 patch.set(color=edge_color)
#             for patch in bp['medians']:
#                 patch.set(color=edge_color)
#             if j == 0:
#                 bp_list.append(bp)
#
#     # ax[i].legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0], bp_list[2]["boxes"][0]],
#     #            ['Cam1 Only', 'Cam2 Only', 'Two Cams'], loc='upper left', prop={'size': 15}, bbox_to_anchor=(1,0.5))
#
#     ax[i].set_xticks([(a + 1) * BIG_INTERVAL + SMALL_INTERVAL for a in range(3)], ['X', 'Y', 'Z'], fontsize=14)
#     ax[i].tick_params(axis='y', labelsize=12)
#     ax[i].set_ylabel('Ground-truth (mm)', fontsize=15)
#     ax[i].set_title(target_name, fontsize=20)

# Separate XYZ
fig, ax = plt.subplots(1, 3, figsize=(25, 6.5))
for i, axis_name in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
    bp_list = []
    for j, (cam1_tar_list, cam2_tar_list, two_cam_tar_list) in enumerate(zip([cam1_tar1_gt_list,
                                                                              cam1_tar2_gt_list,
                                                                              cam1_tar4_gt_list],
                                                                             [cam2_tar1_gt_list,
                                                                              cam2_tar2_gt_list,
                                                                              cam2_tar4_gt_list],
                                                                             [two_cam_tar1_gt_list,
                                                                              two_cam_tar2_gt_list,
                                                                              two_cam_tar4_gt_list])):
        for w, (method, method_name, edge_color, fill_color) in enumerate(zip([cam1_tar_list, two_cam_tar_list, cam2_tar_list, ],
                                                                              ['Cam1 Only', 'Two Cams', 'Cam2 Only'],
                                                                              ['blue', 'red', 'green'],
                                                                              ['cyan', 'yellow', 'orange']                                                                              )):
            axis_value = [gt[i] * 1000 for gt in method]  # all gt values of a particular axis, convert m to mm

            bp = ax[i].boxplot([axis_value], positions=[(j + 1) * BIG_INTERVAL + w * SMALL_INTERVAL], patch_artist=True,
                             widths=0.5, whis=99)

            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)
                patch.set(edgecolor=edge_color)
            for patch in bp['caps']:
                patch.set(color=edge_color)
            for patch in bp['whiskers']:
                patch.set(color=edge_color)
            for patch in bp['medians']:
                patch.set(color=edge_color)
            if j == 0:
                bp_list.append(bp)

    if i == 1:
        ax[i].legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0], bp_list[2]["boxes"][0]],
                   ['Cam1 Only', 'Two Cams', 'Cam2 Only'], loc='lower center', prop={'size': 19}, bbox_to_anchor=(0.5, -0.23), ncol=3)

    ax[i].set_xticks([(a + 1) * BIG_INTERVAL + SMALL_INTERVAL for a in range(3)], ['Target1', 'Target2', 'Target4'], fontsize=19)
    ax[i].tick_params(axis='y', labelsize=15)
    if i == 0:
        ax[i].set_ylabel('Ground-truth Position (mm)', fontsize=19)
    ax[i].set_title(axis_name, fontsize=23)

plt.savefig('gt_distribution.pdf', bbox_inches='tight')
plt.show()