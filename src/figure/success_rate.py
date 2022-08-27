# Plot the success rate curves
# It wasn't used in our project in the end, since the curves are too hard to see and analyze.

import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('../evaluation/error_dict.pickle', 'rb') as f:
    error_dict = pickle.load(f)

with open('../evaluation/error_dict_opt.pickle', 'rb') as f:
    error_dict_opt = pickle.load(f)

def success_rate(thres_list, value_list):
    # given a threshold list, count the number of success
    rate_list = []
    for thres in thres_list:
        count = 0
        for value in value_list:
            if value < thres:
                count += 1
        success_rate = (count / len(value_list)) * 100
        rate_list.append(success_rate)
    return rate_list


thres_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # error threshold in mm

# separate target
fig, ax = plt.subplots(1,3, figsize=(20, 5))
for i, (target, target_name) in enumerate(zip(['target1', 'target2', 'target4'],['Target1', 'Target2', 'Target4'] )):
    for pose_model, color in zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
                                                  ['red', 'blue', 'green']):
        error = error_dict[pose_model][target]
        error_opt = error_dict_opt[pose_model][target]
        error = [1000 * j for j in error]  # convert m to mm
        error_opt = [1000 * j for j in error_opt]  # convert m to mm
        rate_list = success_rate(thres_list, error)
        rate_list_opt = success_rate(thres_list, error_opt)
        ax[i].plot(thres_list, rate_list, color=color, linestyle='dotted')
        ax[i].plot(thres_list, rate_list_opt, color=color)

    ax[i].legend(['ViTPose_L', 'ViTPose_L*', 'ViTPose_B', 'ViTPose_B*', 'OpenPose', 'OpenPose*'], loc="lower right", prop={'size': 15})
    ax[i].set_title(target_name, fontsize=20)
    ax[i].set_xlabel('Error Threshold (mm)', fontsize=16)
    if i == 0:
        ax[i].set_ylabel('Success Rate %', fontsize=17)
    ax[i].set_ylim(2, 105)
    ax[i].tick_params(axis='x', labelsize=15)
    ax[i].tick_params(axis='y', labelsize=15)

# # separate HPE models
# fig, ax = plt.subplots(1,3, figsize=(20, 5))
# for i, (pose_model, pose_model_name) in enumerate(zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
#                                               ['ViTPose_L', 'ViTPose_B', 'OpenPose'])):
#
#     for target, color in zip(['target1', 'target2', 'target4'],
#                                                   ['red', 'blue', 'green']):
#         error = error_dict[pose_model][target]
#         error_opt = error_dict_opt[pose_model][target]
#         error = [1000 * j for j in error]  # convert m to mm
#         error_opt = [1000 * j for j in error_opt]  # convert m to mm
#         rate_list = success_rate(thres_list, error)
#         rate_list_opt = success_rate(thres_list, error_opt)
#         ax[i].plot(thres_list, rate_list, color=color, linestyle='dotted')
#         ax[i].plot(thres_list, rate_list_opt, color=color)
#
#     ax[i].legend(['Target1', 'Target1*', 'Target2', 'Target2*', 'Target4', 'Target4*'], loc="lower right", prop={'size': 15})
#     ax[i].set_title(pose_model_name, fontsize=20)
#     ax[i].set_xlabel('Error Threshold (mm)', fontsize=16)
#     ax[i].set_ylabel('Success Rate %', fontsize=16)
#     ax[i].set_ylim(0, 105)
#     ax[i].tick_params(axis='x', labelsize=15)
#     ax[i].tick_params(axis='y', labelsize=15)
#
# plt.savefig('success_rate.pdf', bbox_inches='tight')
plt.show()


