import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('../evaluation/error_dict_opt.pickle', 'rb') as f:
    error_dict_opt = pickle.load(f)

# # mean and std
# BIG_INTERVAL = 10
# SMALL_INTERVAL = 1
#
# plt.rcParams["figure.figsize"] = [10, 5]
# # plt.rcParams["figure.autolayout"] = True
# for i, (pose_model, pose_model_name) in enumerate(zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
#                                                       ['ViTPose_large', 'ViTPose_base', 'OpenPose'])):
#
#     for j, (target, color) in enumerate(zip(['target1', 'target2', 'target4'], ['red', 'blue', 'green'])):
#         error_opt = error_dict_opt[pose_model][target]
#         error_opt = [1000 * z for z in error_opt]  # convert m to mm
#         error_opt_mean = np.mean(error_opt)
#         error_opt_std = np.std(error_opt)
#
#         plt.errorbar((i+1)*BIG_INTERVAL+j*SMALL_INTERVAL, error_opt_mean, yerr=error_opt_std, color=color, fmt='o', capsize=3)
#
# plt.legend(['target1', 'target2', 'target4'], loc='upper left', prop={'size': 13})
# plt.xticks([(i + 1) * BIG_INTERVAL + SMALL_INTERVAL for i in range(3)], ['ViTPose_large', 'ViTPose_base', 'OpenPose'], fontsize=10)
# plt.tick_params(axis='y', labelsize=10)
# plt.ylim(0, 40)
# plt.xlabel('HPE model', fontsize=10)
# plt.ylabel('Mean with std (mm)', fontsize=10)
# plt.title('Error mean and std', fontsize=10)
#
# plt.savefig('error_mean_std.pdf', bbox_inches='tight')
# plt.show()


# # x-axis is HPE model
# BIG_INTERVAL = 5
# SMALL_INTERVAL = 0.8
#
# plt.rcParams["figure.figsize"] = [10, 5]
# bp_list = []
# for i, (pose_model, pose_model_name) in enumerate(zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
#                                                       ['ViTPose_large', 'ViTPose_base', 'OpenPose'])):
#
#     for j, (target, edge_color, fill_color) in enumerate(zip(['target1', 'target2', 'target4'],
#                                                              ['blue', 'red', 'green'],
#                                                              ['cyan', 'yellow', 'orange'])):
#         error_opt = error_dict_opt[pose_model][target]
#         error_opt = [1000 * z for z in error_opt]  # convert m to mm
#
#         bp = plt.boxplot([error_opt], positions=[(i+1)*BIG_INTERVAL+j*SMALL_INTERVAL], patch_artist=True, widths=0.3, whis=99)
#
#         for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#             plt.setp(bp[element], color=edge_color)
#
#         for patch in bp['boxes']:
#             patch.set(facecolor=fill_color)
#
#         if i == 0:
#             bp_list.append(bp)
#
#
# plt.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0], bp_list[2]["boxes"][0]],
#            ['target1', 'target2', 'target4'], loc='upper left', prop={'size': 15}, bbox_to_anchor=(1.02, 1), borderaxespad=0)
#
# plt.xticks([(i + 1) * BIG_INTERVAL + SMALL_INTERVAL for i in range(3)], ['ViTPose_large', 'ViTPose_base', 'OpenPose'], fontsize=14)
# plt.tick_params(axis='y', labelsize=12)
#
# # plt.savefig('error_distribution.pdf', bbox_inches='tight')
# plt.show()

# x-axis is HPE model
BIG_INTERVAL = 5
SMALL_INTERVAL = 0.8

plt.rcParams["figure.figsize"] = [10, 5]
bp_list = []
for i, (target, target_name) in enumerate(zip(['target1', 'target2', 'target4'],
                                                      ['Target1', 'Target2', 'Target4'])):

    for j, (pose_model, edge_color, fill_color) in enumerate(zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
                                                             ['blue', 'red', 'green'],
                                                             ['pink', 'lime', 'gold'])):
        error_opt = error_dict_opt[pose_model][target]
        error_opt = [1000 * z for z in error_opt]  # convert m to mm

        bp = plt.boxplot([error_opt], positions=[(i+1)*BIG_INTERVAL+j*SMALL_INTERVAL], patch_artist=True, widths=0.3, whis=99)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

        if i == 0:
            bp_list.append(bp)


plt.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0], bp_list[2]["boxes"][0]],
           ['ViTPose_large', 'ViTPose_base', 'OpenPose'], loc='upper left', prop={'size': 15})

plt.xticks([(i + 1) * BIG_INTERVAL + SMALL_INTERVAL for i in range(3)], ['Target1', 'Target2', 'Target4'], fontsize=14)
plt.tick_params(axis='y', labelsize=12)
plt.ylabel('Error (mm)', fontsize=15)

plt.savefig('error_distribution.pdf', bbox_inches='tight')
plt.show()
