# Plot the 3D position error scatter of 3 targets under 3 HPE models

import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('../evaluation/error_dict.pickle', 'rb') as f:
    error_dict = pickle.load(f)

with open('../evaluation/error_dict_opt.pickle', 'rb') as f:
    error_dict_opt = pickle.load(f)

fig, ax = plt.subplots(1,3, figsize=(20, 5))
for i, target in enumerate(['target1', 'target2', 'target4']):
    for pose_model, pose_model_name, color in zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
                                                  ['ViT_large', 'ViT_base', 'OpenPose'],
                                                  ['red', 'blue', 'green']):
        error = error_dict[pose_model][target]
        error_opt = error_dict_opt[pose_model][target]
        error = [1000 * j for j in error]  # convert m to mm
        error_opt = [1000 * j for j in error_opt]  # convert m to mm
        x = [pose_model_name, pose_model_name + '*']
        y = [error, error_opt]
        ax[i].scatter(x=np.repeat(x, len(y[0])), y=y, color=color)

    ax[i].set_title(target)
    ax[i].set_xlabel('HPE model')
    ax[i].set_ylabel('Error (mm)')
    # ax[i].set_ylim(bottom=0, top=70)

plt.show()

# rainbow bubble
# with open('../error_dict_opt.pickle', 'rb') as f:
#     error_dict_opt = pickle.load(f)
#
# plt.rcParams["figure.figsize"] = [10, 10]
# for i, (pose_model, pose_model_name) in enumerate(zip(['vit_large_err', 'vit_base_err', 'open_pose_err'],
#                                                       ['ViT_large', 'ViT_base', 'OpenPose'])):
#     for target, color in zip(['target1', 'target2', 'target4'], ['red', 'blue', 'green']):
#
#         error_opt = error_dict_opt[pose_model][target]
#         error_opt = [1000 * j for j in error_opt]  # convert m to mm
#
#         x = [pose_model_name + '*']
#         y = [error_opt]
#         plt.scatter(x=np.repeat(x, len(y[0])), y=y, color=color, alpha=0.2)
#
# plt.xlabel('HPE model')
# plt.ylabel('Error (mm)')
# # plt.xlim(left=0, right=4)
# # plt.title('Error mean and std')
# plt.show()

