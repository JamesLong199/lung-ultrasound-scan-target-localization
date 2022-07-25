import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('../evaluation/normal_error_dict_opt.pickle', 'rb') as f:
    error_dict_opt = pickle.load(f)
# print(error_dict_opt)

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
        error_opt = [z for z in error_opt]

        bp = plt.boxplot([error_opt], positions=[(i+1)*BIG_INTERVAL+j*SMALL_INTERVAL], patch_artist=True, widths=0.3, whis=99)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

        if i == 0:
            bp_list.append(bp)


plt.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0], bp_list[2]["boxes"][0]],
           ['ViTPose_large', 'ViTPose_base', 'OpenPose'], loc='upper center', prop={'size': 15})

plt.xticks([(i + 1) * BIG_INTERVAL + SMALL_INTERVAL for i in range(3)], ['Target1', 'Target2', 'Target4'], fontsize=14)
plt.tick_params(axis='y', labelsize=12)
plt.ylabel('Orientation Error (degree)', fontsize=15)

plt.savefig('normal_error_distribution.pdf', bbox_inches='tight')
plt.show()
