# Plot success rate heatmap under increasing error threshold

import pickle

from heat_map import *

with open('../evaluation/error_dict.pickle', 'rb') as f:
    error_dict = pickle.load(f)

with open('../evaluation/error_dict_opt.pickle', 'rb') as f:
    error_dict_opt = pickle.load(f)


def success_rate(thres_list, value_list):
    rate_list = []
    for thres in thres_list:
        count = 0
        for value in value_list:
            if value < thres:
                count += 1
        success_rate = (count / len(value_list)) * 100
        rate_list.append(success_rate)
    return np.array(rate_list)

# thres_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # error threshold in mm
thres_list = [10, 15, 20, 25, 30, 35, 40]  # error threshold in mm

# separate target
fig, ax = plt.subplots(2, 3, figsize=(17.5, 5))
for i, (target, target_name) in enumerate(zip(['target1', 'target2', 'target4'],['Target1', 'Target2', 'Target4'] )):
    data = np.zeros((3, len(thres_list)))
    data_opt = np.zeros((3, len(thres_list)))

    for j, pose_model in enumerate(['vit_large_err', 'vit_base_err', 'open_pose_err']):
        error = error_dict[pose_model][target]
        error_opt = error_dict_opt[pose_model][target]
        error = [1000 * j for j in error]  # convert m to mm
        error_opt = [1000 * j for j in error_opt]  # convert m to mm
        rate_list = success_rate(thres_list, error)
        rate_list_opt = success_rate(thres_list, error_opt)

        data[j, :] = rate_list
        data_opt[j, :] = rate_list_opt

    row_labels = ['ViT-L', 'ViT-B', 'OP']
    row_labels_opt = ['ViT-L*', 'ViT-B*', 'OP*']

    add_yticks = True if i == 0 else False
    im = heatmap(data, row_labels, thres_list, ax=ax[0, i], title=target_name, add_yticks=add_yticks, cmap="YlGn", cbarlabel="success rate %")
    texts = annotate_heatmap(im, data, valfmt="{x:.1f}")

    xlabel = 'Error Threshold (mm)' if i == 1 else None
    im_opt = heatmap(data_opt, row_labels_opt, thres_list, ax=ax[1, i], add_xticks=True, add_yticks=add_yticks, xlabel=xlabel, cmap="YlGn", cbarlabel="success rate %")
    texts_opt = annotate_heatmap(im_opt, data_opt, valfmt="{x:.1f}")


# add a color bar
cb_ax = fig.add_axes([0.31, -0.08, 0.4, 0.02])
cbar = fig.colorbar(im, orientation="horizontal", cax=cb_ax)
cbar.ax.set_xlabel('success rate %', fontsize=16)
cbar.ax.tick_params(axis='x', labelsize=13)

plt.subplots_adjust(wspace=0.0, hspace=0.08)
plt.savefig('success_rate_heatmap.pdf', bbox_inches='tight')
plt.show()
