# Overall error of all targets
# Overall success rate of all targets
# Leave-one-out, ViTPose_large

import numpy as np
import json

# Overall error
# err_type = 'normal'  # position, normal
# with open('../evaluation/k_fold_validation_' + err_type + '_result/' + 'ViTPose_large_one_out_err.json', 'rb') as f:
#     vit_large_err = json.load(f)
# all_err = vit_large_err['target1'] + vit_large_err['target2'] + vit_large_err['target4']
# print(err_type + ' error mean: ', np.mean(all_err))
# print(err_type + ' error std: ', np.std(all_err))

# Success rate
with open('../evaluation/k_fold_success_rate/ViTPose_large_one_out_success_rate.json', 'rb') as f:
    success_rate= json.load(f)
tar1_success_rate = np.array(success_rate['target1'])  # (30, 9)
tar2_success_rate = np.array(success_rate['target2'])  # (30, 9)
tar4_success_rate = np.array(success_rate['target4'])  # (30, 9)
all_success_rate = np.vstack((tar1_success_rate, tar2_success_rate, tar4_success_rate))

print('success rate mean: ', np.mean(all_success_rate, axis=0))
print('success rate std: ', np.std(all_success_rate, axis=0))