# Compute success rate mean & std for all thresholds with K-fold validation
# 1. 2-8 split
# 2. 1-9 split
# 3. "Leave one out": 29 train + 1 test

import json
import subprocess
from src.optimize import *

method = 'one_out'  # 28_split, 19_split, one_out
POSE_MODEL = 'OpenPose'  # ViTPose_large, ViTPose_base, OpenPose
thres_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # error threshold in mm


def success_rate(thres_list, value_list):
    rate_list = []
    for thres in thres_list:
        count = 0
        for value in value_list:
            if value < thres:
                count += 1
        success_rate = (count / len(value_list)) * 100
        rate_list.append(success_rate)
    return rate_list


K = 30
fold = 30
fold_len = 1
test_fold = int(K / fold)
train_fold = K - test_fold

# collect data
target12_data = [[], [], []]  # list of list of np.array: [X1_list, X2_list, t2_list]
target1_GT_pos = []  # list of np.array
target2_GT_pos = []

target4_data = [[], [], []]
target4_GT_pos = []
subject_names = []

for SUBJECT_NAME in os.listdir('../data'):
    subject_folder_path = os.path.join('../data', SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        continue
    subject_names.append(SUBJECT_NAME)

    scan_pose = 'front'
    with open(subject_folder_path + '/' + scan_pose + '/' + POSE_MODEL + '/position_data.pickle', 'rb') as f:
        position_data = pickle.load(f)

    target12_data[0].append(position_data[scan_pose][0])  # X1
    target12_data[1].append(position_data[scan_pose][1])  # X2
    target12_data[2].append(position_data[scan_pose][2])  # t2

    with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt.pickle', 'rb') as f:
        ground_truth = pickle.load(f)
    target1_GT_pos.append(ground_truth['target_1'])
    target2_GT_pos.append(ground_truth['target_2'])

    scan_pose = 'side'
    with open(subject_folder_path + '/' + scan_pose + '/' + POSE_MODEL + '/position_data.pickle', 'rb') as f:
        position_data = pickle.load(f)

    target4_data[0].append(position_data[scan_pose][0])  # X1
    target4_data[1].append(position_data[scan_pose][1])  # X2
    target4_data[2].append(position_data[scan_pose][2])  # t2

    with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt.pickle', 'rb') as f:
        ground_truth = pickle.load(f)
    target4_GT_pos.append(ground_truth['target_4'])

target12_data = np.array(target12_data)  # (3,30,3,1)
target4_data = np.array(target4_data)  # (3,30,3,1)
target1_GT_pos = np.array(target1_GT_pos)  # (30, 3, 1)
target2_GT_pos = np.array(target2_GT_pos)
target4_GT_pos = np.array(target4_GT_pos)

success_rate_dict = {'target1': [], 'target2': [], 'target4': []}

# K-fold cross validation
for i in range(fold):
    print(i + 1, "th fold")
    print("=============================================================")

    # train
    test_idx_from = i * test_fold * fold_len
    test_idx_to = (i + 1) * test_fold * fold_len

    target12_data_train = np.hstack((target12_data[:, 0:test_idx_from], target12_data[:, test_idx_to:]))
    target4_data_train = np.hstack((target4_data[:, 0:test_idx_from], target4_data[:, test_idx_to:]))

    target1_GT_pos_test = target1_GT_pos[test_idx_from:test_idx_to]
    target2_GT_pos_test = target2_GT_pos[test_idx_from:test_idx_to]
    target4_GT_pos_test = target4_GT_pos[test_idx_from:test_idx_to]

    target1_GT_train = np.vstack((target1_GT_pos[0:test_idx_from], target1_GT_pos[test_idx_to:]))
    target2_GT_train = np.vstack((target2_GT_pos[0:test_idx_from], target2_GT_pos[test_idx_to:]))
    target4_GT_train = np.vstack((target4_GT_pos[0:test_idx_from], target4_GT_pos[test_idx_to:]))

    target1_ratio = optimize_front_linear(target12_data_train, target1_GT_train)
    target2_ratio = optimize_front_linear(target12_data_train, target2_GT_train)
    print("target1_ratio: \n", target1_ratio)
    print("target2_ratio: \n", target2_ratio)

    target4_ratio = optimize_side(target4_data_train, target4_GT_train)
    print("target4_ratio: \n", target4_ratio)

    # test
    subject_test = subject_names[test_idx_from:test_idx_to]

    tar1_pos_fold = []
    tar2_pos_fold = []
    tar4_pos_fold = []

    for target1_gt_pos, target2_gt_pos, target4_gt_pos, SUBJECT_NAME in zip(target1_GT_pos_test, target2_GT_pos_test,
                                                                            target4_GT_pos_test, subject_test):

        print("subject: ", SUBJECT_NAME)
        # front
        subprocess.run([
            "python", "../compute_target.py",
            "--pose_model={}".format(POSE_MODEL),
            "--subject_name={}".format(SUBJECT_NAME),
            "--scan_pose={}".format('front'),
            "--target1_r1", str(target1_ratio[0]),
            "--target1_r2", str(target1_ratio[1]),
            "--target2_r1", str(target2_ratio[0]),
            "--target2_r2", str(target2_ratio[1]),
            "--target4_r1", str(target4_ratio[0]),
            "--target4_r2", str(target4_ratio[1])
        ])
        front_path = '../data/' + SUBJECT_NAME + '/front/'
        with open(front_path + POSE_MODEL + '/tmp_target_test.pickle', 'rb') as f:
            front_pos_pred = pickle.load(f)
        target1_pos_pred, target2_pos_pred = front_pos_pred[0], front_pos_pred[1]

        target1_pos_err = np.linalg.norm(target1_gt_pos - target1_pos_pred) * 1000
        target2_pos_err = np.linalg.norm(target2_gt_pos - target2_pos_pred) * 1000
        print("target1 pos error: ", target1_pos_err)
        print("target2 pos error: ", target2_pos_err)
        tar1_pos_fold.append(target1_pos_err)
        tar2_pos_fold.append(target2_pos_err)

        # skip outlier for openpose target 4:
        if POSE_MODEL == 'OpenPose' and (SUBJECT_NAME == 'charles_xu' or SUBJECT_NAME == 'jingyu_wu'):
            tar4_pos_fold += [1000]
            continue

        # side
        subprocess.run([
            "python", "../compute_target.py",
            "--pose_model={}".format(POSE_MODEL),
            "--subject_name={}".format(SUBJECT_NAME),
            "--scan_pose={}".format('side'),
            "--target1_r1", str(target1_ratio[0]),
            "--target1_r2", str(target1_ratio[1]),
            "--target2_r1", str(target2_ratio[0]),
            "--target2_r2", str(target2_ratio[1]),
            "--target4_r1", str(target4_ratio[0]),
            "--target4_r2", str(target4_ratio[1])
        ])
        side_path = '../data/' + SUBJECT_NAME + '/side/'
        with open(side_path + POSE_MODEL + '/tmp_target_test.pickle', 'rb') as f:
            side_pos_pred = pickle.load(f)
        target4_pos_pred = side_pos_pred[0]
        target4_pos_err = np.linalg.norm(target4_gt_pos - target4_pos_pred) * 1000
        print("target4 pos error: ", target4_pos_err)
        tar4_pos_fold.append(target4_pos_err)

    success_rate_dict['target1'].append(success_rate(thres_list, tar1_pos_fold))
    success_rate_dict['target2'].append(success_rate(thres_list, tar2_pos_fold))
    success_rate_dict['target4'].append(success_rate(thres_list, tar4_pos_fold))


success_rate_dict['target1_mean'] = np.mean(np.array(success_rate_dict['target1']), axis=0).tolist()
success_rate_dict['target1_std'] = np.std(np.array(success_rate_dict['target1']), axis=0).tolist()
success_rate_dict['target2_mean'] = np.mean(np.array(success_rate_dict['target2']), axis=0).tolist()
success_rate_dict['target2_std'] = np.std(np.array(success_rate_dict['target2']), axis=0).tolist()
success_rate_dict['target4_mean'] = np.mean(np.array(success_rate_dict['target4']), axis=0).tolist()
success_rate_dict['target4_std'] = np.std(np.array(success_rate_dict['target4']), axis=0).tolist()

json_object = json.dumps(success_rate_dict, indent=4)
with open('k_fold_success_rate/' + POSE_MODEL + '_' + method + '_success_rate.json', 'w') as f:
    f.write(json_object)