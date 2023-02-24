# K-fold cross validation for the ratio model
# 1. 2-8 split
# 2. 1-9 split
# 3. "Leave one out": 29 train + 1 test
# Compute error mean and std.

import json
import subprocess
from src.optimize import *

POSE_MODEL = 'ViTPose_large'  # ViTPose_large, ViTPose_base, OpenPose


def angle_difference(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    if dot_product > 1:
        dot_product = 1
    angle = np.degrees(np.arccos(dot_product))
    return angle


K = 30
fold = 30
fold_len = 1
test_fold = int(K / fold)
train_fold = K - test_fold

# collect data
target12_data = [[], [], []]  # list of list of np.array: [X1_list, X2_list, t2_list]
target1_GT_pos = []  # list of np.array
target2_GT_pos = []
target1_GT_normal = []
target2_GT_normal = []

target4_data = [[], [], []]
target4_GT_pos = []
target4_GT_normal = []
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

    with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt_normal.pickle', 'rb') as f:
        ground_truth = pickle.load(f)
    target1_GT_normal.append(ground_truth['target1_normal'])
    target2_GT_normal.append(ground_truth['target2_normal'])

    scan_pose = 'side'
    with open(subject_folder_path + '/' + scan_pose + '/' + POSE_MODEL + '/position_data.pickle', 'rb') as f:
        position_data = pickle.load(f)

    target4_data[0].append(position_data[scan_pose][0])  # X1
    target4_data[1].append(position_data[scan_pose][1])  # X2
    target4_data[2].append(position_data[scan_pose][2])  # t2

    with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt.pickle', 'rb') as f:
        ground_truth = pickle.load(f)
    target4_GT_pos.append(ground_truth['target_4'])

    with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt_normal.pickle', 'rb') as f:
        ground_truth = pickle.load(f)
    target4_GT_normal.append(ground_truth['target4_normal'])

target12_data = np.array(target12_data)  # (3,30,3,1)
target4_data = np.array(target4_data)  # (3,30,3,1)
target1_GT_pos = np.array(target1_GT_pos)  # (30, 3, 1)
target2_GT_pos = np.array(target2_GT_pos)
target4_GT_pos = np.array(target4_GT_pos)
target1_GT_normal = np.array(target1_GT_normal)
target2_GT_normal = np.array(target2_GT_normal)
target4_GT_normal = np.array(target4_GT_normal)

pos_err_dict = {'target1': [], 'target2': [], 'target4': []}  # store the error mean of each fold
normal_err_dict = {'target1': [], 'target2': [], 'target4': []}

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
    target1_GT_normal_test = target1_GT_normal[test_idx_from:test_idx_to]
    target2_GT_normal_test = target2_GT_normal[test_idx_from:test_idx_to]
    target4_GT_normal_test = target4_GT_normal[test_idx_from:test_idx_to]

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

    tar1_normal_fold = []
    tar2_normal_fold = []
    tar4_normal_fold = []

    for target1_gt_pos, target2_gt_pos, target4_gt_pos, target1_gt_normal, target2_gt_normal, target4_gt_normal, SUBJECT_NAME in zip(
            target1_GT_pos_test, target2_GT_pos_test,
            target4_GT_pos_test, target1_GT_normal_test, target2_GT_normal_test, target4_GT_normal_test, subject_test):

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

        with open(front_path + '/pcd_coordinates.pickle', 'rb') as f:
            coordinates = pickle.load(f)
        with open(front_path + '/pcd_normals.pickle', 'rb') as f:
            normals = pickle.load(f)
        idx = np.argmin(np.square(coordinates[:, 0:2] - target1_pos_pred.squeeze()[0:2]).sum(axis=1))
        target1_normal_pred = normals[idx]
        idx = np.argmin(np.square(coordinates[:, 0:2] - target2_pos_pred.squeeze()[0:2]).sum(axis=1))
        target2_normal_pred = normals[idx]

        target1_normal_err = angle_difference(target1_normal_pred, target1_gt_normal)
        target2_normal_err = angle_difference(target2_normal_pred, target2_gt_normal)
        print("target1 normal error: ", target1_normal_err)
        print("target2 normal error: ", target2_normal_err)
        tar1_normal_fold.append(target1_normal_err)
        tar2_normal_fold.append(target2_normal_err)

        # skip outlier for openpose target 4:
        if POSE_MODEL == 'OpenPose' and (SUBJECT_NAME == 'charles_xu' or SUBJECT_NAME == 'jingyu_wu'):
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

        with open(side_path + '/pcd_coordinates.pickle', 'rb') as f:
            coordinates = pickle.load(f)
        with open(side_path + '/pcd_normals.pickle', 'rb') as f:
            normals = pickle.load(f)
        idx = np.argmin(np.square(coordinates[:, 0:2] - target4_pos_pred.squeeze()[0:2]).sum(axis=1))
        target4_normal_pred = normals[idx]

        target4_normal_err = angle_difference(target4_normal_pred, target4_gt_normal)
        print("target4 normal error: ", target4_normal_err)
        tar4_normal_fold.append(target4_normal_err)

    pos_err_dict['target1'].append(np.mean(tar1_pos_fold))
    pos_err_dict['target2'].append(np.mean(tar2_pos_fold))
    normal_err_dict['target1'].append(np.mean(tar1_normal_fold))
    normal_err_dict['target2'].append(np.mean(tar2_normal_fold))
    if len(tar4_pos_fold) != 0:
        pos_err_dict['target4'].append(np.mean(tar4_pos_fold))
        normal_err_dict['target4'].append(np.mean(tar4_normal_fold))


pos_err_dict['target1_mean'] = np.mean(pos_err_dict['target1'])
pos_err_dict['target1_std'] = np.std(pos_err_dict['target1'])
pos_err_dict['target2_mean'] = np.mean(pos_err_dict['target2'])
pos_err_dict['target2_std'] = np.std(pos_err_dict['target2'])
pos_err_dict['target4_mean'] = np.mean(pos_err_dict['target4'])
pos_err_dict['target4_std'] = np.std(pos_err_dict['target4'])

normal_err_dict['target1_mean'] = np.mean(normal_err_dict['target1'])
normal_err_dict['target1_std'] = np.std(normal_err_dict['target1'])
normal_err_dict['target2_mean'] = np.mean(normal_err_dict['target2'])
normal_err_dict['target2_std'] = np.std(normal_err_dict['target2'])
normal_err_dict['target4_mean'] = np.mean(normal_err_dict['target4'])
normal_err_dict['target4_std'] = np.std(normal_err_dict['target4'])

pos_json_object = json.dumps(pos_err_dict, indent=4)
with open('k_fold_validation_position_result/' + POSE_MODEL + '_one_out_err.json', 'w') as f:
    f.write(pos_json_object)

normal_json_object = json.dumps(normal_err_dict, indent=4)
with open('k_fold_validation_normal_result/' + POSE_MODEL + '_one_out_err.json', 'w') as f:
    f.write(normal_json_object)