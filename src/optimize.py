# Script for optimizing the ratio parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import time

POSE_MODEL = 'OpenPose'  # ViTPose_large, ViTPose_base, OpenPose

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Planar_Euclidean_Loss(nn.Module):
    '''
    Euclidean distance of x,y coordinates between 3D points, ignore z coordinate
    '''

    def __init__(self):
        super(Planar_Euclidean_Loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.nn.functional.mse_loss(pred[0:2,:], target[0:2,:])  # only consider xy-axis
        return loss


class Target4_Model(nn.Module):
    '''
    target computation model
    '''

    def __init__(self, r1, r2):
        super(Target4_Model, self).__init__()
        self.r1 = nn.Parameter(torch.tensor(r1), requires_grad=True)
        self.r2 = nn.Parameter(torch.tensor(r2), requires_grad=True)

    def forward(self, X1, X2, t2):
        X3 = X1 + self.r1 * (X2 - X1)
        pred = X3 + self.r2 * torch.norm(X3 - X1) * t2
        return pred


class PositionDataset(Dataset):
    def __init__(self, X1, X2, t2, target):
        self.X1 = X1
        self.X2 = X2
        self.t2 = t2
        self.target = target

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.t2[i], self.target[i]


def optimize_side(data, target, params=[0.35, 0.1], epoch=1000, lr=0.01, use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    X1, X2, t2 = data
    dataset = PositionDataset(X1, X2, t2, target)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = Target4_Model(*params)
    model.to(device)
    print('model.state_dict():', model.state_dict())
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = Planar_Euclidean_Loss()

    for ep in range(epoch):
        total_loss = 0
        for i, data in enumerate(data_loader):
            X1, X2, t2, target = data

            X1 = X1.to(device)
            X2 = X2.to(device)
            t2 = t2.to(device)
            target = target.to(device)

            opt.zero_grad()
            pred = model.forward(X1, X2, t2)
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            loss.backward()
            opt.step()

        if ep % 100 == 0:
            print('epoch: {}, loss: {}, r1: {}, r2: {}'.format(ep, total_loss, model.state_dict()['r1'], model.state_dict()['r2']))


def optimize_front_linear(data, target):
    '''
    Optimize the two ratios using least square.
    Only use x&y values.
     :X1, X2 --3D coordinates of the right shoulder and the right hip
     :t2 -- the direction vector of the line connecting X3 and target
    '''
    X1, X2, t2 = data[0], data[1], data[2]

    # Onlu use x&y values
    X1_xy = np.hstack(X1)[0:2,:].T.reshape(-1, 1)   # (2nx1)
    X2_xy = np.hstack(X2)[0:2,:].T.reshape(-1, 1)

    r1_coeff = X2_xy - X1_xy    # 2nx1
    r2_coeff = (np.hstack(t2)[0:2,:].T * np.sqrt(((np.hstack(X1)-np.hstack(X2))**2).sum(axis=0)).reshape(-1,1)).reshape(-1,1)  # 2nx1
    A = np.hstack([r1_coeff, r2_coeff])
    b = np.hstack(target)[0:2,:].T.reshape(-1,1) - X1_xy   # reshape (nx2) to (2nx1)
    r = np.linalg.pinv(A) @ b

    return r


if __name__ == '__main__':

    print("HPE model: ", POSE_MODEL)
    # collect data
    target12_data = [[], [], []]   # list of list of np.array: [X1_list, X2_list, t2_list]
    target1_GT = []               # list of np.array
    target2_GT = []

    target4_data = [[], [], []]
    target4_GT = []

    for SUBJECT_NAME in os.listdir('data'):
        subject_folder_path = os.path.join('data', SUBJECT_NAME)
        if os.path.isfile(subject_folder_path):
            continue

        scan_pose = 'front'
        with open(subject_folder_path + '/' + scan_pose + '/' + POSE_MODEL + '/position_data.pickle', 'rb') as f:
            position_data = pickle.load(f)

        target12_data[0].append(position_data[scan_pose][0])  # X1
        target12_data[1].append(position_data[scan_pose][1])  # X2
        target12_data[2].append(position_data[scan_pose][2])  # t2

        with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt.pickle', 'rb') as f:
            ground_truth = pickle.load(f)
        target1_GT.append(ground_truth['target_1'])
        target2_GT.append(ground_truth['target2_3d'])

        # skip outlier for openpose target 4:
        if POSE_MODEL == 'OpenPose' and (SUBJECT_NAME == 'charles_xu' or SUBJECT_NAME == 'jingyu_wu'):
            continue

        scan_pose = 'side'
        with open(subject_folder_path + '/' + scan_pose + '/' + POSE_MODEL + '/position_data.pickle', 'rb') as f:
            position_data = pickle.load(f)

        target4_data[0].append(position_data[scan_pose][0])  # X1
        target4_data[1].append(position_data[scan_pose][1])  # X2
        target4_data[2].append(position_data[scan_pose][2])  # t2

        with open(subject_folder_path + '/' + scan_pose + '/two_cam_gt.pickle', 'rb') as f:
            ground_truth = pickle.load(f)
        target4_GT.append(ground_truth['target_4'])

    start_time = time.time()
    optimize_side(target4_data, target4_GT)
    print('training used {:.3f} s'.format(time.time() - start_time))

    target1_ratio = optimize_front_linear(target12_data, target1_GT)
    target2_ratio = optimize_front_linear(target12_data, target2_GT)
    print("target1_ratio: \n", target1_ratio)
    print("target2_ratio: \n", target2_ratio)









