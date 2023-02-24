# Visualiza ground-truth in 2D image to see whether they are accurate

import os
import pickle
import cv2
import matplotlib.pyplot as plt
from utils.triangulation import *
from utils.trajectory_io import *


def draw_target_point(folder_path, target_point1, target_point2):
    image1 = plt.imread(folder_path + '/color_images/cam_1.jpg')
    cv2.circle(image1, (int(target_point1[0]), int(target_point1[1])), 2, (36, 255, 12), 2, -1)

    plt.subplot(121)
    plt.imshow(image1)

    image2 = plt.imread(folder_path + '/color_images/cam_2.jpg')
    cv2.circle(image2, (int(target_point2[0]), int(target_point2[1])), 2, (36, 255, 12), 2, -1)

    plt.subplot(122)
    plt.imshow(image2)

    plt.show()


if __name__ == '__main__':

    data_path = '../data'

    # loop through the data folder
    for SUBJECT_NAME in os.listdir(data_path):
        print("subject: ", SUBJECT_NAME)
        subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
        if os.path.isfile(subject_folder_path):
            continue

        scan_pose = 'side'
        with open(subject_folder_path + '/' + scan_pose + '/cam_2_gt.pickle', 'rb') as f:
            ground_truth = pickle.load(f)
        target4_GT = ground_truth['target_4']

        # read RGB intrinsics
        with open(subject_folder_path + '/' + scan_pose + '/intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
            cam1_intr = pickle.load(f)

        with open(subject_folder_path + '/' + scan_pose + '/intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
            cam2_intr = pickle.load(f)

        # read extrinsics
        camera_poses = read_trajectory(subject_folder_path + '/' + scan_pose + "/odometry.log")
        T_cam1_base = camera_poses[0].pose
        T_base_cam1 = np.linalg.inv(T_cam1_base)
        T_cam2_base = camera_poses[1].pose
        T_base_cam2 = np.linalg.inv(T_cam2_base)

        target4_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target4_GT)).squeeze()
        target4_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target4_GT)).squeeze()

        draw_target_point(subject_folder_path + '/' + scan_pose, target4_2d_cam1, target4_2d_cam2)

        scan_pose = 'front'
        with open(subject_folder_path + '/' + scan_pose + '/cam_2_gt.pickle', 'rb') as f:
            ground_truth = pickle.load(f)
        target1_GT = ground_truth['target_1']
        target2_GT = ground_truth['target_2']

        # read RGB intrinsics
        with open(subject_folder_path + '/' + scan_pose + '/intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
            cam1_intr = pickle.load(f)

        with open(subject_folder_path + '/' + scan_pose + '/intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
            cam2_intr = pickle.load(f)

        # read extrinsics
        camera_poses = read_trajectory(subject_folder_path + '/' + scan_pose + "/odometry.log")
        T_cam1_base = camera_poses[0].pose
        T_base_cam1 = np.linalg.inv(T_cam1_base)
        T_cam2_base = camera_poses[1].pose
        T_base_cam2 = np.linalg.inv(T_cam2_base)

        target1_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target1_GT)).squeeze()
        target1_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target1_GT)).squeeze()

        target2_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target2_GT)).squeeze()
        target2_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target2_GT)).squeeze()

        draw_target_point(subject_folder_path + '/' + scan_pose, target1_2d_cam1, target1_2d_cam2)
        draw_target_point(subject_folder_path + '/' + scan_pose, target2_2d_cam1, target2_2d_cam2)