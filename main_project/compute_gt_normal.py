# Compute the ground-truth normal, using the two_cam method

import os
import pickle
import numpy as np

import open3d as o3d
from utils.pose_conversion import *
from utils.trajectory_io import *

data_path = 'data'

depth_cam1_intr = np.array([
    [603.98217773, 0, 310.87359619],
    [0, 603.98217773, 231.11578369],
    [0, 0, 1]
])
depth_cam2_intr = np.array([
    [595.13745117, 0, 318.53710938],
    [0, 595.13745117, 245.47492981],
    [0, 0, 1]
])


def two_cam_TSDF(folder_path):
    camera_poses = read_trajectory(folder_path + "odometry.log")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_poses)):
        color = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(i + 1))
        depth = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(i + 1))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

        cam_intr = depth_cam1_intr if i == 0 else depth_cam2_intr  # use depth camera's intrinsics

        intr = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=cam_intr[0, 0],
            fy=cam_intr[1, 1],
            cx=cam_intr[0, 2],
            cy=cam_intr[1, 2]
        )
        volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose))

    pcd = volume.extract_point_cloud()
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    coordinates = np.asarray(downpcd.points)
    normals = np.asarray(downpcd.normals)

    return coordinates, normals

# loop through the data folder
for SUBJECT_NAME in os.listdir(data_path):
    subject_folder_path = os.path.join(data_path, SUBJECT_NAME)
    if os.path.isfile(subject_folder_path):
        continue
    print("Subject: ", SUBJECT_NAME)

    # frontal points: target 1 & target 2
    front_folder_path = subject_folder_path + '/front/'
    with open(front_folder_path + 'two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)
    two_cam_tar1_gt, two_cam_tar2_gt = two_cam_gt['target_1'].flatten(), two_cam_gt['target2_3d'].flatten()
    coordinates, normals = two_cam_TSDF(front_folder_path)

    # target 1
    tar1_init_estimate = two_cam_tar1_gt.squeeze()  # an initial estimate of the target point
    idx = np.argmin(np.square(coordinates[:, 0:2] - tar1_init_estimate[0:2]).sum(axis=1))
    target1_normal = normals[idx]

    # target 2
    tar2_init_estimate = two_cam_tar2_gt.squeeze()  # an initial estimate of the target point
    idx = np.argmin(np.square(coordinates[:, 0:2] - tar2_init_estimate[0:2]).sum(axis=1))
    target2_normal = normals[idx]

    gt_normal_dict = {'target1_normal': target1_normal, 'target2_normal': target2_normal}
    with open(front_folder_path + 'two_cam_gt_normal.pickle', 'wb') as f:
        pickle.dump(gt_normal_dict, f)

    # side point: target 4
    side_folder_path = subject_folder_path + '/side/'
    with open(side_folder_path + 'two_cam_gt.pickle', 'rb') as f:
        two_cam_gt = pickle.load(f)
    two_cam_tar4_gt = two_cam_gt['target_4'].flatten()
    coordinates, normals = two_cam_TSDF(side_folder_path)

    # target 4
    tar4_init_estimate = two_cam_tar4_gt.squeeze()  # an initial estimate of the target point
    idx = np.argmin(np.square(coordinates[:, 0:2] - tar4_init_estimate[0:2]).sum(axis=1))
    target4_normal = normals[idx]

    gt_normal_dict = {'target4_normal': target4_normal}
    with open(side_folder_path + 'two_cam_gt_normal.pickle', 'wb') as f:
        pickle.dump(gt_normal_dict, f)

