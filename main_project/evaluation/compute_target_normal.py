# Compute the target normal, using the two_cam method
# Compute again in evaluation since we forgot to save the normals before.

import os
import pickle

import open3d as o3d
from utils.trajectory_io import *

data_path = '../data'

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
    coordinates, normals = two_cam_TSDF(front_folder_path)

    for pose_model in ['OpenPose', 'ViTPose_base', 'ViTPose_large']:
        model_folder_path = front_folder_path + pose_model + '/'
        with open(model_folder_path + 'final_target_opt.pickle', 'rb') as f:
            target = pickle.load(f)
        tar1, tar2 = target[0].flatten(), target[1].flatten()

        # target 1
        tar1_init_estimate = tar1.squeeze()
        idx = np.argmin(np.square(coordinates[:, 0:2] - tar1_init_estimate[0:2]).sum(axis=1))
        target1_normal = normals[idx]

        # target 2
        tar2_init_estimate = tar2.squeeze()
        idx = np.argmin(np.square(coordinates[:, 0:2] - tar2_init_estimate[0:2]).sum(axis=1))
        target2_normal = normals[idx]

        target_normal_dict = {'target1_normal': target1_normal, 'target2_normal': target2_normal}
        with open(model_folder_path + 'final_target_normal_opt.pickle', 'wb') as f:
            pickle.dump(target_normal_dict, f)

    # side point: target 4
    side_folder_path = subject_folder_path + '/side/'
    coordinates, normals = two_cam_TSDF(side_folder_path)

    for pose_model in ['OpenPose', 'ViTPose_base', 'ViTPose_large']:
        model_folder_path = side_folder_path + pose_model + '/'
        with open(model_folder_path + 'final_target_opt.pickle', 'rb') as f:
            target = pickle.load(f)
        tar4 = target[0].flatten()

        # target 4
        tar4_init_estimate = tar4.squeeze()
        idx = np.argmin(np.square(coordinates[:, 0:2] - tar4_init_estimate[0:2]).sum(axis=1))
        target4_normal = normals[idx]
        target_normal_dict = {'target4_normal': target4_normal}
        with open(model_folder_path + 'final_target_normal_opt.pickle', 'wb') as f:
            pickle.dump(target_normal_dict, f)