# read excel subject data
# compute and save ground-truth to the data folder

import pandas as pd
import pyrealsense2 as rs
import pickle
import open3d as o3d

from utils.trajectory_io import *
from utils.triangulation import *

excel_data_df = pd.read_excel("subject_2D_gt.xlsx")
data = excel_data_df.to_numpy()

# depth cameras info:
depth_unit = 0.0010000000474974513

depth_cam1_intr = np.array([
    [603.98217773, 0, 310.87359619],
    [0, 603.98217773, 231.11578369],
    [0, 0, 1]
])
color_cam1_intr = np.array([
    [614.11938477, 0, 315.48043823],
    [0, 613.32940674, 236.26939392],
    [0, 0, 1]
])

depth_cam2_intr = np.array([
    [595.13745117, 0, 318.53710938],
    [0, 595.13745117, 245.47492981],
    [0, 0, 1]
])
color_cam2_intr = np.array([
    [613.22027588, 0, 318.16168213],
    [0, 612.14581299, 235.91072083],
    [0, 0, 1]
])


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cam_intr):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = 640
    _intrinsics.height = 480
    _intrinsics.ppx = cam_intr[0, 2]
    _intrinsics.ppy = cam_intr[1, 2]
    _intrinsics.fx = cam_intr[0, 0]
    _intrinsics.fy = cam_intr[1, 1]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [0, 0, 0, 0, 0]

    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth[y, x] * depth_unit)
    #result[0]: right, result[1]: down, result[2]: forward
    # return result[2], -result[0], -result[1]
    return result


def one_cam_TSDF(color, depth, depth_intr, T_cam_base):
    # return a point cloud
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=1 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=depth_intr[0, 0],
        fy=depth_intr[1, 1],
        cx=depth_intr[0, 2],
        cy=depth_intr[1, 2]
    )

    volume.integrate(rgbd, intr, np.linalg.inv(T_cam_base))
    pcd = volume.extract_point_cloud()
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # o3d.visualization.draw_geometries([downpcd])
    coordinates = np.asarray(downpcd.points)
    return coordinates


def two_cam_TSDF(folder_path, camera_poses):
    # TSDF volume
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
    # o3d.visualization.draw_geometries([downpcd])
    coordinates = np.asarray(downpcd.points)
    return coordinates


def post_process(pc, original_3d):
    # keep the original (x,y), and find a new z
    idx = np.argmin(np.square(pc[:, 0:2] - original_3d.squeeze()[0:2]).sum(axis=1))
    new_z = pc[idx, 2]
    new_3d = np.vstack((original_3d[0:2], [new_z]))
    return new_3d


def one_cam_3d_gt(target_2d, depth_data, depth_intr, cam_pc, T_cam_base):
    target_3d_cam = convert_depth_to_phys_coord_using_realsense(int(target_2d[0]), int(target_2d[1]),
                                                            depth_data, depth_intr)  # in the cam frame
    target_3d_base = from_homog(
        T_cam_base @ to_homog(np.array(target_3d_cam).reshape(-1, 1)))  # convert to the base frame
    target_3d_base = post_process(cam_pc, target_3d_base)  # TSDF volume post-processing
    return target_3d_base


for subject in data:
    SUBJECT_NAME = subject[0]
    print("subject name: ", SUBJECT_NAME)

    # Convert String to Tuple using map() + tuple() + int + split()
    tar1_cam1_2d = np.array(tuple(map(float, subject[1][1:-2].split(', '))))
    tar1_cam2_2d = np.array(tuple(map(float, subject[2][1:-2].split(', '))))
    tar2_cam1_2d = np.array(tuple(map(float, subject[3][1:-2].split(', '))))
    tar2_cam2_2d = np.array(tuple(map(float, subject[4][1:-2].split(', '))))
    tar4_cam1_2d = np.array(tuple(map(float, subject[5][1:-2].split(', '))))
    tar4_cam2_2d = np.array(tuple(map(float, subject[6][1:-2].split(', '))))

    # frontal pose: two targets
    folder_path = 'data/' + SUBJECT_NAME + '/front/'
    camera_poses = read_trajectory(folder_path + "odometry.log")
    T_cam1_base = camera_poses[0].pose
    T_cam2_base = camera_poses[1].pose
    T_base_cam1 = np.linalg.inv(T_cam1_base)
    T_base_cam2 = np.linalg.inv(T_cam2_base)

    # one-cam method:
    for i, (front_targets, depth_intr, T_cam_base) in enumerate(
            zip([(tar1_cam1_2d, tar2_cam1_2d), (tar1_cam2_2d, tar2_cam2_2d)], [depth_cam1_intr, depth_cam2_intr],
                [T_cam1_base, T_cam2_base])):
        color_img = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(i+1))
        depth_img = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(i+1))
        depth_data = np.asanyarray(depth_img)
        cam_pc = one_cam_TSDF(color_img, depth_img, depth_intr, T_cam_base)
        target1_3d = one_cam_3d_gt(front_targets[0], depth_data, depth_intr, cam_pc, T_cam_base)
        target2_3d = one_cam_3d_gt(front_targets[1], depth_data, depth_intr, cam_pc, T_cam_base)

        gt_dict = {'target_1': target1_3d, 'target_2': target2_3d}
        with open(folder_path + 'cam_{}_gt.pickle'.format(i+1), 'wb') as f:
            pickle.dump(gt_dict, f)

    # two-cam method:
    two_cam_pc = two_cam_TSDF(folder_path, camera_poses)
    target1_3d = reconstruct(tar1_cam1_2d, tar1_cam2_2d, color_cam1_intr, color_cam2_intr, T_base_cam1, T_base_cam2)
    target1_3d = post_process(two_cam_pc, target1_3d)
    target2_3d = reconstruct(tar2_cam1_2d, tar2_cam2_2d, color_cam1_intr, color_cam2_intr, T_base_cam1, T_base_cam2)
    target2_3d = post_process(two_cam_pc, target2_3d)
    gt_dict = {'target_1': target1_3d, 'target_2': target2_3d}
    with open(folder_path + 'two_cam_gt.pickle', 'wb') as f:
        pickle.dump(gt_dict, f)

    # side pose: one target
    folder_path = 'data/' + SUBJECT_NAME + '/side/'
    camera_poses = read_trajectory(folder_path + "odometry.log")
    T_cam1_base = camera_poses[0].pose
    T_cam2_base = camera_poses[1].pose
    T_base_cam1 = np.linalg.inv(T_cam1_base)
    T_base_cam2 = np.linalg.inv(T_cam2_base)

    # one-cam method:
    for i, (side_target, depth_intr, T_cam_base) in enumerate(
        zip([tar4_cam1_2d, tar4_cam2_2d], [depth_cam1_intr, depth_cam2_intr], [T_cam1_base, T_cam2_base])):
        color_img = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(i + 1))
        depth_img = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(i + 1))
        depth_data = np.asanyarray(depth_img)
        cam_pc = one_cam_TSDF(color_img, depth_img, depth_intr, T_cam_base)
        target4_3d = one_cam_3d_gt(side_target, depth_data, depth_intr, cam_pc, T_cam_base)

        gt_dict = {'target_4': target4_3d}
        with open(folder_path + 'cam_{}_gt.pickle'.format(i + 1), 'wb') as f:
            pickle.dump(gt_dict, f)

    # two-cam method:
    two_cam_pc = two_cam_TSDF(folder_path, camera_poses)
    target4_3d = reconstruct(tar4_cam1_2d, tar4_cam2_2d, color_cam1_intr, color_cam2_intr, T_base_cam1, T_base_cam2)
    target4_3d = post_process(two_cam_pc, target4_3d)
    gt_dict = {'target_4': target4_3d}
    with open(folder_path + 'two_cam_gt.pickle', 'wb') as f:
        pickle.dump(gt_dict, f)


