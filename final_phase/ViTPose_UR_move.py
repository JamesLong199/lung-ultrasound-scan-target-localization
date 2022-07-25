# Perform 2D to 3D mapping
# Given camera intrinsics, extrinsics, 2D coordinates in two views
# Compute the corresponding 3D coordinates
# navigate the robot to that 3D coordinates

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math3d as m3d
import URBasic
import time
from utils.pose_conversion import *
from utils.trajectory_io import *
import json
import pickle
from triangulation import *

from subject_info import SUBJECT_NAME, SCAN_POSE
import argparse

parser = argparse.ArgumentParser(description='Compute target')
parser.add_argument('--pose_model', type=str, default='ViTPose_large', help='pose model')
args = parser.parse_args()
POSE_MODEL = args.pose_model

folder_path = 'final_phase/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'

# read intrinsics
with open(folder_path + 'intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
    cam1_intr = pickle.load(f)
    print("cam1_intr: \n", cam1_intr)

with open(folder_path + 'intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
    cam2_intr = pickle.load(f)
    print("cam2_intr: \n", cam2_intr)

# read extrinsics
camera_poses = read_trajectory(folder_path + "odometry.log")

# TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=1 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# show RGBD images from all the views
for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(i+1))
    depth = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(i+1))

    # depth_image = np.asanyarray(depth)
    # color_image = np.asanyarray(color)
    #
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
    # depth_colormap_dim = depth_colormap.shape
    # color_colormap_dim = color_image.shape
    #
    # # If depth and color resolutions are different, resize color image to match depth image for display
    # if depth_colormap_dim != color_colormap_dim:
    #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
    #                                      interpolation=cv2.INTER_AREA)
    #     images = np.hstack((resized_color_image, depth_colormap))
    # else:
    #     images = np.hstack((color_image, depth_colormap))

    # Show images
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # plt.imshow(images)
    # plt.show()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    cam_intr = cam1_intr if i == 0 else cam2_intr

    intr = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=cam_intr[0,0],
        fy=cam_intr[1,1],
        cx=cam_intr[0,2],
        cy=cam_intr[1,2]
    )

    volume.integrate(rgbd,
                     intr,
                     np.linalg.inv(camera_poses[i].pose))
    # break

# point cloud generation
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.01)

downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([downpcd])
# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)
coordinates = np.asarray(downpcd.points)
normals = np.asarray(downpcd.normals)

# UR3e configuration
ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value

# robot_start_position = (np.radians(-355.54), np.radians(-181.98), np.radians(119.77),
#                         np.radians(-28.18), np.radians(91.45), np.radians(357.25))  # joint
# robot_start_position = (np.radians(-269.76), np.radians(-5.37), np.radians(-82.09),
#                         np.radians(-2.54), np.radians(90.21), np.radians(221.59))  # joint

robot_start_position = (np.radians(-99.73), np.radians(-171.50), np.radians(85.98),
                        np.radians(-1.81), np.radians(90.23), np.radians(221.59))   # front

robot_waypoint = (np.radians(98.57), np.radians(-113.38), np.radians(-38.86),
                 np.radians(-90.), np.radians(91.28), np.radians(155.32))

# robot_start_position = (np.radians(-99.73), np.radians(-171.50), np.radians(85.98),
#                         np.radians(-1.81), np.radians(90.23), np.radians(221.59))  # side


# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)
robot.init_realtime_control()
time.sleep(1)
robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)


with open(folder_path + POSE_MODEL + '/final_target.pickle', 'rb') as f:
    X_targets =  pickle.load(f)

for i, X_target in enumerate(X_targets):
    print("========================================")
    print(f"{i}th target point: {X_target.squeeze()}")

    # need to convert robot base coordinate to camera coordinate
    init_estimate = X_target.squeeze()  # an initial estimate of the target point
    # idx = np.argmin(np.square(coordinates - init_estimate).sum(axis=1))
    idx = np.argmin(np.square(coordinates[:, 0:2] - init_estimate[0:2]).sum(axis=1))

    t_target_base = coordinates[idx]
    target_normal_base = normals[idx]

    print('closest point in base frame:', t_target_base)
    print('closest point normal in base frame:', target_normal_base)

    TCP_6d_pose_base = robot.get_actual_tcp_pose()
    print("robot pose: ", TCP_6d_pose_base)

    # Want to align the orientation of TCP to be (0,0,1)
    T_tcp_base = np.asarray(m3d.Transform(TCP_6d_pose_base).get_matrix())  # 4x4 matrix
    R_tcp_base = T_tcp_base[0:3, 0:3]
    R_base_tcp = np.linalg.inv(R_tcp_base)

    target_normal_tcp = R_base_tcp @ target_normal_base.reshape(-1,1)

    # compute the rotation from (0,0,1)/tcp's z-axis to the negative direction of the target normal in tcp frame
    R_target_tcp = rotation_align(np.array([0, 0, 1]), -target_normal_tcp.squeeze())

    R_target_base = R_tcp_base @ R_target_tcp
    T_target_base = np.vstack([np.hstack([R_target_base, t_target_base.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

    target_6d_pose_base = m3d.Transform(T_target_base).get_pose_vector()
    print("Final 6d pose base: ", target_6d_pose_base)

    robot.movej(q=robot_waypoint, a=ACCELERATION, v=VELOCITY)
    robot.movej(pose=target_6d_pose_base, a=ACCELERATION, v=VELOCITY)

    time.sleep(3)
    robot.movej(q=robot_waypoint, a=ACCELERATION, v=VELOCITY)
    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)


robot.close()
# exit(1)
exit(0)

