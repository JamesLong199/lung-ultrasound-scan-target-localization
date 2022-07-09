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
import json
import pickle
from triangulation import *
import pyrealsense2 as rs

#### UR3e Robot Configuration



# Depth Camera Configuration

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print('device_product_line:', device_product_line)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# get camera intrinsic parameters
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
depth_intrinsics = depth_profile.get_intrinsics()
intr = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                 [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                 [0, 0, 1]])

print("Intrinsics: \n", intr)


# retrieve 2D openpose keypoints, and camera extrinsic matrices

# with open('OpenPose_UR_data/keypoints_json/cam_0_keypoints.json') as f:
#     data = json.load(f)
#     neck_0 = np.array(data['people'][0]['pose_keypoints_2d'][3:5]).reshape(-1,1)
#     print("neck_0: \n", neck_0)
#     print("Neck 0 Confidence: ", data['people'][0]['pose_keypoints_2d'][5])
#
# with open('OpenPose_UR_data/keypoints_json/cam_1_keypoints.json') as f:
#     data = json.load(f)
#     neck_1 = np.array(data['people'][0]['pose_keypoints_2d'][3:5]).reshape(-1,1)
#     print("neck_1: \n", neck_1)
#     print("Neck 1 Confidence: ", data['people'][0]['pose_keypoints_2d'][5])

with open('ViTPose_UR_data/keypoints/cam_0_keypoints.pickle','rb') as f:
    cam0_keypoints = pickle.load(f)
    r_shoulder0 = cam0_keypoints[0]['keypoints'][6]
    target_point0 = np.array(r_shoulder0[:2])
    confidence = r_shoulder0[2]
    print("target point 0: ", target_point0)
    print("target point 0 confidence: ", confidence)

with open('ViTPose_UR_data/keypoints/cam_1_keypoints.pickle','rb') as f:
    cam1_keypoints = pickle.load(f)
    r_shoulder1 = cam1_keypoints[0]['keypoints'][6]
    target_point1 = np.array(r_shoulder1[:2])
    confidence = r_shoulder1[2]
    print("target point 1: ", target_point1)
    print("target point 1 confidence: ", confidence)


# read extrinsics
with open('ViTPose_UR_data/extrinsics/cam_0_extrinsics.pickle','rb') as f:
    T_cam0 = pickle.load(f)
    print("T_cam0: \n", T_cam0)

with open('ViTPose_UR_data/extrinsics/cam_1_extrinsics.pickle','rb') as f:
    T_cam1 = pickle.load(f)
    print("T_cam1: \n", T_cam1)


# perform triangulation
X_target = reconstruct(target_point0, target_point1, intr, intr, T_cam0, T_cam1)

print("3D target coordinate in base frame: \n", X_target)  # target coordinate in base frame

# convert it to coordinate in the camera frame (first view)

# this must be the same as the first pose/view in ViTPose_UR_collect_data.py
tcp_pose_base = (0.37575, -0.00551, 0.44270, 2.952, -0.116, 0.194)

# values obtained with Charuco board 5/12/2022
t_cam_tcp = np.array([-0.02455914, -0.00687368, -0.01111772])
R_cam_tcp = np.array([
    [0.99966082, -0.02335906, -0.01151501],
    [0.02283641,  0.99878791, -0.04360292],
    [0.01251957,  0.04332517,  0.99898258]
])

# retrieve the transformation from tcp to base
T_tcp_base = np.asarray(m3d.Transform(tcp_pose_base).get_matrix())
T_cam_tcp = np.vstack([np.hstack([R_cam_tcp, t_cam_tcp.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

T_cam_base = T_tcp_base @ T_cam_tcp
T_base_cam = np.linalg.inv(T_cam_base)

X_target_base_h = to_homog(X_target)
X_target_cam_h = T_base_cam @ X_target_base_h
X_target_cam = from_homog(X_target_cam_h)

print("3D target coordinate in camera frame: \n", X_target_cam)  # target coordinate in camera frame
# TSDF volumn

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
               "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


camera_poses = read_trajectory("ViTPose_UR_data/odometry.log")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=1 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# show RGBD images from all the views
for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("data/color_images/cam_{}.jpg".format(i))
    depth = o3d.io.read_image("data/depth_images/cam_{}.png".format(i))

    depth_image = np.asanyarray(depth)
    color_image = np.asanyarray(color)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    plt.imshow(images)
    plt.show()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=613.22027588,
        fy=612.14581299,
        cx=318.16168213,
        cy=235.91072083
    )

    volume.integrate(rgbd,
                     intr,
                     np.linalg.inv(camera_poses[i].pose))


# point cloud generation
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.005)

downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd])
# o3d.visualization.draw_geometries([downpcd], point_show_normal=True)
coordinates = np.asarray(downpcd.points)
normals = np.asarray(downpcd.normals)

# need to convert robot base coordinate to camera coordinate
init_estimate = X_target_cam.squeeze()  # an initial estimate of the target point
idx = np.argmin(np.square(coordinates - init_estimate).sum(axis=1))
print('closest point in cam frame:', coordinates[idx, :])

target_normal = normals[idx]  # a unit vector
# print("target normal: ", target_normal)

t_target_cam = coordinates[idx]
# print("t_target_cam: ", t_target_cam)

# decode rotation from the target normal vector
# compute the rotation from (0,0,1)/camera's z-axis to the negative direction of the target normal
R_target_cam = rotation_align(np.array([0, 0, 1]), -target_normal)
# print("R_target_cam: \n", R_target_cam)

# using nearest point as target position
# T_target_cam = np.vstack([np.hstack([R_target_cam, t_target_cam.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

# using original target as target position
T_target_cam = np.vstack([np.hstack([R_target_cam, X_target_cam.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

T_target_base = T_tcp_base @ T_cam_tcp @ T_target_cam

target_6d_pose_base = m3d.Transform(T_target_base).get_pose_vector()
target_6d_pose_base[2] += 0.02
print("Final 6d pose base:  ", target_6d_pose_base)



# # UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value


robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))  # joint

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

robot.movej(pose=target_6d_pose_base, a=ACCELERATION, v=VELOCITY)

exit(0)