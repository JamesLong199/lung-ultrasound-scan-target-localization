# Perform 2D to 3D mapping
# Given camera intrinsics, extrinsics, 2D coordinates in two views
# Compute the corresponding 3D coordinates
# navigate the robot to that 3D coordinates

import open3d as o3d
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


# retrieve 2D openpose_lib keypoints, and camera extrinsic matrices

with open('OpenPose_UR_data/keypoints_json/cam_0_keypoints.json') as f:
    data = json.load(f)
    neck_0 = np.array(data['people'][0]['pose_keypoints_2d'][3:5]).reshape(-1,1)
    print("neck_0: \n", neck_0)
    print("Neck 0 Confidence: ", data['people'][0]['pose_keypoints_2d'][5])

with open('OpenPose_UR_data/keypoints_json/cam_1_keypoints.json') as f:
    data = json.load(f)
    neck_1 = np.array(data['people'][0]['pose_keypoints_2d'][3:5]).reshape(-1,1)
    print("neck_1: \n", neck_1)
    print("Neck 1 Confidence: ", data['people'][0]['pose_keypoints_2d'][5])

# read extrinsics
with open('OpenPose_UR_data/extrinsics/cam_0_extrinsics.pickle','rb') as f:
    T_cam0 = pickle.load(f)
    print("T_cam0: \n", T_cam0)

with open('OpenPose_UR_data/extrinsics/cam_1_extrinsics.pickle','rb') as f:
    T_cam1 = pickle.load(f)
    print("T_cam1: \n", T_cam1)


# perform triangulation
X_neck = reconstruct(neck_0, neck_1, intr, intr, T_cam0, T_cam1)

print("3D neck coordinate: \n", X_neck)
