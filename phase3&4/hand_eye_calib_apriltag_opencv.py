# eye-in-hand calibration with Apriltag
# Collect different sets of matrix A and B
# Reference: https://www.torsteinmyhre.name/snippets/robcam_calibration.html

import pyrealsense2 as rs
import URBasic
import time
import numpy as np
import math3d as m3d
from utils.apriltag_utils.TagDetector import TagDetector
from utils.pose_conversion import *
from utils.apriltag_utils.annotate_tag import *
import cv2
import argparse


all_poses = [(0.514, 0, 0.045, 2.523, 1.829, 0.042),
             (0.474, 0.160, 0.024, 1.75, 2.342, -0.286),
             (0.443, -0.056, 0.022, 1.157, 2.979, 0.465),
             (0.45, 0.133, -0.128, 2.537, 1.256, 0.023),
             (0.346, 0.225, 0.012, 1.907, 1.925, -0.155)]




# UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)
robot.init_realtime_control()
time.sleep(1)

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
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
intr = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                 [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                 [0, 0, 1]])

# AprilTag detector configuration
tag_size = 0.048
tag_family = 'tagStandard41h12'
cam_type = "standard"
tagDetector = TagDetector(intr, None, tag_family, cam_type)
print("AprilTag detector prepared!")
time.sleep(1)


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff

parser = argparse.ArgumentParser(description="""Training CycleGAN model. """)
parser.add_argument("--type", type=str, default="eye_in_hand", choices=("eye_in_hand","eye_to_hand"),
                    help="eye-in-hand calibration or eye-to-hand calibration")

args = parser.parse_args()
TYPE = args.type

A_list = []  # store 4x4 numpy array
B_list = []

R_tcp_base_list = []
t_tcp_base_list = []
R_tag_cam_list = []
t_tag_cam_list = []

try:
    for i, pose in enumerate(all_poses):

        T_tag_cam = None
        T_tcp_base = None

        robot.movej(pose=pose, a=ACCELERATION, v=VELOCITY)
        time.sleep(1)

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        show_frame(color_image)

        # Locate camera with AprilTag
        _, detection_results = tagDetector.detect_tags(color_image, tag_size)

        T_tag_cam = np.zeros((4, 4))
        if len(detection_results) != 0:
            result = detection_results[0]  # use only one tag!
            annotate_tag(result, color_image)
            show_frame(color_image)

            R_tag_cam_list.append(result.pose_R)
            t_tag_cam_list.appned(result.pose_t.squeeze())
        else:
            print("No Tag detected")

        T_tcp_base = np.asarray(m3d.Transform(pose).get_matrix())

        if TYPE == "eye_to_hand":
            T_tcp_base = np.linalg.inv(T_tcp_base)   # T_base_tcp

        R_tcp_base = T_tcp_base[0:3, 0:3]
        t_tcp_base = T_tcp_base[0:3, 3]
        R_tcp_base_list.append(R_tcp_base)
        t_tcp_base_list.append(t_tcp_base)



finally:
    # Stop streaming
    pipeline.stop()

    robot.close()

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )
    T = np.zeros(4,4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    T[3,3] = 1

    print("Hand-eye calibration T: \n", T)

    exit(0)
