# Eye-to-hand calibrate one camera
# Obtain cam pose in base frame
# Serve as camera extrinsics and trajectory for TSDF volume integration

import pyrealsense2 as rs
import URBasic
import time
import numpy as np
import math3d as m3d
from utils.apriltag_utils.TagDetector import TagDetector
from utils.pose_conversion import *
from utils.apriltag_utils.annotate_tag import *
from utils.trajectory_io import *
import cv2
import argparse

robot_start_position = (np.radians(-355.54), np.radians(-181.98), np.radians(119.77),
                        np.radians(-28.18), np.radians(91.45), np.radians(357.25))  # joint

# eye-to-hand robot poses
all_poses = [
            # (0.2179, 0.2306, 0.6086, 0.854, 0.573, 4.059),
            # (0.2441, 0.2899, 0.4071, 0.151, 0.321, -0.514),
            # (0.0827, 0.2636, 0.4934, 0.338, 0.748, 0.744),
            # (0.0228, 0.3971, 0.3023, 0.003, 0.250, 0.777),
            # (0.3135, 0.2506, 0.3338, 2.467, -0.975, -5.423),
            # (0.2574, 0.1648, 0.6322, 0.116, -2.839, -5.484),
            # (0.2329, 0.2061, 0.7154, 0.045, 0.298, -0.621),
            # (0.0959, 0.3046, 0.5151, 0.213, -0.238, -0.868),
            (-0.0486, 0.4036, 0.3672, 5.29, -3, 1.055),

            # (0.166, 0.477, 0.342, 0.319, 0.784, 0.248),
            (0.224, 0.404, 0.506, 0.438, 0.600, 0.080),
            (0.0643, 0.4696, 0.4764, 0.303, 0.359, 0.132),
            ]

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
robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

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

parser = argparse.ArgumentParser(description="""Eye-to-hand calibration algorithm. """)
parser.add_argument("--type", type=str, default="eye_to_hand", choices=("eye_in_hand","eye_to_hand"),
                    help="eye-in-hand calibration or eye-to-hand calibration")
parser.add_argument('--cam', type=int, default=1, help='Camera 1 or camera 2')


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
            print("tag translation: ", result.pose_t.squeeze())
            annotate_tag(result, color_image)
            show_frame(color_image)

            R_tag_cam_list.append(result.pose_R)
            t_tag_cam_list.append(result.pose_t.squeeze())
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

    # robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)
    robot.close()

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_tcp_base_list,
        t_gripper2base=t_tcp_base_list,
        R_target2cam=R_tag_cam_list,
        t_target2cam=t_tag_cam_list,
    )
    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[0:3,3] = t.squeeze()
    T[3,3] = 1.

    print("Hand-eye calibration T: \n", T)

    write_to_file('data/odometry.log', args.cam-1, R, t.squeeze())

    exit(0)
