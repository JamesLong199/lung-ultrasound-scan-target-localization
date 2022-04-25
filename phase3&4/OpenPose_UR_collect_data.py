# Manipulate UR3e to move in a designated circular path to scan and
# take pictures of the mannequin to reconstruct its surface using UR's base
# change the transformation matrix of the first camera to be identity
# use the manually measured camera offset in TCP frame

import pyrealsense2 as rs
import URBasic
import time
from scipy.spatial.transform import Rotation as R
from utils.apriltag_utils.annotate_tag import *
import math3d as m3d
import pickle

# UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value


robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))  # joint

# define two tool poses (in base frame) here
all_poses = [ (0.37575, -0.00551, 0.44270, 2.952, -0.116, 0.194),
              (0.30761, -0.07658, 0.53570, 2.986, 0.243, 0.753)]


# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1)  # just a short wait to make sure everything is initialised

# hand-eye calibration: much more accurate!!!
t_cam_tcp = np.array([-0.02847469, 0.00485245, 0.04717332])
R_cam_tcp = np.array([
    [0.99965676,  0.01999816, -0.0169242],
    [-0.02057245,  0.99919404, -0.03446816],
    [0.01622126,  0.0348045,   0.99926249]
])

T_cam_tcp = np.zeros((4,4))
T_cam_tcp[0:3,0:3] = R_cam_tcp
T_cam_tcp[0:3,3] = t_cam_tcp
T_cam_tcp[3,3] = 1

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

time.sleep(0.5)


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff


def write_to_file(i, transformation):
    with open('OpenPose_UR_data/extrinsics/cam_{}_extrinsics.pickle'.format(i), 'wb') as f:
        pickle.dump(transformation, f)

#### Start circling:

try:
    for i, pose in enumerate(all_poses):
        robot.movej(pose=pose, a=ACCELERATION, v=VELOCITY)
        time.sleep(1)
        # take pictures:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        print('capture number {} completed'.format(i))

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        show_frame(color_image)

        cv.imwrite('OpenPose_UR_data/color_images/cam_{}.jpg'.format(i), color_image)
        cv.imwrite('../../openpose/python/OpenPose_UR/images/cam_{}.jpg'.format(i), color_image)

        T_tcp_base = np.asarray(m3d.Transform(pose).get_matrix())
        T_cam_base = T_tcp_base @ T_cam_tcp  # coordinates in cam frame to coordinates in base frame

        R_cam_base = T_cam_base[0:3, 0:3]
        t_cam_base = T_cam_base[0:3, 3]

        # compute camera extrinsic matrix: coordinates in base frame to coordinates in camera frame
        R_base_cam = T_cam_base[0:3, 0:3].T
        t_base_cam = R_base_cam @ (-t_cam_base)

        T_base_cam = np.zeros((4, 4))
        T_base_cam[0:3, 0:3] = R_base_cam
        T_base_cam[0:3, 3] = t_base_cam
        T_base_cam[3, 3] = 1

        write_to_file(i, T_base_cam)

finally:
    # Stop streaming
    pipeline.stop()
    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

    robot.close()
    exit(0)