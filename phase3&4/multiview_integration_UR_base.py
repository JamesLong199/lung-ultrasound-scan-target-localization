# Manipulate UR3e to move in a designated circular path to scan and
# take pictures of the mannequin to reconstruct its surface using UR's base
# change the transformation matrix of the first camera to be identity
# use the manually measured camera offset in TCP frame

import pyrealsense2 as rs
import URBasic
import time
from scipy.spatial.transform import Rotation as R
from utils.apriltag_utils.annotate_tag import *

# UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value


robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))  # joint
# robot_start_position = (0.435, 0, 0.19, 3.142, 0, 0) # pose in base frame

# define two tool poses (in base frame) here

# only rotation
all_poses = [ (0.43414, 0.11073, 0.34996, 2.965, 0.326, 0.215),
              (0.43414, 0.01073, 0.34996, 2.708, 1.377, 0.286)]

# vertical pose
# all_poses = [(0.394, 0.261, 0.049, 2.31, 1.016, -0.432),
#              (0.394, 0.261, 0.049, 2.31, 1.016, -0.432)]

# all_poses = [ (0.38011, 0.00178, 0.4396, 3.153, 0.029, 0.38),
#               (0.33711, 0.22542, 0.41898, 2.446, 0.975, 0.411) ]

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

# manually measured camera offset in TCP frame
cam_t_tcp = np.array([-0.041, -0.002, 0.02])
cam_R_tcp = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


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


def write_to_file(i, rot_mat, translation):
    file1 = open('RGBD/odometry.log', 'a')  # 'a' means append
    L1 = str(i) + '   ' + str(i) + '   ' + str(i + 1) + '\n'
    L2 = str(rot_mat[0, 0]) + '   ' + str(rot_mat[0, 1]) + '   ' + str(rot_mat[0, 2]) + '   ' + str(
        translation[0]) + '\n'
    L3 = str(rot_mat[1, 0]) + '   ' + str(rot_mat[1, 1]) + '   ' + str(rot_mat[1, 2]) + '   ' + str(
        translation[1]) + '\n'
    L4 = str(rot_mat[2, 0]) + '   ' + str(rot_mat[2, 1]) + '   ' + str(rot_mat[2, 2]) + '   ' + str(
        translation[2]) + '\n'
    L5 = "0   0   0   1\n"
    file1.write(L1)
    file1.write(L2)
    file1.write(L3)
    file1.write(L4)
    file1.write(L5)
    file1.close()


#### Start circling:
cam1_R_base, cam2_R_base = None, None
cam1_t_base, cam2_t_base = None, None

# make the first camera view identity transformation
write_to_file(0, np.eye(3), np.array([0, 0, 0]))

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

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        show_frame(color_image)

        cv.imwrite('RGBD/color/{}.jpg'.format(i), color_image)
        cv.imwrite('RGBD/depth/{}.png'.format(i), depth_image)

        # compute extrinsic parameters in the base frame and update the odometry.log file
        tcp_t_base = np.array(pose[:3])
        tcp_R_base = R.from_rotvec(pose[3:]).as_matrix()

        cam_t_base = tcp_R_base @ cam_t_tcp + tcp_t_base
        cam_R_base = tcp_R_base @ cam_R_tcp

        if i == 0:
            cam1_R_base = cam_R_base
            cam1_t_base = cam_t_base
        elif i == 1:
            cam2_R_base = cam_R_base
            cam2_t_base = cam_t_base

    relative_R = cam1_R_base.T @ cam2_R_base  # relative rotation
    relative_t = cam1_R_base.T @ (cam2_t_base - cam1_t_base)  # relative translation
    write_to_file(1, relative_R, relative_t)

finally:
    # Stop streaming
    pipeline.stop()
    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

    robot.close()
    exit(0)