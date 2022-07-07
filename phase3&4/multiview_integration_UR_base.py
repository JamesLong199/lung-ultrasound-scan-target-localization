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

# UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value


robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))  # joint
# robot_start_position = (0.435, 0, 0.19, 3.142, 0, 0) # pose in base frame

# define two tool poses (in base frame) here

# only rotation
# all_poses = [ (0.43414, 0.11073, 0.34996, 2.965, 0.326, 0.215),
#               (0.43414, 0.01073, 0.34996, 2.708, 1.377, 0.286)]

# all_poses = [ (0.48615, 0.17334, 0.22594, 3.179, -0.191, 0.028),
#               (0.40051, -0.07548, 0.34226, 3.387, 0.073, 0.241)]

# vertical pose
# all_poses = [(0.394, 0.261, 0.049, 2.31, 1.016, -0.432),
#              (0.394, 0.261, 0.049, 2.31, 1.016, -0.432)]

all_poses = [ (0.38011, 0.00178, 0.4396, 3.153, 0.029, 0.38),
              (0.33711, 0.22542, 0.41898, 2.446, 0.975, 0.411) ]

# all_poses = [(0.22297, 0.15508, 0.28471, 1.898, 2.080, 0.311),
#               (0.38406, 0.30548, 0.27386, 1.072, 2.684, -0.248)]

# all_poses = [(0.37402, -0.10703, 0.44335, 2.156, 1.942, 0.307),
#              (0.42078, 0.06133, 0.37028, 2.163, 2.077, 0.276),
#              (0.38173, 0.19213, 0.35512, 1.519, 2.203, 0.003),
#             (0.31991, 0.25781, 0.35438, 1.104, 2.294, -0.148),
#             (0.34995, 0.30566, 0.29499, 0.442, 2.675, -0.345)
#              ]

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
# time.sleep(5)

# manually measured camera offset in TCP frame

# transformation from camera frame to tcp frame

# manually estimated offsets
# t_cam_tcp = np.array([-0.041, -0.002, 0.02])
# R_cam_tcp = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# hand-eye calibration: much more accurate!!!
# t_cam_tcp = np.array([-0.02847469, 0.00485245, 0.04717332])
# R_cam_tcp = np.array([
#     [0.99965676,  0.01999816, -0.0169242],
#     [-0.02057245,  0.99919404, -0.03446816],
#     [0.01622126,  0.0348045,   0.99926249]
# ])

# new values obtained 5/4/2022
# t_cam_tcp = np.array([-0.02763843, 0.00309861, 0.02161887])
# R_cam_tcp = np.array([
#     [0.99974095,  -0.02216273, -0.00518159],
#     [0.02204299,  0.9995122, -0.02212383],
#     [0.00566939,  0.02200388,   0.99974181]
# ])

# t_cam_tcp = np.array([-0.03696902, -0.00744645, 0.02541028])
# R_cam_tcp = np.array([
#     [0.99940741, -0.03188109,  0.01297763],
#     [0.03171749,  0.99941716,  0.01262278],
#     [-0.01337249, -0.01220368,  0.99983611]
# ])

# new values obtained 5/11/2022
# t_cam_tcp = np.array([-0.03396755, -0.00100364, 0.01737927])
# R_cam_tcp = np.array([
#     [0.99989021, -0.01452523, -0.00292946],
#     [0.01437072,  0.99878044, -0.04723476],
#     [0.00361199,  0.04718747,  0.99887952]
# ])

# values obtained with Charuco board 5/12/2022
# t_cam_tcp = np.array([-0.02455914, -0.00687368, -0.01111772])
# R_cam_tcp = np.array([
#     [0.99966082, -0.02335906, -0.01151501],
#     [0.02283641,  0.99878791, -0.04360292],
#     [0.01251957,  0.04332517,  0.99898258]
# ])

# another values obtained with Charuco board 5/12/2022
# t_cam_tcp = np.array([-0.02551455, -0.00849935, -0.01734741])
# R_cam_tcp = np.array([
#     [0.99989235, -0.00995085, -0.0107831],
#     [0.00963223,  0.99952694, -0.02920807],
#     [0.01106864,  0.02910106,  0.99951519]
# ])

# 6/22/2022: new script
# t_cam_tcp = np.array([-0.02455366, -0.00223298, 0.02301918])
# R_cam_tcp = np.array([
#     [0.99900937,  0.03935652, -0.02076871],
#     [-0.03967849,  0.99909497, -0.01532501],
#     [0.02014678,  0.0161339,   0.99966685]
# ])

# 6/22/2022: old script
t_cam_tcp = np.array([-0.03454292, -0.00486638, 0.00510053])
R_cam_tcp = np.array([
    [0.99798864, -0.06332879, -0.00285177],
    [0.06329785,  0.9979458,  -0.00987775],
    [0.00347146,  0.00967737,  0.99994715]
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

T_cam_base_list = []

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

        cv.imwrite('RGBD/color/{}.jpg'.format(i), color_image)
        cv.imwrite('RGBD/depth/{}.png'.format(i), depth_image)

        T_tcp_base = np.asarray(m3d.Transform(pose).get_matrix())
        T_cam_base = T_tcp_base @ T_cam_tcp
        T_cam_base_list.append(T_cam_base)

    for i,T_cam_base in enumerate(T_cam_base_list):
        T_cam2_cam1 = np.linalg.inv(T_cam_base_list[0]) @ T_cam_base_list[i]
        R_cam2_cam1 = T_cam2_cam1[0:3,0:3]
        t_cam2_cam1 = T_cam2_cam1[0:3, 3]
        write_to_file(i, R_cam2_cam1, t_cam2_cam1)

finally:
    # Stop streaming
    pipeline.stop()
    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

    robot.close()
    exit(0)