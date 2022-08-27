# Eye-to-hand calibrate one camera
# Obtain cam pose in base frame
# Serve as camera extrinsics and trajectory for TSDF volume integration

import pyrealsense2 as rs
import URBasic
import time
import math3d as m3d
from utils.apriltag_utils.TagDetector import TagDetector
from utils.apriltag_utils.annotate_tag import *
from utils.trajectory_io import *
import argparse

parser = argparse.ArgumentParser(description="""Eye-to-hand calibration algorithm. """)
parser.add_argument("--type", type=str, default="eye_to_hand", choices=("eye_in_hand","eye_to_hand"),
                    help="eye-in-hand calibration or eye-to-hand calibration")
parser.add_argument('--cam', type=int, default=2, help='Camera 1 or camera 2')


args = parser.parse_args()
TYPE = args.type


def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))


def invsqrt(mat):
    u,s,v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)


def calibrate(A, B):
    #transform pairs A_i, B_i
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += np.outer(log(Rb), log(Ra))

    Rx = np.dot(invsqrt(np.dot(M.T, M)), M.T)

    C = np.zeros((3*N, 3))
    d = np.zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - np.dot(Rx, tb)

    tx = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))

    T = np.vstack( (np.hstack((Rx, tx)), [0,0,0,1]) )
    return T


robot_start_position = (np.radians(-355.54), np.radians(-181.98), np.radians(119.77),
                        np.radians(-28.18), np.radians(91.45), np.radians(357.25))  # joint

# eye-to-hand robot poses

# # robot poses for 1st camera (left)
# all_poses = [
#             (0.2441, 0.2899, 0.4071, 0.151, 0.321, -0.514),
#             (0.0827, 0.2636, 0.4934, 0.338, 0.748, 0.744),
#             (0.0228, 0.3971, 0.3023, 0.003, 0.250, 0.777),
#             (0.3135, 0.2506, 0.3338, 2.467, -0.975, -5.423),
#             (0.2574, 0.1648, 0.6322, 0.116, -2.839, -5.484),
#             (0.0959, 0.3046, 0.5151, 0.213, -0.238, -0.868),
#             (-0.0486, 0.4036, 0.3672, 5.29, -3, 1.055),
#             (0.166, 0.477, 0.342, 0.319, 0.784, 0.248),
#             (0.224, 0.404, 0.506, 0.438, 0.600, 0.080),
#             (0.0643, 0.4696, 0.4764, 0.303, 0.359, 0.132),
#             (0.086, 0.539, 0.474, 1.015, -1.746, 5.062),
#             (0.179, 0.327, 0.638, 0.497, 0.204, -0.729),
#             (0.287, 0.281, 0.741, 0.009, 0.473, 1.758),
#             (0.209, 0.425, 0.587, 0.182, 0.230, -1.062),
#             (0.127, 0.424, 0.587, 0.067, 0.827, -0.495),
#             (0.146, 0.354, 0.477, 0.160, -0.035, -0.198),
#             (0.252, 0.406, 0.353, 0.027, -0.12, -0.205),
#             (0.207, -0.07, 0.637, 0.592, 0.882, -1.184),
#             (0.245, 0.01, 0.597, 0.599, 0.471, -0.045),
#             (0.235, 0.240, 0.533, 0.064, 0.405, -0.171),
#             (0.074, 0.242, 0.446, 0.221, -0.415, 2.109),
#             (0.0598, 0.614, 0.354, 2.824, -5.020, -0.674),
#             (0.124, 0.576, 0.508, 1.738, -2.866, 4.556),
#             (0.292, 0.546, 0.500, 1.810, -3.516, -4.316),
#             (0.172, 0.581, 0.604, 4.094, -3.219, -2.085),
#             (0.358, 0.524, 0.430, 1.865, -0.368, 4.819),
#             (0.318, 0.590, 0.306, 1.295, 0.656, 4.410),
#             (0.211, 0.576, 0.615, 0.568, 0.612, 2.760),
#             (0.203, 0.562, 0.580, 0.651, 0.326, 2.747),
#             (0.228, 0.582, 0.620, 3.454, -3.425, -2.419),
#             (0.179, 0.593, 0.599, 2.945, -4.647, -1.266),
#             (0.190, 0.640, 0.526, 3.637, 0.363, 4.043),
#             (0.225, 0.618, 0.304, 4.319, -1.410, 3.434),
#             (0.387, 0.562, 0.437, 2.159, 0.540, 4.439),
#             (0.169, 0.618, 0.383, 0.977, -0.697, -3.714),
#             (0.341, 0.451, 0.282, 1.119, -0.449, -4.228),
#             (0.216, 0.491, 0.501, 2.778, -3.607, -3.253),
#             (0.202, 0.608, 0.394, 1.492, 1.518, 4.461),
#             (0.190, 0.327, 0.464, 0.477, 0.136, 2.174),
#             (0.343, 0.354, 0.517, 0.345, 0.525, 2.613),
#             ]

# robot poses for 2nd camera (right)
all_poses = [
            (0.2441, 0.2899, 0.4071, 0.151, 0.321, -0.514),
            (0.0827, 0.2636, 0.4934, 0.338, 0.748, 0.744),
            (0.0228, 0.3971, 0.3023, 0.003, 0.250, 0.777),
            (0.3135, 0.2506, 0.3338, 2.467, -0.975, -5.423),
            (0.0959, 0.3046, 0.5151, 0.213, -0.238, -0.868),
            (0.166, 0.477, 0.342, 0.319, 0.784, 0.248),
            (0.224, 0.404, 0.506, 0.438, 0.600, 0.080),
            (0.0643, 0.4696, 0.4764, 0.303, 0.359, 0.132),
            (0.179, 0.327, 0.638, 0.497, 0.204, -0.729),
            (0.209, 0.425, 0.587, 0.182, 0.230, -1.062),
            (0.146, 0.354, 0.477, 0.160, -0.035, -0.198),
            (0.252, 0.406, 0.353, 0.027, -0.12, -0.205),
            (0.253, 0.429, 0.308, 0.118, -0.161, -1.382),
            (0.235, 0.240, 0.533, 0.064, 0.405, -0.171),
            (0.074, 0.242, 0.446, 0.221, -0.415, 2.109),
            (0.0598, 0.614, 0.354, 2.824, -5.020, -0.674),
            (0.124, 0.576, 0.508, 1.738, -2.866, 4.556),
            (0.292, 0.546, 0.500, 1.810, -3.516, -4.316),
            (0.358, 0.524, 0.430, 1.865, -0.368, 4.819),
            (0.318, 0.590, 0.306, 1.295, 0.656, 4.410),
            (0.211, 0.576, 0.615, 0.568, 0.612, 2.760),
            (0.203, 0.562, 0.580, 0.651, 0.326, 2.747),
            (0.228, 0.582, 0.620, 3.454, -3.425, -2.419),
            (0.225, 0.618, 0.304, 4.319, -1.410, 3.434),
            (0.387, 0.562, 0.437, 2.159, 0.540, 4.439),
            (0.169, 0.618, 0.383, 0.977, -0.697, -3.714),
            (0.341, 0.451, 0.282, 1.119, -0.449, -4.228),
            (0.216, 0.491, 0.501, 2.778, -3.607, -3.253),
            (0.202, 0.608, 0.394, 1.492, 1.518, 4.461),
            (0.190, 0.327, 0.464, 0.477, 0.136, 2.174),
            (0.343, 0.354, 0.517, 0.345, 0.525, 2.613),
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

print('device_product_line:', device_product_line)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# get camera intrinsic parameters
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()
intr = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                 [0, color_intrinsics.fy, color_intrinsics.ppy],
                 [0, 0, 1]])

# AprilTag detector configuration
tag_size = 0.072
tag_family = 'tagStandard41h12'
cam_type = "standard"
tagDetector = TagDetector(intr, None, tag_family, cam_type)
print("AprilTag detector prepared!")
time.sleep(1)


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff


T_tcp_base_list = []
T_tag_cam_list = []

try:
    for i, pose in enumerate(all_poses):

        print(f"{i}th pose: ", pose)

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
            # print("tag translation: ", result.pose_t.squeeze())
            annotate_tag(result, color_image)
            show_frame(color_image)

            T_tag_cam[0:3, 0:3] = result.pose_R
            T_tag_cam[0:3, 3] = result.pose_t.squeeze()
            T_tag_cam[3, 3] = 1

            T_tag_cam_list.append(T_tag_cam)
        else:
            print("No Tag detected")
            continue

        T_tcp_base = np.asarray(m3d.Transform(pose).get_matrix())

        if TYPE == "eye_to_hand":
            T_tcp_base = np.linalg.inv(T_tcp_base)   # T_base_tcp

        T_tcp_base_list.append(T_tcp_base)


finally:
    # Stop streaming
    pipeline.stop()

    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)
    robot.close()

    A_list = []  # store 4x4 numpy array
    B_list = []

    assert len(T_tag_cam_list) == len(T_tcp_base_list)
    for i, (T_tag_cam1, T_tcp_base1) in enumerate(zip(T_tag_cam_list, T_tcp_base_list)):
        for (T_tag_cam2, T_tcp_base2) in zip(T_tag_cam_list[i+1:], T_tcp_base_list[i+1:]):
            A = np.linalg.inv(T_tcp_base2) @ T_tcp_base1
            B = T_tag_cam2 @ np.linalg.inv(T_tag_cam1)

            A_list.append(A)
            B_list.append(B)

    T = calibrate(A_list, B_list)
    print("Hand-eye calibration T: \n", T)

    R = T[0:3, 0:3]
    t = T[0:3, 3]
    write_to_file('data/odometry.log', args.cam-1, R, t.squeeze())

    exit(0)
