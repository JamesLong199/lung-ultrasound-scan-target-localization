# eye-in-hand calibration with Apriltag
# Collect different sets of matrix A and B
# Reference:
# https://www.torsteinmyhre.name/snippets/robcam_calibration.html
# https://github.com/kyle-bersani/opencv-examples/blob/master/PostureEstimation/EstimatePostureWithCharucoBoard.py

import pyrealsense2 as rs
import URBasic
import time
import numpy as np
import math3d as m3d
import os
import pickle
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from utils.apriltag_utils.TagDetector import TagDetector
from utils.pose_conversion import *
from utils.apriltag_utils.annotate_tag import *

# all_poses = [(0.514, 0, 0.045, 2.523, 1.829, 0.042),
#              (0.474, 0.160, 0.024, 1.75, 2.342, -0.286),
#              (0.443, -0.056, 0.022, 1.157, 2.979, 0.465),
#              (0.45, 0.133, -0.128, 2.537, 1.256, 0.023),
#              (0.346, 0.225, 0.012, 1.907, 1.925, -0.155)]

robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-60.35),
                        np.radians(-102.05), np.radians(84.56), np.radians(112.04))  # joint


# all_poses = [(0.411, 0.277, -0.139, 2.421, 0.228, 0.350),
#              (0.484, 0.073, -0.119, 2.333, 2.295, -0.596),
#              (0.464, 0.048, 0.084, 0.068, 3.587, -0.259),
#              (0.425, -0.005, 0.196, 1.162, 3.027, -0.152),
#              (0.429, -0.081, 0.017, 3.219, 0.819, -0.384),
#              (0.442, 0.026, 0.130, 2.244, 2,179, 0.138)]

all_poses = [(0.44497, 0.14524, 0.34179, 2.263, 1.865, -0.011),
             (0.36152, 0.31837, 0.28701, 0.878, 2.638, -0.321),
             (0.40286, 0.34352, 0.10117, 0.816, 2.562, -0.580),
             (0.42503, -0.07771, 0.19552, 2.187, 2.249, 0.363),
             (0.51702, 0.08467, 0.20872, 0.304, 3.076, 0.026)]

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


# UR Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

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
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))  # forgot to use (rs.stream.color)
depth_intrinsics = depth_profile.get_intrinsics()
intr = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                 [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                 [0, 0, 1]])

# CharUco configuration
# Check for camera calibration data

cameraMatrix = np.array( [[613.22027588,   0.,         318.16168213],
                         [  0.,         612.14581299, 235.91072083],
                         [  0.,           0.,           1.        ]]
                        )

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

print("camera matrix: \n", cameraMatrix)
print("distCoeffs: \n", distCoeffs)
# f.close()
if cameraMatrix is None or distCoeffs is None:
    print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
    exit()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_250)
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 12

# Create grid board object we're using in our stream
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=0.060,
    markerLength=0.044,
    dictionary=ARUCO_DICT)

time.sleep(1)


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff


A_list = []  # store 4x4 numpy array
B_list = []

try:
    for i, pose1 in enumerate(all_poses):
        for j, pose2 in enumerate(all_poses[i+1:]):

            T_tag_cam1, T_tag_cam2 = None, None
            T_tcp1_base, T_tcp2_base = None, None

            for z, pose in enumerate([pose1, pose2]):

                robot.movej(pose=pose, a=ACCELERATION, v=VELOCITY)
                time.sleep(1)

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print("no frame")
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                show_frame(color_image)

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                # Locate camera with CharUco
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

                # Refine detected markers
                # Eliminates markers not part of our board, adds missing markers to the board
                corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                    image=gray,
                    board=CHARUCO_BOARD,
                    detectedCorners=corners,
                    detectedIds=ids,
                    rejectedCorners=rejectedImgPoints,
                    cameraMatrix=cameraMatrix,
                    distCoeffs=distCoeffs)

                # Outline all of the markers detected in our image
                color_image = aruco.drawDetectedMarkers(color_image, corners, borderColor=(0, 0, 255))

                T_tag_cam = np.zeros((4, 4))
                # Only try to find CharucoBoard if we found markers
                if ids is not None and len(ids) > 10:

                    # Get charuco corners and ids from detected aruco markers
                    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=gray,
                        board=CHARUCO_BOARD)

                    # Require more than 20 squares
                    if response is not None and response > 20:
                        # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video
                        board_pose_cam, rvec, tvec = aruco.estimatePoseCharucoBoard(
                            charucoCorners=charuco_corners,
                            charucoIds=charuco_ids,
                            board=CHARUCO_BOARD,
                            cameraMatrix=cameraMatrix,
                            distCoeffs=distCoeffs,
                            rvec=None, tvec=None)
                        if board_pose_cam:
                            # Draw the camera posture calculated from the gridboard
                            color_image = aruco.drawAxis(color_image, cameraMatrix, distCoeffs, rvec, tvec, 0.3)

                        # transform rvec to R matrix
                        R_mat = R.from_rotvec(rvec.squeeze()).as_matrix()
                        T_tag_cam[0:3, 0:3] = R_mat
                        T_tag_cam[0:3, 3] = tvec.squeeze()
                        T_tag_cam[3, 3] = 1

                     # Display our image
                    cv2.imshow('QueryImage', color_image)

                else:
                    print("Markers not found")

                print("T_tag_cam: \n", T_tag_cam)
                T_tcp_base = np.asarray(m3d.Transform(pose).get_matrix())

                if z == 0:
                    T_tag_cam1 = T_tag_cam
                    T_tcp1_base = T_tcp_base
                elif z == 1:
                    T_tag_cam2 = T_tag_cam
                    T_tcp2_base = T_tcp_base

            A = np.linalg.inv(T_tcp2_base) @ T_tcp1_base
            B = T_tag_cam2 @ np.linalg.inv(T_tag_cam1)

            A_list.append(A)
            B_list.append(B)

finally:
    # Stop streaming
    pipeline.stop()

    robot.close()

    T = calibrate(A_list, B_list)
    print("Hand-eye calibration T: \n", T)

    exit(0)
