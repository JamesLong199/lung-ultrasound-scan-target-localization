import math
import numpy as np
import cv2 as cv
import time
import math3d as m3d
import pickle

from utils.URBasic.kinematic import Forwardkin_manip, Invkine_manip
from scipy.spatial.transform import Rotation as R
from detect_apriltag import TagDetector
from tag import Tag
from utils import annotate_tag, eulerAnglesToRotationMatrix, rotationMatrixToEulerAngles, rad_to_degree, degree_to_rad
from webcamvideostream import WebcamVideoStream

# UR configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # Robot acceleration value
VELOCITY = 0.5  # Robot speed value

# robot_startposition = (0.37, 0, 0.18, 3.14, 0, 0)   # pose in base frame
robot_startposition = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                       np.radians(-125.05), np.radians(89.56), np.radians(291.04))    # joint

# 3D rectangle space of robot range of motion
# pos_limit = np.array([[0.22, -0.10, -0.20], [0.35, 0.30, 0.17]]).T
pos_limit = np.array([[0.25, -0.10, -0.25], [0.75, 0.30, 0.20]]).T
rot_limit = np.array([[90, -90, -90], [270, 90, 90]]).T     # in terms of RPY / euler angles in degrees,
# not rotational vector

video_resolution = (1280, 720)

with open('cam_param/fisheye_cam_param.pickle', 'rb') as f:
    cam_matrix, dist_coeffs = pickle.load(f)

tag_size = 0.048
tag_family = 'tagStandard41h12'
tags = Tag(tag_size=tag_size, family=tag_family)
tagDetector = TagDetector(
    cam_matrix, dist_coeffs, tag_family, "fisheye")

# Camera parameters in TCP frame
cam_pos_TCP = (0.085, 0, -0.01) # camera position in TCP frame
cam_angle_TCP = (0, 0, 0)       # camera euler angles in TCP frame, must be fixed value

vs = WebcamVideoStream(resolution=video_resolution, src=1).start()

time.sleep(0.2)


def compute_tag_pose(result):
    """
        Compute tag pose in the base frame, given camera's TCP pose
        Essentially two conversion processes of the tag pose:
            tag frame --> TCP frame --> base frame
        Then adjust the tag pose by checking limits
        - Naming convention: (object)_R/T/pos/angle_(frame)

        Input:
        - result: one AprilTag detection result
        - tilt: the tilt angle of wrist 1 of UR
        Output:
        - return the tag pose in the base frame
    """
    print("############################################")
    # First stage: tag frame --> TCP frame:

    # transformation from tag frame to camera frame
    R_tag_cam = result.pose_R
    t_tag_cam = result.pose_t.reshape(3, )

    # transformation from camera frame to tag frame
    t_cam_tag = R_tag_cam.T @ (-t_tag_cam)
    cam_angle_tag = rotationMatrixToEulerAngles(R_tag_cam) * 180 / math.pi  # camera angle in tag frame

    # compute the tag position and orientation in the TCP frame
    # angles between two frames = angle of an object in one frame - angle of the same object in another frame
    tag_angle_TCP = cam_angle_tag - cam_angle_TCP  # compute the angle difference
    tag_R_TCP = eulerAnglesToRotationMatrix(tag_angle_TCP / 180 * math.pi)

    # cam_R_TCP = eulerAnglesToRotationMatrix(cam_angle_TCP)
    # tag_R_TCP = cam_R_TCP @ R_tag_cam.T

    # tag position in TCP frame
    tag_pos_TCP = cam_pos_TCP - tag_R_TCP @ t_cam_tag

    # Second: TCP frame --> Base frame:

    TCP_6d_pose_base = robot.get_actual_tcp_pose()  # returns the 6d TCP/tool pose in the base frame

    # use math3d to convert (x,y,z,rx,rx,rz) to (R,t)
    TCP_pose_base = np.asarray(m3d.Transform(TCP_6d_pose_base).get_matrix())  # 4x4 matrix

    TCP_R_base = TCP_pose_base[0:3, 0:3]
    TCP_T_base = TCP_pose_base[0:3, 3].squeeze()

    # compute the tag position in the base frame using the tag position in the TCP frame
    tag_pos_base = TCP_R_base @ tag_pos_TCP + TCP_T_base  # (3,1)

    # compute the tag orientation in the base frame by multiplying both rotation matrices
    tag_R_base = tag_R_TCP @ TCP_R_base  # 3 x 3 rotational matrix

    # print("tag base angle: ", tags.rotationMatrixToEulerAngles(tag_R_base) * 180 / math.pi)
    tag_pose_base = np.hstack((tag_R_base, tag_pos_base[:, None]))  # [:,None] creates a new axis of length 1

    # use math3d to convert (R,t) to (x,y,z,rx,rx,rz)
    tag_6d_pose_base = m3d.Transform(tag_pose_base).get_pose_vector()
    # print("Tag pose in the base frame: ", tag_6d_pose_base)

    # Third: adjust the tag pose based on position and orientation limits

    # check position and angle limits, and update positions and angles
    # updated_tag_pos_base = check_pos_limit(tag_pos_base)
    tag_angle_base = np.array([tag_6d_pose_base[3], tag_6d_pose_base[4], tag_6d_pose_base[5]])
    # updated_tag_angle_base = check_angle_limit(tag_angle_base)

    r_rotvec = R.from_rotvec(tag_angle_base)
    r_euler = r_rotvec.as_euler('xyz', degrees=True)
    new_r_euler = [r_euler[0], -r_euler[1], -r_euler[2]]
    new_r_euler = R.from_euler('xyz', new_r_euler, degrees=True)
    updated_tag_angle_base = new_r_euler.as_rotvec()

    updated_tag_6d_pose = [tag_pos_base[0], tag_pos_base[1], tag_pos_base[2],
                           updated_tag_angle_base[0], updated_tag_angle_base[1], updated_tag_angle_base[2]]

    print("updated_tag_6d_pose: ", updated_tag_6d_pose)

    ##########################################
    T_tag_cam = np.zeros((4,4))
    T_tag_cam[0:3,0:3] = R_tag_cam
    T_tag_cam[0:3,3] = t_tag_cam
    T_tag_cam[3,3] = 1

    R_cam_tcp = np.eye(3)
    t_cam_tcp = np.array([0.085, 0, -0.01])
    T_cam_tcp = np.zeros((4, 4))
    T_cam_tcp[0:3, 0:3] = R_cam_tcp
    T_cam_tcp[0:3, 3] = t_cam_tcp
    T_cam_tcp[3, 3] = 1

    R_tcp_base = TCP_pose_base[0:3, 0:3]
    t_tcp_base = TCP_pose_base[0:3, 3].squeeze()
    T_tcp_base = np.zeros((4, 4))
    T_tcp_base[0:3, 0:3] = R_tcp_base
    T_tcp_base[0:3, 3] = t_tcp_base
    T_tcp_base[3, 3] = 1

    T_tag_base = T_tcp_base @ T_cam_tcp @ T_tag_cam
    recomputed_6d_pose = m3d.Transform(T_tag_base).get_pose_vector()

    print("recomputed_6d_pose: ", recomputed_6d_pose)

    return updated_tag_6d_pose


def compute_target_pose(tag_pose, x_offset=0, y_offset=0, z_offset=0, roll_offset=0, pitch_offset=0, yaw_offset=0):
    """
        Compute the robot's final pose by adjusting the position and orientation
        - The unit of xyz offsets is meters.
        - The unit of rpy offsets is degrees
    """
    updated_pos = np.array([tag_pose[0] + x_offset, tag_pose[1] + y_offset, tag_pose[2] + z_offset])
    # updated_pos = check_pos_limit(updated_pos)

    r_rotvec = R.from_rotvec([tag_pose[3], tag_pose[4], tag_pose[5]])
    r_euler = r_rotvec.as_euler('xyz', degrees=True)
    new_r_euler = R.from_euler('xyz', [r_euler[0] + roll_offset, r_euler[1] + pitch_offset, r_euler[2] + yaw_offset],
                               degrees=True)
    updated_angle = new_r_euler.as_rotvec()
    # updated_angle = check_angle_limit(updated_angle)

    final_target_6d_pose = [updated_pos[0], updated_pos[1], updated_pos[2],
                            updated_angle[0], updated_angle[1], updated_angle[2]]
    print("Final target pose: ", final_target_6d_pose)
    return final_target_6d_pose


def compute_target_pose_with_inverse_kin(tag_pose, x_offset=0, y_offset=0, z_offset=0, wrist1_tilt=0):
    """
        Compute the robot's target pose with respect to the tag pose in the base frame
        - offsets are in meters
        - wrist1_tilt is in degree
    """

    updated_pos = np.array([tag_pose[0] + x_offset, tag_pose[1] + y_offset, tag_pose[2] + z_offset])
    updated_pos = check_pos_limit(updated_pos)

    target_6d_pose = [updated_pos[0], updated_pos[1], updated_pos[2], tag_pose[3], tag_pose[4], tag_pose[5]]

    curr_joint_rad = robot.get_actual_joint_positions()
    curr_joint_rad[3] = math.radians(-60)
    if wrist1_tilt != 0:
        curr_joint_degree = rad_to_degree(curr_joint_rad)
        curr_joint_degree[3] -= wrist1_tilt
        curr_joint_rad = degree_to_rad(curr_joint_degree)

    # Use inverse and forward kinematics to update the orientation by estimating the joint positions
    target_joint = Invkine_manip(target_6d_pose, init_joint_pos=curr_joint_rad, rob='ur3e')
    target_6d_pose_with_updated_angle = Forwardkin_manip(target_joint, rob='ur3e')  # only take its orientation

    # combine the updated orientation with the previously updated positions
    final_target_6d_pose = [target_6d_pose[0], target_6d_pose[1], target_6d_pose[2],
                            target_6d_pose_with_updated_angle[3], target_6d_pose_with_updated_angle[4],
                            target_6d_pose_with_updated_angle[5]]
    print("Final target pose: ", final_target_6d_pose)
    return final_target_6d_pose


def check_pos_limit(target_pos):
    """
        Function that updates target pos according to robot's range of motion

        Inputs:
            target_pos: tag position in base frame

        Return Value:
            target_pos: updated target position
    """
    assert target_pos.size == 3

    for i in range(3):
        if target_pos[i] > pos_limit[i, 1]:
            target_pos[i] = pos_limit[i, 1]

        elif target_pos[i] < pos_limit[i, 0]:
            target_pos[i] = pos_limit[i, 0]

    return target_pos


def check_angle_limit(target_angle):
    """
        Function that updates target angle according to robot's range of orientation

        Inputs:
            target_angle: tag's rotation vector in base frame

        Return Value:
            target_pos: updated target position
    """
    assert target_angle.size == 3

    r_rotvec = R.from_rotvec(target_angle)
    r_euler = r_rotvec.as_euler('xyz', degrees=True)

    for i in range(3):
        if i != 1 and r_euler[i] < 0:
            r_euler[i] += 360

        if r_euler[i] > rot_limit[i, 1]:
            r_euler[i] = rot_limit[i, 1]

        elif r_euler[i] < rot_limit[i, 0]:
            r_euler[i] = rot_limit[i, 0]

    new_r_euler = R.from_euler('xyz', r_euler, degrees=True)
    target_angle = new_r_euler.as_rotvec()

    return target_angle


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff


"""TAG LOCALIZATION LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")
robotModel = utils.URBasic.robotModel.RobotModel()
robot = utils.URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
robot.movej(q=robot_startposition, a=ACCELERATION, v=VELOCITY)

robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1)  # just a short wait to make sure everything is initialised

try:
    print("starting loop")
    while True:
        frame = vs.read()
        undistorted_img, detection_results = tagDetector.detect_tags("frame", frame, tag_size)
        show_frame(undistorted_img)
        if len(detection_results) != 0:
            result = detection_results[0]  # for now, only consider one tag
            annotate_tag(result, undistorted_img)
            show_frame(undistorted_img)
            tag_pose = compute_tag_pose(result)

            # target_pose = compute_target_pose(tag_pose, x_offset=0.05, y_offset=0.25, z_offset=0.22)
            target_pose = compute_target_pose(tag_pose)
            robot.movej(pose=target_pose, a=ACCELERATION, v=VELOCITY)
            robot.movej(q=robot_startposition, a=ACCELERATION, v=VELOCITY)

            # target_pose = compute_target_pose(tag_pose, y_offset=-0.1, z_offset=0.01)
            # robot.movej(pose=target_pose, a=ACCELERATION, v=VELOCITY)
            # robot.movej(pose=robot_startposition, a=ACCELERATION, v=VELOCITY)


            # target_pose = compute_target_pose(tag_pose, x_offset=-0.07, y_offset=-0.1, z_offset=-0.03, pitch_offset=40)
            # robot.movej(pose=target_pose, a=ACCELERATION, v=VELOCITY)
            # robot.movej(pose=robot_startposition, a=ACCELERATION, v=VELOCITY)

        time.sleep(0.5)

    print("exiting loop")
except KeyboardInterrupt:
    print("closing robot connection")
    # Remember to always close the robot connection, otherwise it is not possible to reconnect
    robot.close()

except:
    robot.close()

finally:
    cv.destroyAllWindows()

    exit(1)
    robot.close()
