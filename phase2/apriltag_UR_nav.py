import time
import math3d as m3d
import pickle

from scipy.spatial.transform import Rotation as R
from utils.apriltag_utils.TagDetector import TagDetector
from utils.apriltag_utils.pose_conversion import *
from utils.apriltag_utils.annotate_tag import *
from utils.webcamvideostream import WebcamVideoStream
import URBasic

# UR configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # Robot acceleration value
VELOCITY = 0.5  # Robot speed value

# robot_start_position = (0.37, 0, 0.18, 3.14, 0, 0)   # pose in base frame
robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))    # joint

# camera and tag detector configuration
with open('cam_param/fisheye_cam_param.pickle', 'rb') as f:
    cam_matrix, dist_coeffs = pickle.load(f)

tag_size = 0.048
tag_family = 'tagStandard41h12'
tagDetector = TagDetector(
    cam_matrix, dist_coeffs, tag_family, "fisheye")

# Camera parameters in TCP frame
cam_pos_TCP = (0.085, 0, -0.01) # camera position in TCP frame
cam_angle_TCP = (0, 0, 0)       # camera euler angles in TCP frame, must be fixed value

video_resolution = (1280, 720)
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

    tag_pose_base = np.hstack((tag_R_base, tag_pos_base[:, None]))  # [:,None] creates a new axis of length 1

    # use math3d to convert (R,t) to (x,y,z,rx,rx,rz)
    tag_6d_pose_base = m3d.Transform(tag_pose_base).get_pose_vector()
    # print("Tag pose in the base frame: ", tag_6d_pose_base)

    # Third: adjust the tag pose based on position and orientation limits

    # check position and angle limits, and update positions and angles
    tag_angle_base = np.array([tag_6d_pose_base[3], tag_6d_pose_base[4], tag_6d_pose_base[5]])

    # negate y and z angles if the camera's y-axis points up and z-axis points towards the camera
    r_rotvec = R.from_rotvec(tag_angle_base)
    r_euler = r_rotvec.as_euler('xyz', degrees=True)
    new_r_euler = [r_euler[0], -r_euler[1], -r_euler[2]]
    new_r_euler = R.from_euler('xyz', new_r_euler, degrees=True)
    updated_tag_angle_base = new_r_euler.as_rotvec()

    updated_tag_6d_pose = [tag_pos_base[0], tag_pos_base[1], tag_pos_base[2],
                           updated_tag_angle_base[0], updated_tag_angle_base[1], updated_tag_angle_base[2]]

    print("updated_tag_6d_pose: ", updated_tag_6d_pose)

    ##########################################
    T_tag_cam = np.zeros((4,4))  # capital T means transformation
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
        Compute the robot's final pose by manually adjusting the position and orientation
        - The unit of xyz offsets is meters.
        - The unit of rpy offsets is degrees
    """
    updated_pos = np.array([tag_pose[0] + x_offset, tag_pose[1] + y_offset, tag_pose[2] + z_offset])

    r_rotvec = R.from_rotvec([tag_pose[3], tag_pose[4], tag_pose[5]])
    r_euler = r_rotvec.as_euler('xyz', degrees=True)
    new_r_euler = R.from_euler('xyz', [r_euler[0] + roll_offset, r_euler[1] + pitch_offset, r_euler[2] + yaw_offset],
                               degrees=True)
    updated_angle = new_r_euler.as_rotvec()

    final_target_6d_pose = [updated_pos[0], updated_pos[1], updated_pos[2],
                            updated_angle[0], updated_angle[1], updated_angle[2]]
    print("Final target pose: ", final_target_6d_pose)
    return final_target_6d_pose


def show_frame(frame):
    cv.imshow('RobotCamera', frame)
    k = cv.waitKey(6) & 0xff


"""TAG LOCALIZATION LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

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
            # robot.movej(pose=tag_pose, a=ACCELERATION, v=VELOCITY)

            # use compute_target_pose if the target pose is different from the tag pose. Ex:
            # target_pose = compute_target_pose(tag_pose, x_offset=0.05, y_offset=0.25, z_offset=0.22)
            # target_pose = compute_target_pose(tag_pose)
            # robot.movej(pose=target_pose, a=ACCELERATION, v=VELOCITY)

            # robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

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
