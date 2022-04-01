import time
import math3d as m3d
import pickle

from scipy.spatial.transform import Rotation as R
from utils.apriltag_utils.TagDetector import TagDetector
from utils.pose_conversion import *
from utils.apriltag_utils.annotate_tag import *
from utils.webcamvideostream import WebcamVideoStream
import URBasic

# UR robot configuration
ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # Robot acceleration value
VELOCITY = 0.5  # Robot speed value

robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))    # joint angles

# camera and tag detector configuration
with open('../cam_param/fisheye_cam_param.pickle', 'rb') as f:
    cam_matrix, dist_coeffs = pickle.load(f)

tag_size = 0.048
tag_family = 'tagStandard41h12'
tagDetector = TagDetector(
    cam_matrix, dist_coeffs, tag_family, "fisheye")

# Camera parameters in TCP frame
cam_pos_TCP = (0, 0.025, 0.025) # camera position in TCP frame
cam_angle_TCP = (0, 0, 0)       # camera euler angles in TCP frame, must be fixed value
R_cam_tcp = eulerAnglesToRotationMatrix(np.array(cam_angle_TCP) /180 * np.pi)   # convert euler angle to rotation matrix

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

    # transformation from tag frame to camera frame
    T_tag_cam = np.zeros((4,4))  # capital T means transformation
    T_tag_cam[0:3,0:3] = R_tag_cam
    T_tag_cam[0:3,3] = t_tag_cam
    T_tag_cam[3,3] = 1

    # transformation from camera frame to robot tcp frame
    t_cam_tcp = np.array(cam_pos_TCP)
    T_cam_tcp = np.zeros((4, 4))
    T_cam_tcp[0:3, 0:3] = R_cam_tcp
    T_cam_tcp[0:3, 3] = t_cam_tcp
    T_cam_tcp[3, 3] = 1

    # transformation from robot tcp frame to base frame
    R_tcp_base = TCP_pose_base[0:3, 0:3]
    t_tcp_base = TCP_pose_base[0:3, 3].squeeze()
    T_tcp_base = np.zeros((4, 4))
    T_tcp_base[0:3, 0:3] = R_tcp_base
    T_tcp_base[0:3, 3] = t_tcp_base
    T_tcp_base[3, 3] = 1

    # transformation from tag frame to base frame
    T_tag_base = T_tcp_base @ T_cam_tcp @ T_tag_cam
    target_6d_pose = m3d.Transform(T_tag_base).get_pose_vector()

    print("target_6d_pose: ", target_6d_pose)

    return target_6d_pose


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
        undistorted_img, detection_results = tagDetector.detect_tags( frame, tag_size)
        # print('undistorted_img:\n',undistorted_img)
        show_frame(undistorted_img)
        if len(detection_results) != 0:
            result = detection_results[0]  # for now, only consider one tag
            annotate_tag(result, undistorted_img)
            show_frame(undistorted_img)
            tag_pose = compute_tag_pose(result)
            robot.movej(pose=tag_pose, a=ACCELERATION, v=VELOCITY)

            robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

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
