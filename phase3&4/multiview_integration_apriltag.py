# Manipulate UR3e to move in a designated circular path to scan and
# take pictures of the mannequin to reconstruct its surface with AprilTag
# change the transformation matrix of the first camera to be identity

import pyrealsense2 as rs
import URBasic
import time
from utils.apriltag_utils.TagDetector import TagDetector
from utils.pose_conversion import *
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
# all_poses = [ (0.43414, 0.11073, 0.34996, 2.965, 0.326, 0.215),
#               (0.43414, 0.01073, 0.34996, 2.708, 1.377, 0.286)]

# vertical pose
# all_poses = [(0.394, 0.261, 0.049, 2.31, 1.016, -0.432),
#              (0.394, 0.261, 0.049, 2.31, 1.016, -0.432)]

all_poses = [ (0.38011, 0.00178, 0.4396, 3.153, 0.029, 0.38),
              (0.33711, 0.22542, 0.41898, 2.446, 0.975, 0.411) ]

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
tag_size = 0.072
tag_family = 'tagStandard41h12'
cam_type = "standard"
tagDetector = TagDetector(intr, None, tag_family, cam_type)

print("AprilTag detector prepared!")

time.sleep(1)


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
T_tag_cam_list = []

try:
    for i, pose in enumerate(all_poses):
        robot.movej(pose=pose, a=ACCELERATION, v=VELOCITY)
        time.sleep(2)
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

        cv.imwrite('RGBD/color/{}.jpg'.format(i), color_image)
        cv.imwrite('RGBD/depth/{}.png'.format(i), depth_image)

        # Locate camera with AprilTag
        _, detection_results = tagDetector.detect_tags(color_image, tag_size)
        show_frame(color_image)

        print("Number of tags detected: ", len(detection_results))
        if len(detection_results) != 0:
            T_tag_cam_avg = 0
            for result in detection_results:
                annotate_tag(result, color_image)
                show_frame(color_image)

                T_tag_cam = np.zeros((4,4))
                T_tag_cam[0:3,0:3] = result.pose_R
                T_tag_cam[0:3,3] = result.pose_t.squeeze()
                T_tag_cam[3,3] = 1

                T_tag_cam_avg += T_tag_cam

            T_tag_cam_avg = T_tag_cam_avg / len(detection_results)
            T_tag_cam_list.append(T_tag_cam_avg)

        else:
            print("No Tag detected")


    # compute the average translation and rotation of three AprilTag results
    for i,T_tag_cam in enumerate(T_tag_cam_list):
        # transformation from the i-th camera to the first camera
        T_cam2_cam1 = T_tag_cam_list[0] @ np.linalg.inv(T_tag_cam)
        R_cam2_cam1 = T_cam2_cam1[0:3,0:3]
        t_cam2_cam1 = T_cam2_cam1[0:3,3]
        write_to_file(i, R_cam2_cam1, t_cam2_cam1)

finally:
    # Stop streaming
    pipeline.stop()
    robot.movej(q=robot_start_position, a=ACCELERATION, v=VELOCITY)

    robot.close()
    exit(0)
