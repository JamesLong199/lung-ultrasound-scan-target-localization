# Take color & depth images from the two Intel Realsense cameras

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

from subject_info import SUBJECT_NAME, SCAN_POSE


# Two Depth Cameras Configuration

# Configure depth and color streams

# ... from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('839112060979')  # change to your device
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ... from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('007522060984')  # change to your device
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

# get camera 1 intrinsic parameters
profile_1 = pipeline_1.get_active_profile()
color_profile_1 = rs.video_stream_profile(profile_1.get_stream(rs.stream.color))
color_intrinsics_1 = color_profile_1.get_intrinsics()
cam1_color_intr = np.array([[color_intrinsics_1.fx, 0, color_intrinsics_1.ppx],
                 [0, color_intrinsics_1.fy, color_intrinsics_1.ppy],
                 [0, 0, 1]])
depth_profile_1 = rs.video_stream_profile(profile_1.get_stream(rs.stream.depth))
depth_intrinsics_1 = depth_profile_1.get_intrinsics()
cam1_depth_intr = np.array([[depth_intrinsics_1.fx, 0, depth_intrinsics_1.ppx],
                 [0, depth_intrinsics_1.fy, depth_intrinsics_1.ppy],
                 [0, 0, 1]])

folder_path = 'src/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'

with open(folder_path + 'intrinsics/cam_1_intrinsics.pickle', 'wb') as f:
    pickle.dump(cam1_color_intr, f)
with open(folder_path + 'intrinsics/cam_1_depth_intr.pickle', 'wb') as f:
    pickle.dump(cam1_depth_intr, f)

# get camera 2 intrinsic parameters
profile_2 = pipeline_2.get_active_profile()
color_profile_2 = rs.video_stream_profile(profile_2.get_stream(rs.stream.color))
color_intrinsics_2 = color_profile_2.get_intrinsics()
cam2_color_intr = np.array([[color_intrinsics_2.fx, 0, color_intrinsics_2.ppx],
                 [0, color_intrinsics_2.fy, color_intrinsics_2.ppy],
                 [0, 0, 1]])
depth_profile_2 = rs.video_stream_profile(profile_2.get_stream(rs.stream.depth))
depth_intrinsics_2 = depth_profile_2.get_intrinsics()
cam2_depth_intr = np.array([[depth_intrinsics_2.fx, 0, depth_intrinsics_2.ppx],
                 [0, depth_intrinsics_2.fy, depth_intrinsics_2.ppy],
                 [0, 0, 1]])

with open(folder_path + 'intrinsics/cam_2_intrinsics.pickle', 'wb') as f:
    pickle.dump(cam2_color_intr, f)
with open(folder_path + 'intrinsics/cam_2_depth_intr.pickle', 'wb') as f:
    pickle.dump(cam2_depth_intr, f)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# adjust lightness
exposure_1 = 250
exposure_2 = 250

try:
    cam1_success, cam2_success = False, False
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        sensor_1 = profile_1.get_device().query_sensors()[1]
        sensor_1.set_option(rs.option.exposure, exposure_1)
        frames_1 = pipeline_1.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames_1 = align.process(frames_1)
        depth_frame_1 = aligned_frames_1.get_depth_frame()
        color_frame_1 = aligned_frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        else:
            cam1_success = True
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)
        cv2.imwrite(folder_path + 'color_images/cam_1.jpg', color_image_1)
        cv2.imwrite(folder_path + 'depth_images/cam_1.png', depth_image_1)
        print("Captured images from camera 1")

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        sensor_2 = profile_2.get_device().query_sensors()[1]
        sensor_2.set_option(rs.option.exposure, exposure_2)
        frames_2 = pipeline_2.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames_2 = align.process(frames_2)
        depth_frame_2 = aligned_frames_2.get_depth_frame()
        color_frame_2 = aligned_frames_2.get_color_frame()
        if not depth_frame_2 or not color_frame_2:
            continue
        else:
            cam2_success = True
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.5), cv2.COLORMAP_JET)
        cv2.imwrite(folder_path + 'color_images/cam_2.jpg', color_image_2)
        cv2.imwrite(folder_path + 'depth_images/cam_2.png', depth_image_2)
        print("Captured images from camera 2")

        # Stack all images horizontally
        images = np.hstack((color_image_1, depth_colormap_1, color_image_2, depth_colormap_2))

        # Show images from both cameras
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        if cam1_success and cam2_success:
            break

finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()