# Deproject pixel to 3D point using rs2_deproject_pixel_to_point()
# Compute 3D target prediction using only one cam, using the interpolation model.
# This script wasn't used in our project.

import pyrealsense2 as rs
import pickle
import open3d as o3d
from scipy.linalg import null_space

from utils.trajectory_io import *
from utils.triangulation import *
from subject_info import SUBJECT_NAME, SCAN_POSE
from run_pipeline import POSE_MODEL

# depth cameras info:
CAM = 2
depth_unit = 0.0010000000474974513
if CAM == 1:
    depth_intr = np.array([
        [603.98217773, 0, 310.87359619],
        [0, 603.98217773, 231.11578369],
        [0, 0, 1]
    ])
    color_intr = np.array([
        [614.11938477, 0, 315.48043823],
        [0, 613.32940674, 236.26939392],
        [0, 0, 1]
    ])
elif CAM == 2:
    depth_intr = np.array([
        [595.13745117, 0, 318.53710938],
        [0, 595.13745117, 245.47492981],
        [0, 0, 1]
    ])
    color_intr = np.array([
        [613.22027588, 0, 318.16168213],
        [0, 612.14581299, 235.91072083],
        [0, 0, 1]
    ])
else:
    depth_intr, color_intr = None, None
    print("No applicable camera!!!")

folder_path = 'data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'

# Get T_cam_base
camera_poses = read_trajectory(folder_path + "odometry.log")
T_cam1_base = camera_poses[0].pose
T_cam2_base = camera_poses[1].pose
T_cam_base = T_cam1_base if CAM == 1 else T_cam2_base

color_image = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(CAM))
depth_image = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(CAM))
depth_data = np.asanyarray(depth_image)


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cam_intr):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = 640
    _intrinsics.height = 480
    _intrinsics.ppx = cam_intr[0, 2]
    _intrinsics.ppy = cam_intr[1, 2]
    _intrinsics.fx = cam_intr[0, 0]
    _intrinsics.fy = cam_intr[1, 1]
    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model = rs.distortion.none
    _intrinsics.coeffs = [0, 0, 0, 0, 0]

    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth[y, x] * depth_unit)
    # result[0]: right, result[1]: down, result[2]: forward
    # return result[2], -result[0], -result[1]
    return result


# TSDF volume
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=1 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image, depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)

intr = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=depth_intr[0, 0],
    fy=depth_intr[1, 1],
    cx=depth_intr[0, 2],
    cy=depth_intr[1, 2]
)

volume.integrate(rgbd, intr, np.linalg.inv(T_cam_base))
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
# o3d.visualization.draw_geometries([downpcd])
coordinates = np.asarray(downpcd.points)

# read pose keypoints
with open(folder_path + POSE_MODEL + '/keypoints/cam_{}_keypoints.pickle'.format(CAM), 'rb') as f:
    cam_keypoints = pickle.load(f)
    if POSE_MODEL == "OpenPose":
        l_shoulder = np.array(cam_keypoints['people'][0]['pose_keypoints_2d'][15:17])
        r_shoulder = np.array(cam_keypoints['people'][0]['pose_keypoints_2d'][6:8])
        l_hip = np.array(cam_keypoints['people'][0]['pose_keypoints_2d'][36:38])
        r_hip = np.array(cam_keypoints['people'][0]['pose_keypoints_2d'][27:29])
    else:
        l_shoulder = cam_keypoints[0]['keypoints'][5][:2]
        r_shoulder = cam_keypoints[0]['keypoints'][6][:2]
        l_hip = cam_keypoints[0]['keypoints'][11][:2]
        r_hip = cam_keypoints[0]['keypoints'][12][:2]


def target12_3D(l_shoulder, r_shoulder, ratio_1=0.3, ratio_2=0.1):
    X1 = convert_depth_to_phys_coord_using_realsense(int(r_shoulder[0]), int(r_shoulder[1]), depth_data,
                                                     depth_intr)  # in the cam frame
    X1 = from_homog(T_cam_base @ to_homog(np.array(X1).reshape(-1, 1)))  # convert to the base frame
    X1 = post_process(X1)
    X2 = convert_depth_to_phys_coord_using_realsense(int(l_shoulder[0]), int(l_shoulder[1]), depth_data, depth_intr)
    X2 = from_homog(T_cam_base @ to_homog(np.array(X2).reshape(-1, 1)))
    X2 = post_process(X2)

    X3 = X1 + ratio_1 * (X2 - X1)

    t1 = ((X2 - X1) / np.linalg.norm(X2 - X1)).squeeze()
    n = np.array([0, 0, 1])  # z-axis of the base frame / normal vector of the base frame's x-y plane
    A = np.vstack([t1, n])
    t2 = null_space(A)  # t2 is perpendicular to both t1 and n
    t2 = t2 / np.linalg.norm(t2)  # normalize t2

    if t2[0] < 0:
        t2 = -t2

    target = X3 + ratio_2 * np.linalg.norm(X2 - X1) * t2
    # print("target before pose process:\n", target)
    target = post_process(target)
    # print("target after pose process:\n", target)

    return target


def target4_3D(r_shoulder, r_hip, ratio_1=0.35, ratio_2=0.1):
    X1 = convert_depth_to_phys_coord_using_realsense(int(r_shoulder[0]), int(r_shoulder[1]), depth_data,
                                                     depth_intr)  # in the cam frame
    X1 = from_homog(T_cam_base @ to_homog(np.array(X1).reshape(-1, 1)))  # convert to the base frame
    X1 = post_process(X1)
    X2 = convert_depth_to_phys_coord_using_realsense(int(r_hip[0]), int(r_hip[1]), depth_data, depth_intr)
    X2 = from_homog(T_cam_base @ to_homog(np.array(X2).reshape(-1, 1)))
    X2 = post_process(X2)

    X3 = X1 + ratio_1 * (X2 - X1)

    t1 = ((X2 - X1) / np.linalg.norm(X2 - X1)).squeeze()
    n = np.array([0, 0, 1])
    A = np.vstack([t1, n])
    t2 = null_space(A)
    t2 = t2 / np.linalg.norm(t2)

    if t2[1] < 0:
        t2 = -t2

    target = X3 + ratio_2 * np.linalg.norm(X1 - X3) * t2
    target = post_process(target)
    return target


def post_process(original_3d):
    # keeping the original x,y value
    idx = np.argmin(np.square(coordinates[:, 0:2] - original_3d.squeeze()[0:2]).sum(axis=1))
    new_z = coordinates[idx, 2]
    new_3d = np.vstack((original_3d[0:2], [new_z]))
    return new_3d


one_cam_target4 = target4_3D(r_shoulder, r_hip)
print("one_cam_target4: \n", one_cam_target4)
