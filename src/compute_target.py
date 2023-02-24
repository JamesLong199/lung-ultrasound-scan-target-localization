# Given HPE results and ratios, compute the final target point

import cv2
import matplotlib.pyplot as plt
from utils.trajectory_io import *
import pickle
from utils.triangulation import *
import open3d as o3d
from scipy.linalg import null_space

from subject_info import SUBJECT_NAME, SCAN_POSE
import argparse
import time

parser = argparse.ArgumentParser(description='Compute target')
parser.add_argument('--pose_model', type=str, default='ViTPose_large', help='pose model')
parser.add_argument('--subject_name', type=str, default='John Doe', help='subject name')
parser.add_argument('--scan_pose', type=str, default='none', help='scan pose')
parser.add_argument('--target1_r1', type=float, default=0.3, help='target1_r1')
parser.add_argument('--target1_r2', type=float, default=0.1, help='target1_r2')
parser.add_argument('--target2_r1', type=float, default=0.3, help='target2_r1')
parser.add_argument('--target2_r2', type=float, default=0.55, help='target2_r2')
parser.add_argument('--target4_r1', type=float, default=0.35, help='target4_r1')
parser.add_argument('--target4_r2', type=float, default=0.1, help='target4_r2')

args = parser.parse_args()
POSE_MODEL = args.pose_model
if args.subject_name != 'John Doe':
    SUBJECT_NAME = args.subject_name
if args.scan_pose != 'none':
    SCAN_POSE = args.scan_pose


def target12_3D(l_shoulder_cam1, l_shoulder_cam2, r_shoulder_cam1, r_shoulder_cam2,
                cam1_intr, cam2_intr, T_base_cam1, T_base_cam2, ratio_1=0.3, ratio_2=0.1):
    '''
    A 3D method to determine the 1st and 2nd target 3D coordinate. Didn't use nipples.
    :X1, X2 --3D coordinates of the right, left shoulder
    :X3 -- 3D coordinate of the point determined by ratio_1
    :t1 -- the direction vector of the line connecting X1 and X2
    :t2 -- the direction vector of the line connecting X3 and target
    :param ratio_1: ratio on the line connecting X1 and X2
    :param ratio_2: ratio between X3 to target and X1 to X2
    '''
    X1 = reconstruct(r_shoulder_cam1, r_shoulder_cam2, cam1_intr, cam2_intr, T_base_cam1, T_base_cam2)
    X1 = triangulation_post_process(X1)
    X2 = reconstruct(l_shoulder_cam1, l_shoulder_cam2, cam1_intr, cam2_intr, T_base_cam1, T_base_cam2)
    X2 = triangulation_post_process(X2)

    X3 = X1 + ratio_1 * (X2 - X1)

    t1 = ((X2 - X1) / np.linalg.norm(X2 - X1)).squeeze()
    n = np.array([0, 0, 1])  # z-axis of the base frame / normal vector of the base frame's x-y plane
    A = np.vstack([t1, n])
    t2 = null_space(A)  # t2 is perpendicular to both t1 and n
    t2 = t2 / np.linalg.norm(t2)  # normalize t2

    if t2[0] < 0:
        t2 = -t2

    target = X3 + ratio_2 * np.linalg.norm(X2 - X1) * t2
    target = triangulation_post_process(target, verbose=False)
    position_data['front'] = [X1, X2, t2]

    return target


def triangulation_post_process(original_3d, verbose=False):
    '''
    Use the original x&y values to find a closest point in the point cloud.
    Use the new point's z value as the new z, combined with the original x&y.
    :param original_3d: The original 3D coordinate from triangulation
    '''
    if verbose:
        print("original_3d: ", original_3d)
    idx = np.argmin(np.square(coordinates[:, 0:2] - original_3d.squeeze()[0:2]).sum(axis=1))
    if verbose:
        new_3d = coordinates[idx]
        print("new_3d: ", new_3d)
    new_z = coordinates[idx, 2]
    new_3d = np.vstack((original_3d[0:2], [new_z]))
    return new_3d


def target4_3D(r_shoulder_cam1, r_shoulder_cam2, r_hip_cam1, r_hip_cam2,
               cam1_intr, cam2_intr, T_base_cam1, T_base_cam2, ratio_1=0.35, ratio_2=0.1):
    '''
    Similar to 1st&2nd target but ratio 2 operated on X1 to X3, instead of X1 to X2
    :X1, X2 --3D coordinates of the right shoulder and the right hip
    :X3 -- 3D coordinate of the point determined by ratio_1
    :t1 -- the direction vector of the line connecting X1 and X2
    :t2 -- the direction vector of the line connecting X3 and target
    :param ratio_1: ratio on the line connecting X1 and X2
    :param ratio_2: ratio between X3 to target and X1 to X3
    '''
    X1 = reconstruct(r_shoulder_cam1, r_shoulder_cam2, cam1_intr, cam2_intr, T_base_cam1, T_base_cam2)
    X1 = triangulation_post_process(X1)
    X2 = reconstruct(r_hip_cam1, r_hip_cam2, cam1_intr, cam2_intr, T_base_cam1, T_base_cam2)
    X2 = triangulation_post_process(X2)

    X3 = X1 + ratio_1 * (X2 - X1)

    t1 = ((X2 - X1) / np.linalg.norm(X2 - X1)).squeeze()
    n = np.array([0, 0, 1])
    A = np.vstack([t1, n])
    t2 = null_space(A)
    t2 = t2 / np.linalg.norm(t2)

    if t2[1] < 0:
        t2 = -t2

    target = X3 + ratio_2 * np.linalg.norm(X1 - X3) * t2
    target = triangulation_post_process(target)
    position_data['side'] = [X1, X2, t2]

    return target


def draw_front_target_point(target_point1, target_point2, l_shoulder1, r_shoulder1, l_shoulder2, r_shoulder2, target):
    image1 = plt.imread(folder_path + 'color_images/cam_1.jpg')
    cv2.circle(image1, (int(target_point1[0]), int(target_point1[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image1, (int(l_shoulder1[0]), int(l_shoulder1[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image1, (int(r_shoulder1[0]), int(r_shoulder1[1])), 2, (36, 255, 12), 2, -1)
    cv2.line(image1, (int(l_shoulder1[0]), int(l_shoulder1[1])), (int(r_shoulder1[0]), int(r_shoulder1[1])),
             color=(255, 0, 0), thickness=2)

    plt.subplot(121)
    plt.imshow(image1)

    image2 = plt.imread(folder_path + 'color_images/cam_2.jpg')
    cv2.circle(image2, (int(target_point2[0]), int(target_point2[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image2, (int(l_shoulder2[0]), int(l_shoulder2[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image2, (int(r_shoulder2[0]), int(r_shoulder2[1])), 2, (36, 255, 12), 2, -1)
    cv2.line(image2, (int(l_shoulder2[0]), int(l_shoulder2[1])), (int(r_shoulder2[0]), int(r_shoulder2[1])),
             color=(255, 0, 0), thickness=2)

    plt.subplot(122)
    plt.imshow(image2)

    plt.show()

    if target == 1:
        plt.imsave(folder_path + 'target1_highlight/cam_1.jpg', image1)
        plt.imsave(folder_path + 'target1_highlight/cam_2.jpg', image2)
    else:
        plt.imsave(folder_path + 'target2_highlight/cam_1.jpg', image1)
        plt.imsave(folder_path + 'target2_highlight/cam_2.jpg', image2)


def draw_side_target_point(target_point1, target_point2, r_hip1, r_shoulder1, r_hip2, r_shoulder2):
    image1 = plt.imread(folder_path + 'color_images/cam_1.jpg')
    cv2.circle(image1, (int(target_point1[0]), int(target_point1[1])), 3, (36, 255, 12), 3, -1)
    cv2.circle(image1, (int(r_hip1[0]), int(r_hip1[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image1, (int(r_shoulder1[0]), int(r_shoulder1[1])), 2, (36, 255, 12), 2, -1)
    cv2.line(image1, (int(r_hip1[0]), int(r_hip1[1])), (int(r_shoulder1[0]), int(r_shoulder1[1])),
             color=(255, 0, 0), thickness=2)

    plt.subplot(121)
    plt.imshow(image1)

    image2 = plt.imread(folder_path + 'color_images/cam_2.jpg')
    cv2.circle(image2, (int(target_point2[0]), int(target_point2[1])), 3, (36, 255, 12), 3, -1)
    cv2.circle(image2, (int(r_hip2[0]), int(r_hip2[1])), 2, (36, 255, 12), 2, -1)
    cv2.circle(image2, (int(r_shoulder2[0]), int(r_shoulder2[1])), 2, (36, 255, 12), 2, -1)
    cv2.line(image2, (int(r_hip2[0]), int(r_hip2[1])), (int(r_shoulder2[0]), int(r_shoulder2[1])),
             color=(255, 0, 0), thickness=2)

    plt.subplot(122)
    plt.imshow(image2)

    plt.show()
    plt.imsave(folder_path + 'target4_highlight/cam_1.jpg', image1)
    plt.imsave(folder_path + 'target4_highlight/cam_2.jpg', image2)

folder_path = 'src/data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'  # when run from src directory
# folder_path = '../data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/'  # when run from evaluation directory

# read RGBD intrinsics
with open(folder_path + 'intrinsics/cam_1_intrinsics.pickle', 'rb') as f:
    cam1_intr = pickle.load(f)

with open(folder_path + 'intrinsics/cam_2_intrinsics.pickle', 'rb') as f:
    cam2_intr = pickle.load(f)

with open(folder_path + 'intrinsics/cam_1_depth_intr.pickle', 'rb') as f:
    depth_cam1_intr = pickle.load(f)

with open(folder_path + 'intrinsics/cam_2_depth_intr.pickle', 'rb') as f:
    depth_cam2_intr = pickle.load(f)

# read extrinsics
camera_poses = read_trajectory(folder_path + "odometry.log")
T_cam1_base = camera_poses[0].pose
T_base_cam1 = np.linalg.inv(T_cam1_base)
# print("T_base_cam1/extr: \n", T_base_cam1)
T_cam2_base = camera_poses[1].pose
T_base_cam2 = np.linalg.inv(T_cam2_base)
# print("T_base_cam2/extr: \n", T_base_cam2)

# read pose keypoints
with open(folder_path + POSE_MODEL + '/keypoints/cam_1_keypoints.pickle', 'rb') as f:
    cam1_keypoints = pickle.load(f)
    if POSE_MODEL == "OpenPose":
        l_shoulder1 = np.array(cam1_keypoints['people'][0]['pose_keypoints_2d'][15:17])
        r_shoulder1 = np.array(cam1_keypoints['people'][0]['pose_keypoints_2d'][6:8])
        l_hip1 = np.array(cam1_keypoints['people'][0]['pose_keypoints_2d'][36:38])
        r_hip1 = np.array(cam1_keypoints['people'][0]['pose_keypoints_2d'][27:29])
    else:
        l_shoulder1 = cam1_keypoints[0]['keypoints'][5][:2]
        r_shoulder1 = cam1_keypoints[0]['keypoints'][6][:2]
        l_hip1 = cam1_keypoints[0]['keypoints'][11][:2]
        r_hip1 = cam1_keypoints[0]['keypoints'][12][:2]

with open(folder_path + POSE_MODEL + '/keypoints/cam_2_keypoints.pickle', 'rb') as f:
    cam2_keypoints = pickle.load(f)
    if POSE_MODEL == "OpenPose":
        l_shoulder2 = np.array(cam2_keypoints['people'][0]['pose_keypoints_2d'][15:17])
        r_shoulder2 = np.array(cam2_keypoints['people'][0]['pose_keypoints_2d'][6:8])
        l_hip2 = np.array(cam2_keypoints['people'][0]['pose_keypoints_2d'][36:38])
        r_hip2 = np.array(cam2_keypoints['people'][0]['pose_keypoints_2d'][27:29])
    else:
        l_shoulder2 = cam2_keypoints[0]['keypoints'][5][:2]
        r_shoulder2 = cam2_keypoints[0]['keypoints'][6][:2]
        l_hip2 = cam2_keypoints[0]['keypoints'][11][:2]
        r_hip2 = cam2_keypoints[0]['keypoints'][12][:2]

############ TSDF volume ############
start_time = time.time()
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=1 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

# show RGBD images from all the views
for i in range(len(camera_poses)):
    # print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(folder_path + "color_images/cam_{}.jpg".format(i+1))
    depth = o3d.io.read_image(folder_path + "depth_images/cam_{}.png".format(i+1))

    depth_image = np.asanyarray(depth)
    color_image = np.asanyarray(color)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                         interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    plt.imsave(folder_path + "depth_colormap/cam_{}.png".format(i+1), depth_colormap)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    plt.imshow(images)
    plt.show()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    cam_intr = depth_cam1_intr if i == 0 else depth_cam2_intr  # use depth camera's intrinsics

    intr = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=cam_intr[0,0],
        fy=cam_intr[1,1],
        cx=cam_intr[0,2],
        cy=cam_intr[1,2]
    )

    volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose))
print("Volume Integration: {:.3f} s".format(time.time() - start_time))

# point cloud generation
pcd = volume.extract_point_cloud()
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd])

coordinates = np.asarray(downpcd.points)
normals = np.asarray(downpcd.normals)

# save data, write pcd
o3d.io.write_point_cloud(folder_path + 'downpcd.ply', downpcd)
with open(folder_path + 'pcd_coordinates.pickle', 'wb') as f:
    pickle.dump(coordinates, f)

with open(folder_path + 'pcd_normals.pickle', 'wb') as f:
    pickle.dump(normals, f)

# Read pcd coordianates
with open(folder_path + 'pcd_coordinates.pickle', 'rb') as f:
    coordinates = pickle.load(f)

final_target_list = []  # the target coordinates used to navigate the robotic arm
position_data = {}  # store

if SCAN_POSE == 'front':
    target1 = target12_3D(l_shoulder1, l_shoulder2, r_shoulder1, r_shoulder2,
                          cam1_intr, cam2_intr, T_base_cam1, T_base_cam2, ratio_1=args.target1_r1,
                          ratio_2=args.target1_r2)
    target2 = target12_3D(l_shoulder1, l_shoulder2, r_shoulder1, r_shoulder2,
                          cam1_intr, cam2_intr, T_base_cam1, T_base_cam2, ratio_1=args.target2_r1,
                          ratio_2=args.target2_r2)

    target1_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target1)).squeeze()
    target1_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target1)).squeeze()

    target2_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target2)).squeeze()
    target2_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target2)).squeeze()

    draw_front_target_point(target1_2d_cam1, target1_2d_cam2, l_shoulder1, r_shoulder1, l_shoulder2, r_shoulder2, target=1)
    draw_front_target_point(target2_2d_cam1, target2_2d_cam2, l_shoulder1, r_shoulder1, l_shoulder2, r_shoulder2, target=2)

    final_target_list.append(target1)
    final_target_list.append(target2)

elif SCAN_POSE == 'side':
    target4 = target4_3D(r_shoulder1, r_shoulder2, r_hip1, r_hip2,
                         cam1_intr, cam2_intr, T_base_cam1, T_base_cam2, ratio_1=args.target4_r1,
                         ratio_2=args.target4_r2)

    target4_2d_cam1 = from_homog(cam1_intr @ T_base_cam1[0:3, :] @ to_homog(target4)).squeeze()
    target4_2d_cam2 = from_homog(cam2_intr @ T_base_cam2[0:3, :] @ to_homog(target4)).squeeze()

    draw_side_target_point(target4_2d_cam1, target4_2d_cam2, r_hip1, r_shoulder1, r_hip2, r_shoulder2)

    final_target_list.append(target4)

# # save results with initial ratio parameters
# with open(folder_path + POSE_MODEL + '/final_target.pickle', 'wb') as f:
#     pickle.dump(final_target_list, f)
#
# with open(folder_path + POSE_MODEL + '/position_data.pickle', 'wb') as f:
#     pickle.dump(position_data, f)

# # save results with optimized ratio parameters based on planar euclidean distance
# with open(folder_path + POSE_MODEL + '/final_target_opt.pickle', 'wb') as f:
#     pickle.dump(final_target_list, f)
#
# with open(folder_path + POSE_MODEL + '/position_data_opt.pickle', 'wb') as f:
#     pickle.dump(position_data, f)

# # save as temporary file for evaluation
# with open(folder_path + POSE_MODEL + '/tmp_target_test.pickle', 'wb') as f:
#     pickle.dump(final_target_list, f)
