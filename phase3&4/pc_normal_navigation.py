# Navigate UR3e with surface normal

import open3d as o3d
import math3d as m3d
import URBasic
import time
from utils.pose_conversion import *

#### UR3e Robot Configuration

ROBOT_IP = '169.254.147.11'  # real robot IP
ACCELERATION = 0.5  # robot acceleration
VELOCITY = 0.5  # robot speed value

robot_start_position = (np.radians(-339.5), np.radians(-110.55), np.radians(-34.35),
                        np.radians(-125.05), np.radians(89.56), np.radians(291.04))  # joint

# initialise robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

robot.init_realtime_control()
time.sleep(1)

# must be the same with tcp_pose_base at the first camera view
tcp_pose_base = (0.43414, 0.11073, 0.34996, 2.965, 0.326, 0.215)
robot.movej(pose=tcp_pose_base, a=ACCELERATION, v=VELOCITY)

# manually measured camera offset
t_cam_tcp = np.array([-0.041, -0.002, 0.02])
R_cam_tcp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cam_ang_tcp = np.array([0, 0, 0])

# read point cloud

path = "ply/TSDF_volume_pc.pcd"
pcd = o3d.io.read_point_cloud(path)

# print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
# o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([downpcd])
print("downpcd: ", downpcd)

coordinates = np.asarray(downpcd.points)
normals = np.asarray(downpcd.normals)

# read mesh

# path = "ply/TSDF_volume_mesh.ply"
# mesh = o3d.io.read_triangle_mesh(path)
# # o3d.visualization.draw_geometries([mesh])
#
# print("Computing normal and rendering it.")
# mesh.compute_vertex_normals()
# # print(np.asarray(mesh.triangle_normals))
# # o3d.visualization.draw_geometries([mesh])
#
# coordinates = np.asarray(mesh.vertices)
# normals = np.asarray(mesh.triangle_normals)

# lower_limit = np.array([-1.,-1.,-1.])   # neighbor hood of target point
# upper_limit = np.array([1.,1.,1.])
# mask = np.less(lower_limit[0], downpcd_points[:,0]) & np.less(downpcd_points[:,0], lower_limit[0]) & \
# np.less(lower_limit[1], downpcd_points[:,1]) & np.less(downpcd_points[:,1], lower_limit[1]) & \
# np.less(lower_limit[2], downpcd_points[:,2]) & np.less(downpcd_points[:,2], lower_limit[2])


# need to convert robot base coordinate to camera coordinate
init_estimate = np.array([0, 0, 0.78])  # an initial estimate of the target point
idx = np.argmin(np.square(coordinates - init_estimate).sum(axis=1))
print('closest point:', coordinates[idx, :])

target_normal = normals[idx]  # a unit vector
print("target normal: ", target_normal)

t_target_cam = coordinates[idx]
print("t_target_cam: ", t_target_cam)

# decode rotation from the target normal vector
R_target_cam = rotation_align(np.array([0, 0, 1]), -target_normal)
print("R_target_cam: \n", R_target_cam)

# retrieve the transformation from tcp to base
T_tcp_base = np.asarray(m3d.Transform(tcp_pose_base).get_matrix())
T_cam_tcp = np.vstack([np.hstack([R_cam_tcp, t_cam_tcp.reshape(-1, 1)]), np.array([0, 0, 0, 1])])
T_target_cam = np.vstack([np.hstack([R_target_cam, t_target_cam.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

T_target_base = T_tcp_base @ T_cam_tcp @ T_target_cam

target_6d_pose_base = m3d.Transform(T_target_base).get_pose_vector()
target_6d_pose_base[0:3] = tcp_pose_base[0:3]
print("Changed orientation:\n ", target_6d_pose_base)

robot.movej(pose=target_6d_pose_base, a=ACCELERATION, v=VELOCITY)
exit(0)
