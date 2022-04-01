import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt


class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
               "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


camera_poses = read_trajectory("RGBD/odometry.log")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=1 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(len(camera_poses)):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("RGBD/color/{}.jpg".format(i))
    depth = o3d.io.read_image("RGBD/depth/{}.png".format(i))

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

    # Show images
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    plt.imshow(images)
    plt.show()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    # volume.integrate(rgbd, intr, np.linalg.inv(camera_poses[i].pose)) # use the depth camera's intrinsics
    volume.integrate(rgbd,
                     o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                     np.linalg.inv(camera_poses[i].pose))
    break


# mesh generation:

# print("Extract a triangle mesh from the volume and visualize it.")
# mesh = volume.extract_triangle_mesh()
# o3d.io.write_triangle_mesh("ply/TSDF_volume_mesh.ply", mesh) # save the mesh as a ply file
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh],
#                                   front=[0.5297, -0.1873, -0.8272],
#                                   lookat=[0.0712, 0.0312, 0.7251],
#                                   up=[-0.0558, -0.9809, 0.1864],
#                                   zoom=0.47)

# o3d.visualization.draw_geometries([mesh])


# point cloud generation

pcd = volume.extract_point_cloud()
o3d.io.write_point_cloud("ply/TSDF_volume_pc.pcd", pcd)  # save the point cloud as a pcd file

# print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
# o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)


# o3d.visualization.draw_geometries([pc],
#                                   front = [0.5297, -0.1873, -0.8272],
#                                   lookat = [0.0712, 0.0312, 0.7251],
#                                   up = [-0.0558, -0.9809, 0.1864],
#                                   zoom = 0.47)



# visualize distribution of point cloud coordinates

# all_points = np.asarray(mesh.vertices) # pc.points / mesh.vertices
# all_points = np.asarray(pc.points)
# x_axis = all_points[:,0]
# y_axis = all_points[:,1]
# z_axis = all_points[:,2]
#
# print("mean of x-axis: ", np.mean(all_points[:,0]))
# print("median of x-axis: ", np.median(all_points[:,0]))
# print("std of x-axis: ", np.std(all_points[:,0]))
#
# print("mean of y-axis: ", np.mean(all_points[:,1]))
# print("median of y-axis: ", np.median(all_points[:,1]))
# print("std of y-axis: ", np.std(all_points[:,1]))
#
# print("mean of z-axis: ", np.mean(all_points[:,2]))
# print("median of z-axis: ", np.median(all_points[:,2]))
# print("std of z-axis: ", np.std(all_points[:,2]))
#
# fig, ax = plt.subplots(1,3, figsize=(13,5))
# for i,axis in enumerate(['x','y','z']):
#     ax[i].hist(all_points[:,i])
#     ax[i].set_title('{}-axis'.format(axis))
#
# plt.show()

# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('RealSense', np.array(depth))
# cv2.waitKey(100000)
