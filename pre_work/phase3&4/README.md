# Phase 3: Multiview RGBD Image Integration With IntelRealsense

# Functionalilty
- Capture multiple RGBD images with IntelRealsense D415 camera mounted on robot arm tool
- Compute transformation matrix between from other camera views to the first camera view
- Integrate all views together with Open3D TSDF volume integration

# Procedure
- Input all robot target poses in the program
- Clear content in RGBD/odometry.log
- RGBD image capture without Apriltag (multiview_integration_UR_base.py)
  - Input the measured camera pose in robot tool frame in the program
  - Run the script
- RGBD image capture with Apriltag (multiview_integration_Apriltag.py)
  - Do not need to measure camera pose in robot tool frame
  - Put one or more Apriltags of tagStandard41h12 family with tag size 4.8 cm at fixed positions
  - Make sure there is at least one Apriltag detectable in each camera view  
  - Run the script
- Multiview RGBD image integration (generate_pc_mesh.py)
  - Block/unblock different parts of code to generate and visualize the integrated point cloud / mesh of multiview RGBD images
  - Run the script
 

# Phase 4: UR navigation with target pose

# UR navigation with ViTPose procedure
1. Run `ViTPose_UR_collect_data.py` to take pictures from two views, and compute their extrinsic matrices.
2. Run the ViT pose estimation script with the two images to obtain the pose keypoints. 
3. Run `ViTPose_UR_move.py` to compute the target keypoint's coordinate and the robot arm pose in the base frame.

# Procedure of computing the target pose's rotation
1. Run `generate_pc_mesh.py` to generate the point cloud / mesh of the view.
2. Supply target point's coordinate, run `pc_normal_navigation.py` to obtain the target pose's rotation in the base frame.

# Pipeline
1. Capture 2D images in two views
2. Use ViTPose to compute 2D image coordinate of target scan point in each view. 
3. Compute 3D camera coordinate (in the 1st camera view) of the target scan point using triangulation.
4. Perform two-view integration to obtain the point cloud (in the 1st camera view)
5. Locate the 3D camera coordinate in the point cloud, and find corresponding normal estimation. 
6. Compute the target robot's pose in the base frame. 
7. Move the robot arm to the target pose