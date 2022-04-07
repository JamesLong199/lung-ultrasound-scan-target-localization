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
 

