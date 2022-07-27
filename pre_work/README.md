# Project Preparatory Work
These works record our learning progress from Spring 2021 to Summer 2022. (Note that this is not the main folder of the project.) 

## Timeline
### Phase 1
- Camera calibration
- Apriltag detection & localization
  - Compare Apriltag detection results to measured values
### Phase 2
- UR3 robot arm navigation with Apriltag
  - Attach camera to robot arm
  - navigate robot arm tool to the location of Apriltag
### Phase 3
- Multiview RGBD image integration with IntelRealsense 
  - Use Apriltag to compute relative transformation between different camera views
### Phase 4
- UR3 robot arm navigation with 3D point cloud data
  - Move robot arm tool to a target point in given 3D point cloud
  - Coordinate conversion
    - Robot base -- Robot tool -- Camera -- Scene point
- Compute scan target locations using human pose estimation (HPE) models' results.
    

