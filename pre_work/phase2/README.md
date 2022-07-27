# Phase 2: Robot Arm Navigation With Apriltag


# Functionality
- Detect Apriltag, visualize detection result 
- Compute Apriltag center location in robot base frame
- Move robot arm to the Apriltag center (tag should remain stationary throughout the process)
- Move robot arm back to start position 
- If there is no tag detected, robot arm stays put

# Procedure
- Use Apriltag tagStandard41h12 with tag size 4.8 cm
- Start robot, set robot to remote control mode
- Run python script, wait until robot reach start position
- Place tag under the camera view, hold the tag still
- Robot arm will reach tag center automatically after detection, which will be visualized in another window

# Caution
- Tag should be placed at positions reachable by the robot
- Two possible consequences if this is not followed
  - Robot pendent screen displays error message, robot does not move at all
  - Robot moves and subsequently comes to emergency stop 

# Demo Video
https://user-images.githubusercontent.com/66498825/162058424-806a339f-830f-44c0-942c-1f70bc83bebf.mp4

